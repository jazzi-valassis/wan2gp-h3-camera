"""Visual camera planning; generation continues through WanGP's native form."""

import hashlib
import json
import math
import os
import re
import subprocess
import threading
from pathlib import Path
import tempfile
import time

import gradio as gr
from PIL import Image

from shared.utils.frame_scheduler import normalize_frame_count
from shared.utils.plugins import WAN2GPPlugin
from shared.utils.prompt_parser import split_prompt_units
try:
    from shared.prompt_enhancer.images import window_contexts, resolve_injected_positions
except ImportError:
    window_contexts = None
    resolve_injected_positions = None
try:
    from shared.api import create_gradio_webui_session
except ImportError:
    create_gradio_webui_session = None

from .camera_plan import DEFAULT_PATH, PRESETS, compile_plan, validate_path, keyframe_frame, strip_camera_plan
from .editor import BRIDGE_JS, render_editor
from .image_anchors import merge_timed_images
from . import timing as camera_timing
from . import view_anchors


MAX_PLAN_BYTES = 64 * 1024
PLAN_FORMAT = "wangp-h3-camera-v1"
HOST_GLOBALS = ("get_state_model_type", "get_model_def", "get_model_family", "get_computed_fps",
                "refresh_prompt_labels", "refresh_video_length_label")
FORM_INPUTS = (
    "prompt", "state", "force_fps", "video_length", "video_guide", "video_source",
    "image_mode", "image_prompt_type", "image_start", "multi_prompts_gen_type",
    "sliding_window_size", "sliding_window_discard_last_frames",
    "sliding_window_trim_first_frames", "video_prompt_type", "image_refs", "image_end",
)
FORM_OUTPUTS = (
    "prompt", "multi_prompts_gen_type", "video_length", "sliding_window_size",
    "sliding_window_discard_last_frames", "sliding_window_trim_first_frames",
    "image_end", "image_prompt_type", "image_prompt_type_endcheckbox", "image_end_row",
)
LABEL_OUTPUTS = ("prompt", "wizard_prompt", "image_end", "prompt_info_label", "wizard_prompt_info_label")
ANCHOR_OUTPUTS = ("image_refs", "frames_positions", "video_prompt_type", "video_prompt_type_image_refs", "image_refs_row")
ANCHOR_INPUTS = ("path_json", "motion", "interpolation", "close_loop", *FORM_INPUTS, "stabilize_roll", "frames_positions")


class H3CameraPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "H3 Camera"
        self.version = "0.5.0"
        self.type = ["extension"]
        self.description = "Visual single-shot camera paths for MiniMax H3"

    def setup_ui(self):
        for name in dict.fromkeys((*FORM_INPUTS, *FORM_OUTPUTS, *LABEL_OUTPUTS, *ANCHOR_OUTPUTS, "model_choice_target")):
            self.request_component(name)
        for name in HOST_GLOBALS:
            self.request_global(name)
        self.request_global("PROMPT_TOOLS_ATTACH_JS")
        self.request_global("save_path")
        self.add_custom_js(f"({BRIDGE_JS})();")
        # WanGP's plugin generation API renders the view anchors in the live queue.
        self._wangp_session = None
        if create_gradio_webui_session is not None:
            try:
                self._wangp_session = create_gradio_webui_session(self)
            except Exception:
                self._wangp_session = None
        self.insert_after("prompt", self.build_panel)

    def post_ui_setup(self, components):
        if not hasattr(self, "_components"):
            self._components = dict(components)
        return {}

    def _supports_camera(self, model_type):
        if not model_type:
            return False
        definition = self.get_model_def(model_type) or {}
        return (self.get_model_family(model_type) == "minimax_h3"
                and not definition.get("audio_only", False)
                and not definition.get("control_net_weight_size", 0)
                and all(flag in definition.get("image_prompt_types_allowed", "") for flag in "SE"))

    def build_panel(self):
        components = self._components
        state = components.get("state")
        missing = [name for name in self.component_requests if name not in components]
        missing_globals = [name for name in HOST_GLOBALS if not callable(getattr(self, name, None))]
        if not isinstance(getattr(self, "PROMPT_TOOLS_ATTACH_JS", None), str):
            missing_globals.append("PROMPT_TOOLS_ATTACH_JS")
        model_type = self.get_state_model_type(state.value) if state is not None and not missing_globals else None
        with gr.Accordion("H3 Camera - single-shot planner", open=False,
                          visible=bool(missing or missing_globals) or self._supports_camera(model_type),
                          elem_id="h3-camera-panel") as panel:
            gr.Markdown("Plan an orbit, rise or push around your subject. Camera movement is guided by the prompt; "
                        "the diagram is a planning aid, not an exact camera simulation. Write the scene and audio "
                        "in the main prompt, then apply the path below.")
            if missing or missing_globals:
                details = []
                if missing:
                    details.append("form controls: " + ", ".join(missing))
                if missing_globals:
                    details.append("host functions: " + ", ".join(missing_globals))
                gr.Markdown("This WanGP build is missing required " + "; ".join(details) +
                            ". See the plugin README for the tested WanGP host and required APIs.")
                return panel
            with gr.Row():
                motion = gr.Dropdown(label="Subject and environment", choices=[
                    ("Allow motion from the scene prompt", "animated"),
                    ("Freeze the scene; move only the camera", "frozen"),
                ], value="animated")
                interpolation = gr.Dropdown(label="Movement between keyframes", choices=[
                    ("Smooth ease in / ease out", "smooth"), ("Linear / steady speed", "linear"),
                ], value="smooth")
            stabilize_roll = gr.Checkbox(label="Stabilize camera roll (keep upright; no banking)", value=True,
                                         info="Use pan and tilt to frame the subject. Turn off for intentional camera roll.")
            gr.HTML(render_editor())
            initial_timing = self.timing_value(
                components["video_length"].value, components["force_fps"].value,
                components["video_guide"].value, components["video_source"].value, model_type)
            # Keep the bridge fields mounted. Gradio visible=False removes their DOM nodes.
            gr.HTML("<style>#h3-camera-timing{display:none!important}</style>")
            timing = gr.Textbox(value=initial_timing, elem_id="h3-camera-timing", show_label=False)
            with gr.Accordion("Camera path JSON and saved plans", open=False):
                path = gr.Textbox(value=DEFAULT_PATH, label="Normalized keyframes", lines=6,
                                  elem_id="h3-camera-path")
                with gr.Row():
                    upload = gr.File(label="Load camera plan (.json)", file_types=[".json"], type="filepath")
                    with gr.Column():
                        save = gr.Button("Save camera plan")
                        download = gr.File(label="Saved camera plan", interactive=False)
                        export_state = gr.State(value=None, delete_callback=self.cleanup_export)
            close_loop = gr.Checkbox(label="Close loop: reuse the selected Start Image as the End Image",
                                     value=False)
            views_ready = self._wangp_session is not None and window_contexts is not None
            with gr.Accordion("Generated view anchors (recommended for orbits)", open=True):
                gr.Markdown("Text alone does not tell H3 which side to end on, so some seeds orbit past the subject's back. "
                            "With this on, **Apply** first renders a still of each keyframe view from the Start Image with "
                            "Qwen Image Edit Plus (2511), the Multiple-Angles LoRA and the 8-step Lightning LoRA, then injects "
                            "the views at their keyframe times. Keyframes need orbit angles within 5 degrees of a 45-degree step "
                            "and elevations near -30, 0, 30 or 60 degrees; other keyframes keep text guidance only. It needs an "
                            "H3 Ref2VA model and one Start Image; otherwise Apply uses text guidance and says why. Views take "
                            "about 15-30 seconds each plus a model switch (cached for the same image, view and seed), appear in "
                            "the gallery, and need the Media Generator tab focused while they render.")
                with gr.Row():
                    anchor_views = gr.Checkbox(label="Anchor keyframe views with generated images", value=views_ready,
                                               interactive=views_ready)
                    view_seed = gr.Number(label="View image seed", value=42, precision=0, minimum=0)
                views = gr.Gallery(label="Generated views", columns=4, height=260, interactive=False)
                if not views_ready:
                    gr.Markdown("This WanGP build lacks the plugin generation API or timed-image labels. Update WanGP to "
                                "generate view anchors.")
            with gr.Accordion("Automatic hold timing", open=True):
                gr.Markdown("H3 often arrives at a hold early or leaves it late. When a render of a camera plan with one hold "
                            "appears in the output folder, this measures the still interval from background motion, retimes "
                            "the clip so the hold sits exactly on the planned frames, re-measures it and saves a `_timed` copy "
                            "next to the original (same duration, FPS and size; audio retimed with pitch kept). The original "
                            "is never changed. Clips that need more than a 2x speed change while the camera moves are left as "
                            "they are. It runs while this page is open.")
                auto_timing = gr.Checkbox(label="Correct hold timing automatically after each render", value=True)
                auto_video = gr.Video(label="Latest render with corrected hold timing", interactive=False)
                auto_status = gr.Markdown()
                auto_shown = gr.State(value=0)
                auto_timer = gr.Timer(5)
            with gr.Accordion("Image anchors for a camera hold", open=False):
                gr.Markdown("For H3 Ref2VA, choose an image showing the view the camera should hold. Use the same aspect ratio "
                            "as the Start Image: native injection takes its canvas from the first injected image. The button places it at both ends "
                            "of one interior hold using native Inject Frames, retains your other references, and updates "
                            "their Picture numbers. A Start Image is required. The anchor is a generated-view guide, "
                            "not an exact 3D camera lock. It also constrains the subject's pose at the two anchor frames.")
                hold_image = gr.Image(label="Image of the held camera view", type="pil")
                hold_keyframe = gr.Number(label="Hold starts at keyframe", value=2, precision=0, minimum=1, maximum=23)
                anchor_hold = gr.Button("Anchor hold and apply camera path", interactive=window_contexts is not None)
                if window_contexts is None:
                    gr.Markdown("This WanGP build lacks the native timed-image label helper. Update WanGP to enable image anchoring.")
            with gr.Accordion("Movement image checkpoints", open=False):
                gr.Markdown("Guide the camera between keyframes with an image at a chosen time. For early arrival, "
                            "supply a view still approaching the stop shortly before the hold. For a late rise, "
                            "supply an already elevated view shortly after the hold. The camera is asked to pass through "
                            "this view without another stop. Use the Start Image's aspect ratio. Existing holds, "
                            "image references, path times and clip duration are retained.")
                checkpoint_image = gr.Image(label="View during the camera move", type="pil")
                checkpoint_time = gr.Number(label="Checkpoint time (seconds)", value=6.5, precision=6, minimum=0)
                anchor_checkpoint = gr.Button("Add or replace movement checkpoint and apply", interactive=window_contexts is not None)
                with gr.Accordion("Extract a checkpoint from an earlier render", open=False):
                    checkpoint_video = gr.File(label="Earlier render", file_types=["video"], type="filepath")
                    checkpoint_source_frame = gr.Number(label="Source frame (1-based)", value=1, precision=0, minimum=1)
                    extract_checkpoint = gr.Button("Use this video frame as the checkpoint image")
                    extraction_status = gr.Markdown()
            with gr.Accordion("Correct timing in a generated clip", open=False):
                gr.Markdown("Align a reviewed clip's arrival and departure frames to the selected hold in the camera path. "
                            "The clip must already contain the wanted views and a stationary hold. This exports a new video "
                            "at the same duration, FPS and size. Subject motion is retimed too. The original clip is preserved.")
                timing_video = gr.Video(label="Generated clip to retime", sources=["upload"])
                timing_keyframe = gr.Number(label="Hold starts at camera keyframe", value=2, precision=0, minimum=1, maximum=23)
                inspect_timing = gr.Button("Suggest arrival and departure frames")
                gr.Markdown("Suggestions use background motion. Review the clip and adjust the frame numbers when needed.")
                with gr.Row():
                    observed_arrival = gr.Number(label="Observed arrival frame (1-based)", value=None, precision=0, minimum=1)
                    observed_departure = gr.Number(label="Observed departure frame (1-based)", value=None, precision=0, minimum=1)
                audio_timing = gr.Dropdown(label="Audio timing", choices=[("Retime with video (preserve pitch)","retime"),
                    ("Keep original audio timeline","preserve"),("Remove audio","mute")], value="retime")
                frame_sampling = gr.Dropdown(label="Frame sampling", choices=[("Nearest frame (can repeat frames)","nearest"),
                    ("Blend neighboring frames (can ghost)","blend")], value="nearest")
                export_timing = gr.Button("Export clip with corrected timing")
                timing_status = gr.Markdown()
                retimed_video = gr.Video(label="Clip with corrected timing", interactive=False)
                timing_report = gr.File(label="Timing report", interactive=False)
                with gr.Accordion("Automatically check and correct candidates", open=False):
                    gr.Markdown("Upload one to three renders in preference order. Try each until a corrected clip passes "
                                "independent motion and hold-timing checks. Failed candidates are reported without returning "
                                "a video. Camera angles and roll still need visual review. This does not launch new generations.")
                    candidates = gr.File(label="Candidate renders (maximum 3)",file_count="multiple",file_types=["video"],type="filepath")
                    verify_candidates = gr.Button("Check candidates and return verified timing")
                    verified_video = gr.Video(label="Clip passing timing checks",interactive=False)
                    verification_report = gr.File(label="Candidate verification report",interactive=False)
                    verification_status = gr.Markdown()
            gr.Markdown("Timing follows **Number of Frames** and **Frames per Second** in the generation form. "
                        "Apply again after changing the path, timing or scene. Applying uses one prompt and one "
                        "window (up to 481 frames), with no trimmed frames. A closed loop requires one active "
                        "Start Image and a path returning to its origin; matching endpoints cannot guarantee an orbit.")
            with gr.Row():
                preview = gr.Button("Preview camera prompt")
                apply = gr.Button("Apply camera path to generation form", variant="primary")
            table = gr.Dataframe(headers=["Keyframe", "Time (s)", "Orbit angle from start (degrees)", "Elevation (degrees)", "Distance"],
                                 datatype=["number"] * 5, interactive=False)
            compiled = gr.Textbox(label="Camera prompt preview", lines=6, interactive=False)
            status = gr.Markdown()
            # Changes when Apply finishes; drives the native label refresh after the queue-backed Apply.
            gr.HTML("<style>#h3-camera-applied{display:none!important}</style>")
            applied = gr.Textbox(elem_id="h3-camera-applied", show_label=False)
            args = [path, motion, interpolation, close_loop, *[components[name] for name in FORM_INPUTS], stabilize_roll,
                    components["frames_positions"]]
            preview.click(self.preview, inputs=args, outputs=[table, compiled, status], api_name="h3_camera_preview")
            label_refresh = dict(fn=self.refresh_prompt_labels,
                                 inputs=[components[name] for name in ("state", "multi_prompts_gen_type", "image_mode")],
                                 outputs=[components[name] for name in LABEL_OUTPUTS], show_progress="hidden", api_name=False)
            length_refresh = dict(fn=self.refresh_video_length_label,
                                  inputs=[components[name] for name in ("state", "video_length", "force_fps", "video_guide", "video_source")],
                                  outputs=components["video_length"], show_progress="hidden", api_name=False)
            smart_apply = False
            if self._wangp_session is not None:
                try:
                    with self._wangp_session.plugin_ui_context():
                        apply.click(self.apply_camera, inputs=[anchor_views, view_seed, *args],
                                    outputs=[*[components[name] for name in FORM_OUTPUTS], table, compiled, status,
                                             *[components[name] for name in ANCHOR_OUTPUTS], views, applied],
                                    api_name="h3_camera_apply")
                    smart_apply = True
                except RuntimeError:
                    pass  # No main WebUI queue bridge (older hosts, tests): Apply stays text-only.
            if smart_apply:
                # The queue-backed click finishes in a follow-up event; refresh native labels when it does.
                applied.change(**label_refresh).then(**length_refresh).then(
                    fn=None, inputs=None, outputs=None, js=self.PROMPT_TOOLS_ATTACH_JS, api_name=False)
            else:
                anchor_views.interactive = False
                apply.click(self.apply_plan, inputs=args, outputs=[*[components[name] for name in FORM_OUTPUTS],
                            table, compiled, status], api_name="h3_camera_apply").success(**label_refresh).success(
                    **length_refresh).then(fn=None, inputs=None, outputs=None, js=self.PROMPT_TOOLS_ATTACH_JS, api_name=False)
            auto_timer.tick(self.auto_timing_tick, inputs=[auto_timing, auto_shown],
                            outputs=[auto_video, auto_status, auto_shown], show_progress="hidden", api_name=False)
            anchor_hold.click(self.apply_hold_image, inputs=[hold_image, hold_keyframe, *args],
                              outputs=[*[components[name] for name in FORM_OUTPUTS], table, compiled, status,
                                       *[components[name] for name in ANCHOR_OUTPUTS]], api_name="h3_camera_anchor_hold").success(
                self.refresh_prompt_labels,
                inputs=[components[name] for name in ("state", "multi_prompts_gen_type", "image_mode")],
                outputs=[components[name] for name in LABEL_OUTPUTS], show_progress="hidden", api_name=False,
            ).success(
                self.refresh_video_length_label,
                inputs=[components[name] for name in ("state", "video_length", "force_fps", "video_guide", "video_source")],
                outputs=components["video_length"], show_progress="hidden", api_name=False,
            ).then(fn=None, inputs=None, outputs=None, js=self.PROMPT_TOOLS_ATTACH_JS, api_name=False)
            anchor_checkpoint.click(self.apply_checkpoint_image, inputs=[checkpoint_image, checkpoint_time, *args],
                                    outputs=[*[components[name] for name in FORM_OUTPUTS], table, compiled, status,
                                             *[components[name] for name in ANCHOR_OUTPUTS]], api_name="h3_camera_anchor_checkpoint").success(
                self.refresh_prompt_labels,
                inputs=[components[name] for name in ("state", "multi_prompts_gen_type", "image_mode")],
                outputs=[components[name] for name in LABEL_OUTPUTS], show_progress="hidden", api_name=False,
            ).success(
                self.refresh_video_length_label,
                inputs=[components[name] for name in ("state", "video_length", "force_fps", "video_guide", "video_source")],
                outputs=components["video_length"], show_progress="hidden", api_name=False,
            ).then(fn=None, inputs=None, outputs=None, js=self.PROMPT_TOOLS_ATTACH_JS, api_name=False)
            upload.change(self.load_plan, inputs=upload, outputs=[path, motion, interpolation, close_loop, stabilize_roll])
            extract_checkpoint.click(self.extract_checkpoint, inputs=[checkpoint_video, checkpoint_source_frame],
                                     outputs=[checkpoint_image, extraction_status], api_name="h3_camera_extract_checkpoint")
            timing_video.change(lambda: (None,None,""), inputs=None, outputs=[observed_arrival,observed_departure,timing_status], api_name=False)
            inspect_timing.click(self.inspect_clip_timing, inputs=[timing_video,path,timing_keyframe],
                                 outputs=[observed_arrival,observed_departure,timing_status], api_name="h3_camera_inspect_timing")
            export_timing.click(self.correct_clip_timing,
                                inputs=[timing_video,path,timing_keyframe,observed_arrival,observed_departure,audio_timing,frame_sampling],
                                outputs=[retimed_video,timing_report,timing_status], api_name="h3_camera_correct_timing")
            verify_candidates.click(self.verify_clip_candidates,
                                    inputs=[candidates,path,timing_keyframe,audio_timing,frame_sampling],
                                    outputs=[verified_video,verification_report,verification_status],api_name="h3_camera_verify_candidates")
            save.click(self.save_plan_for_session, inputs=[path, motion, interpolation, close_loop, export_state, stabilize_roll],
                       outputs=[download, export_state])

        timing_inputs = [*[components[name] for name in ("video_length", "force_fps", "video_guide", "video_source", "state")], interpolation]
        for name in ("video_length", "force_fps", "video_guide", "video_source"):
            components[name].change(self.update_timing, inputs=timing_inputs, outputs=timing,
                                    queue=False, show_progress="hidden")
        interpolation.change(self.update_timing, inputs=timing_inputs, outputs=timing,
                             queue=False, show_progress="hidden")
        target = components["model_choice_target"]
        target.change(self.update_visibility, inputs=[target, *timing_inputs[:4], interpolation], outputs=[panel, timing],
                      queue=False, show_progress="hidden")
        return panel

    def timing_value(self, frame_count, force_fps, video_guide, video_source, model_type, interpolation="smooth"):
        if not self._supports_camera(model_type):
            return "{}"
        try:
            fps = float(self.get_computed_fps(force_fps or "", model_type, video_guide, video_source))
            frames = float(frame_count)
            if not math.isfinite(fps) or fps <= 0 or not math.isfinite(frames) or frames <= 0:
                return "{}"
            definition = self.get_model_def(model_type)
            frames = normalize_frame_count(math.ceil(frames), max(22, definition.get("frames_minimum", 107)),
                                           definition.get("frames_steps", 17), definition.get("frames_offset", 5))
            return json.dumps({"frame_count": frames, "fps": fps, "seconds": max(0, frames - 1) / fps,
                               "easing": interpolation})
        except (TypeError, ValueError, OSError):
            return "{}"

    def update_timing(self, frame_count, force_fps, video_guide, video_source, state, interpolation="smooth"):
        return self.timing_value(frame_count, force_fps, video_guide, video_source, self.get_state_model_type(state), interpolation)

    def update_visibility(self, target, frame_count, force_fps, video_guide, video_source, interpolation="smooth"):
        # The model target arrives before the state update and includes a timestamp.
        model_type = str(target or "").split("|", 1)[0].strip()
        return (gr.update(visible=self._supports_camera(model_type)),
                self.timing_value(frame_count, force_fps, video_guide, video_source, model_type, interpolation))

    def _compile(self, path_json, motion, interpolation, close_loop, prompt, state,
                 force_fps, video_length, video_guide, video_source, image_mode,
                 image_prompt_type, image_start, multi_prompts_gen_type, sliding_window_size,
                 sliding_window_discard_last_frames, sliding_window_trim_first_frames, video_prompt_type, image_refs, image_end,
                 stabilize_roll=True, frames_positions=""):
        model_type = self.get_state_model_type(state)
        if not self._supports_camera(model_type):
            raise gr.Error("Select a MiniMax H3 FL2VA or Ref2VA video model before applying a camera path.")
        if int(image_mode or 0):
            raise gr.Error("Select Text to Video or Image to Video; camera paths need a video timeline.")
        if video_source is not None or any(flag in (image_prompt_type or "") for flag in "VL"):
            raise gr.Error("Camera planning starts a single new shot. Turn off Video Continuation and use a Start Image or reference instead.")
        if "G" in (video_prompt_type or ""):
            raise gr.Error("Turn off Control Video editing before applying a camera path. Reference images, video and audio can remain enabled.")
        if multi_prompts_gen_type in ("W", "PW") and len(split_prompt_units(prompt or "", multi_prompts_gen_type)) > 1:
            raise gr.Error("The prompt currently contains multiple sliding-window shots. Keep one scene and choose 'All the Lines are Part of the Same Prompt' before applying a camera path.")
        if motion not in ("animated", "frozen"):
            raise gr.Error("Choose whether the subject can move or the scene is frozen.")
        definition = self.get_model_def(model_type)
        try:
            requested = float(video_length)
            if not math.isfinite(requested) or requested <= 0:
                raise ValueError("The generation frame count must be positive and finite.")
            frames = normalize_frame_count(math.ceil(requested), max(22, definition.get("frames_minimum", 107)),
                                           definition.get("frames_steps", 17), definition.get("frames_offset", 5))
            window_def = definition.get("sliding_window_defaults", {})
            maximum = min(481, int(window_def.get("window_max", 481)))
            if frames > maximum:
                raise ValueError(f"This single-shot camera planner supports up to {maximum} frames. Reduce Number of Frames before applying; use H3 Multishot for longer scripts.")
            fps = float(self.get_computed_fps(force_fps or "", model_type, video_guide, video_source))
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("Choose a positive finite frame rate before applying a camera path.")
            has_start = "S" in (image_prompt_type or "") and isinstance(image_start, list) and len(image_start) == 1 and image_start[0] is not None
            if close_loop and not has_start:
                raise ValueError("For a closed loop, select Start Image and provide exactly one image in the native Start Image gallery.")
            has_end = "E" in (image_prompt_type or "") and isinstance(image_end, list) and len(image_end) == 1 and image_end[0] is not None
            if close_loop and not has_end and "I" in (video_prompt_type or "") and image_refs:
                raise ValueError("Adding an End Image changes H3 reference numbering: it becomes <Picture 2>, shifting other images. Enable an End Image in the native form first, update the scene's reference labels, then apply the closed camera loop.")
            anchors = self.native_image_anchors(frames, fps, image_start if has_start else None,
                                               image_end if has_end else None, image_refs, frames_positions,
                                               video_prompt_type)
            result = compile_plan(path_json, prompt=prompt, frame_count=frames, fps=fps,
                                  frozen=motion == "frozen", close_loop=bool(close_loop),
                                  reference_mode=bool(definition.get("reference_image_enabled")),
                                  has_start_image=has_start, interpolation=interpolation, stabilize_roll=stabilize_roll,
                                  image_anchors=anchors)
        except (ValueError, TypeError, OverflowError) as error:
            raise gr.Error(str(error)) from error
        result["sliding_window_size"] = max(frames, int(window_def.get("window_min", 124)))
        result["sliding_window_discard_last_frames"] = 0
        result["sliding_window_trim_first_frames"] = 0
        result["multi_prompts_gen_type"] = "FG"
        result["image_start"] = image_start
        result["image_prompt_type"] = image_prompt_type or ""
        return result

    @staticmethod
    def _summary(result, applied=False):
        text = (f"**{result['frame_count']} frames at {result['fps']:g} fps "
                f"({result['frame_count'] / result['fps']:.3f} seconds).** "
                f"{result['summary']} One prompt, one window ({result['sliding_window_size']} frames), "
                "Discard Last Frames = 0 and Trim First Frames = 0.")
        if result["close_loop"]:
            text += " The Start Image will also condition the last frame, replacing the End Image."
        if applied:
            text += " **Applied. Use Generate or Add to Queue on this page.**"
        return text

    def preview(self, *args):
        result = self._compile(*args)
        return result["rows"], result["prompt"], self._summary(result)

    def apply_plan(self, *args):
        result = self._compile(*args)
        return self._apply_result(result)

    def _apply_result(self, result):
        changes = {name: gr.update(value=result[name]) for name in FORM_OUTPUTS[:6] if name in result}
        changes["video_length"] = gr.update(value=result["frame_count"])
        if result["close_loop"]:
            changes.update({"image_end": gr.update(value=result["image_start"]),
                            "image_prompt_type": gr.update(value=result["image_prompt_type"].replace("E", "") + "E"),
                            "image_prompt_type_endcheckbox": gr.update(value=True),
                            "image_end_row": gr.update(visible=True)})
        return (*[changes.get(name, gr.update()) for name in FORM_OUTPUTS],
                result["rows"], result["prompt"], self._summary(result, applied=True))

    @staticmethod
    def native_image_anchors(frames, fps, image_start, image_end, image_refs, positions, video_prompt_type):
        if "F" not in (video_prompt_type or ""):
            return []
        if window_contexts is None:
            raise ValueError("This WanGP build lacks timed-image label support. Update WanGP or turn off injected frames before applying.")
        windows = [dict(frame_num=frames, overlap_frames=0, discard_last_frames=0, trim_last_frames=0)]
        context = window_contexts(windows, image_start, image_end, image_refs, fps=float(fps),
                                  window_size=frames, positions=positions or "", picture_labels=True)[0]
        anchors = []
        for label in context.labels:
            end = re.fullmatch(r"<Picture (\d+)>: end image", label)
            if end:
                anchors.append({"frame": frames - 1, "picture": int(end[1])})
                continue
            match = re.fullmatch(r"<Picture (\d+)>: injected frame at ([0-9.e+-]+)s", label)
            if match:
                anchors.append({"frame": int(math.floor(float(match[2]) * float(fps) + 0.5)),
                                "picture": int(match[1])})
        return anchors

    def _anchor_form(self, args):
        if len(args) != len(ANCHOR_INPUTS):
            raise gr.Error("Reload the camera panel before adding image anchors.")
        values = dict(zip(ANCHOR_INPUTS, args))
        if window_contexts is None:
            raise gr.Error("Update WanGP to enable the native timed-image label helper.")
        definition = self.get_model_def(self.get_state_model_type(values["state"])) or {}
        if not definition.get("reference_image_enabled") or not definition.get("custom_frames_injection"):
            raise gr.Error("Image anchoring requires an H3 Ref2VA model with native Inject Frames enabled.")
        initial = self._compile(*args)
        if "S" not in initial["image_prompt_type"] or not isinstance(initial["image_start"], list) or len(initial["image_start"]) != 1 or initial["image_start"][0] is None:
            raise gr.Error("Add exactly one active Start Image before adding image anchors.")
        if "E" in initial["image_prompt_type"] and values["image_end"] and len(values["image_end"]) != 1:
            raise gr.Error("Use at most one active End Image when adding image anchors.")
        if values["image_refs"] and "I" not in (values["video_prompt_type"] or ""):
            raise gr.Error("Enable the existing Reference Images first so their Picture labels can be preserved.")
        return values, initial

    @staticmethod
    def _injection_positions(values, initial):
        if "F" not in (values["video_prompt_type"] or ""):
            return []
        tokens = (values["frames_positions"] or "").replace(",", " ").split()
        if not tokens or any(token.upper() == "X" for token in tokens):
            raise ValueError("For this single-shot image helper, use explicit frame positions or one L endpoint; remove X window slots first.")
        if len(tokens) > len(values["image_refs"] or []):
            raise ValueError("Each existing injected frame needs an image before more anchors can be added.")
        frames = initial["frame_count"]
        positions = resolve_injected_positions(" ".join(tokens), len(values["image_refs"] or []),
            windows=[dict(frame_num=frames, overlap_frames=0, discard_last_frames=0, trim_last_frames=0)],
            source_frames=0, source_overlap=0, window_size=frames, discard_frames=0, reuse_frames=0, alignment_shift=0)
        if any(not 0 <= frame < frames for frame in positions):
            raise ValueError("All existing injected frames must be inside this single shot before adding image anchors.")
        return positions

    @staticmethod
    def _picture_numbers(identities, positions, initial, values):
        has_end = "E" in initial["image_prompt_type"] and bool(values["image_end"]) and values["image_end"][0] is not None
        frames = initial["frame_count"]
        context = window_contexts([dict(frame_num=frames, overlap_frames=0, discard_last_frames=0, trim_last_frames=0)],
            "start", ["end"] if has_end else None, identities, fps=initial["fps"], window_size=frames,
            positions=" ".join(str(frame+1) for frame in positions), picture_labels=True)[0]
        return {identity: int(re.match(r"<Picture (\d+)>", label)[1])
                for identity, label in zip(context.images, context.labels)}

    def _apply_image_updates(self, values, initial, positions, additions, description):
        refs = list(values["image_refs"] or [])
        before = self._picture_numbers(list(range(len(refs))), positions, initial, values)
        refs, positions, identities = merge_timed_images(refs, positions, additions)
        after = self._picture_numbers(identities, positions, initial, values)
        renumber = {number: after[identity] for identity, number in before.items() if identity in after}
        prompt = strip_camera_plan(values["prompt"] or "")
        prompt = re.sub(r"<Picture\s+([1-9]\d*)>",
                        lambda match: f"<Picture {renumber[int(match[1])]}>" if int(match[1]) in renumber
                        and renumber[int(match[1])] != int(match[1]) else match[0], prompt)
        flags = values["video_prompt_type"] or ""
        native_positions = " ".join(str(frame+1) for frame in positions)
        values = dict(values, prompt=prompt, image_refs=refs, frames_positions=native_positions,
                      video_prompt_type="".join(flag for flag in flags if flag not in "FIK") + "FI")
        result = self._compile(*[values[name] for name in ANCHOR_INPUTS])
        result["summary"] += description + " Existing images are retained with their native Picture labels updated."
        return (*self._apply_result(result), gr.update(value=refs), gr.update(value=native_positions, visible=True),
                gr.update(value=values["video_prompt_type"]), gr.update(value="FI"), gr.update(visible=True))

    def apply_hold_image(self, image, hold_keyframe, *args):
        values, initial = self._anchor_form(args)
        if image is None:
            raise gr.Error("Choose an image showing the camera view to hold.")
        try:
            number = float(hold_keyframe)
            if not math.isfinite(number) or not number.is_integer() or not 1 <= number < len(initial["path"]):
                raise ValueError("Choose the starting keyframe of an existing hold.")
            left, right = initial["path"][int(number)-1:int(number)+1]
            if any(left[field] != right[field] for field in ("azimuth", "elevation", "distance")):
                raise ValueError("The chosen keyframe must be followed by an identical camera pose at a later time.")
            first, last = keyframe_frame(left, initial["frame_count"]), keyframe_frame(right, initial["frame_count"])
            if first == 0 or last == initial["frame_count"]-1:
                raise ValueError("Choose an interior hold; use native Start/End Image controls for endpoint holds.")
            if first == last:
                raise ValueError("This hold is shorter than one displayed frame. Increase its duration before adding image anchors.")
            positions = self._injection_positions(values, initial)
            if "F" in (values["video_prompt_type"] or "") and positions[:2] != [first, last]:
                raise ValueError("Existing injected frames use a different timeline. Preserve or clear those inputs before using this single-hold helper.")
            return self._apply_image_updates(values, initial, positions, {first: image, last: image},
                f" The held-view image is injected at frames {first+1} {last+1} (1-based).")
        except (ValueError, TypeError, OverflowError) as error:
            raise gr.Error(str(error)) from error

    def apply_checkpoint_image(self, image, seconds, *args):
        values, initial = self._anchor_form(args)
        if image is None:
            raise gr.Error("Choose an image showing the camera view during the move.")
        try:
            if isinstance(seconds, bool):
                raise ValueError("Checkpoint time must be a number of seconds.")
            seconds = float(seconds)
            if not math.isfinite(seconds) or not 0 < seconds < initial["end_seconds"]:
                raise ValueError("Choose a checkpoint time strictly inside this clip.")
            frame = int(math.floor(seconds * initial["fps"] + 0.5))
            path = initial["path"]
            segment = next(((left, right) for left, right in zip(path, path[1:])
                if keyframe_frame(left, initial["frame_count"]) < frame < keyframe_frame(right, initial["frame_count"])), None)
            if segment is None or all(segment[0][field] == segment[1][field] for field in ("azimuth", "elevation", "distance")):
                raise ValueError("Place a movement checkpoint between keyframes in a moving segment, outside the hold.")
            positions = self._injection_positions(values, initial)
            return self._apply_image_updates(values, initial, positions, {frame: image},
                f" Movement checkpoint at {frame / initial['fps']:.6f}s (frame {frame+1}, 1-based). The camera path and its times are unchanged.")
        except (ValueError, TypeError, OverflowError) as error:
            raise gr.Error(str(error)) from error

    @staticmethod
    def _image_file(item):
        """Return (path, size) for a native gallery item: a path, (path, caption), file dict or PIL image."""
        if isinstance(item, (tuple, list)) and item:
            item = item[0]
        if isinstance(item, dict):
            item = item.get("path") or (item.get("image") or {}).get("path")
        if isinstance(item, (str, Path)) and Path(item).is_file():
            with Image.open(item) as image:
                return str(item), image.size
        if isinstance(item, Image.Image):
            target = Path(tempfile.mkdtemp(prefix="h3_camera_views_")) / "start.png"
            item.save(target)
            return str(target), item.size
        raise gr.Error("The Start Image could not be read for view generation.")

    @staticmethod
    def _as_updates(values):
        """Streamed results are diffs against the previous yield, so every output must stay an update dict:
        a raw value following gr.update() becomes a patch the browser cannot apply."""
        return tuple(value if isinstance(value, dict) and value.get("__type__") == "update" else gr.update(value=value)
                     for value in values)

    def _views_unavailable(self, use_views, args):
        """Why view anchors cannot be added to this form, or None."""
        if not use_views:
            return "view anchors are turned off."
        if self._wangp_session is None or window_contexts is None:
            return "this WanGP build lacks the plugin generation API or timed-image labels."
        try:
            _, initial = self._anchor_form(args)
        except gr.Error as error:
            return str(error.message if hasattr(error, "message") else error)
        targets, _ = view_anchors.anchor_targets(initial["path"], initial["frame_count"], keyframe_frame)
        if not targets:
            return ("no keyframe matches a view the Multiple-Angles LoRA can render (orbit within 5 degrees of a "
                    "45-degree step, elevation near -30, 0, 30 or 60 degrees).")
        return None

    def apply_camera(self, use_views, seed, *args):
        """Apply the camera path, first anchoring generated keyframe views when this form allows it."""
        # WanGP pumps the queue only for handlers whose own code uses the plugin session.
        session = self._wangp_session
        anchors = len(ANCHOR_OUTPUTS)
        text_only = list(self.apply_plan(*args))  # Validates the plan; invalid input raises before any change.
        reason = self._views_unavailable(use_views, args)
        if reason:
            text_only[-1] += f" Text guidance only: {reason}"
            yield self._as_updates((*text_only, *[gr.update()] * anchors, gr.update(), str(time.time_ns())))
            return
        for final, updates in self._view_anchor_steps(session, seed, args):
            yield (*updates, gr.update(value=str(time.time_ns())) if final else gr.update())

    @staticmethod
    def _item_path(item):
        if isinstance(item, (tuple, list)) and item:
            item = item[0]
        if isinstance(item, dict):
            item = item.get("path") or (item.get("image") or {}).get("path")
        return str(item) if isinstance(item, (str, Path)) else None

    def _without_generated_views(self, values, initial):
        """Drop views injected by an earlier Apply so a changed path never keeps stale anchors."""
        refs = list(values["image_refs"] or [])
        flags = values["video_prompt_type"] or ""
        positions = self._injection_positions(values, initial)
        stale = [index for index in range(len(positions)) if view_anchors.is_generated_view(self._item_path(refs[index]))]
        if not stale:
            return values
        before = self._picture_numbers(list(range(len(refs))), positions, initial, values)
        keep = [index for index in range(len(refs)) if index not in stale]
        kept_positions = [frame for index, frame in enumerate(positions) if index not in stale]
        cleaned = dict(values, image_refs=[refs[index] for index in keep],
                       frames_positions=" ".join(str(frame + 1) for frame in kept_positions),
                       video_prompt_type=flags if kept_positions else flags.replace("F", ""))
        after = self._picture_numbers(keep, kept_positions, initial, cleaned)
        renumber = {before[index]: after[index] for index in keep}
        cleaned["prompt"] = re.sub(r"<Picture\s+([1-9]\d*)>",
                                   lambda match: f"<Picture {renumber.get(int(match[1]), int(match[1]))}>",
                                   strip_camera_plan(values["prompt"] or ""))
        return cleaned

    def _view_anchor_steps(self, session, seed, args):
        """Yield (final, updates): progress messages, then the anchored and applied form."""
        count = len(FORM_OUTPUTS) + 3 + len(ANCHOR_OUTPUTS) + 1
        status_index = len(FORM_OUTPUTS) + 2

        def status_only(message):
            updates = [gr.update()] * count
            updates[status_index] = gr.update(value=message)
            return tuple(updates)

        values, initial = self._anchor_form(args)
        try:
            seed = int(seed)
            if seed < 0:
                raise ValueError("Use a view image seed of 0 or more.")
            values = self._without_generated_views(values, initial)
            positions = self._injection_positions(values, initial)
        except (ValueError, TypeError, OverflowError) as error:
            raise gr.Error(str(error)) from error
        targets, skipped = view_anchors.anchor_targets(initial["path"], initial["frame_count"], keyframe_frame)
        image_path, size = self._image_file(initial["image_start"][0])
        resolution = view_anchors.view_resolution(*size)
        image_hash = hashlib.sha1(Path(image_path).read_bytes()).hexdigest()
        cache = self.__dict__.setdefault("_view_cache", {})
        prompts = list(dict.fromkeys(targets.values()))
        images = {}
        for prompt in prompts:
            cached = cache.get((image_hash, prompt, seed, resolution))
            if cached and Path(cached).is_file():
                images[prompt] = cached
        missing = [prompt for prompt in prompts if prompt not in images]
        if missing:
            tasks = view_anchors.view_tasks(missing, image_path, resolution, seed)
            yield False, status_only(f"Generating {len(missing)} view image(s) with Qwen Image Edit Plus (2511) in the WanGP queue...")
            job = session.submit(tasks)
            started = last = time.time()
            while not job.done:
                time.sleep(0.5)
                if time.time() - last >= 5:
                    last = time.time()
                    yield False, status_only(f"Generating {len(missing)} view image(s)... {int(last - started)}s elapsed.")
            result = job.result(timeout=1.0)
            if result.cancelled:
                raise gr.Error("View generation was cancelled; the generation form is unchanged.")
            files = list(result.generated_files or [])
            if not result.success or len(files) != len(missing):
                errors = "; ".join(error.message for error in result.errors) or "the queue returned no images"
                raise gr.Error(f"View generation failed ({errors}); the generation form is unchanged.")
            for prompt, file in zip(missing, files):
                images[prompt] = cache[(image_hash, prompt, seed, resolution)] = file
        frames = sorted(targets)
        description = (f" Generated view anchors at frames {' '.join(str(frame + 1) for frame in frames)} (1-based), seed {seed}."
                       + view_anchors.describe(initial["path"], initial["frame_count"], keyframe_frame, targets, skipped))
        updates = self._apply_image_updates(values, initial, positions,
                                            {frame: images[targets[frame]] for frame in frames}, description)
        yield True, self._as_updates((*updates, [(images[prompt], prompt.replace("<sks> ", "")) for prompt in prompts]))

    def auto_timing_tick(self, enabled, shown):
        """Timer callback: start corrections for new renders and show the latest result once."""
        folder = getattr(self, "save_path", None)
        watcher = self.__dict__.setdefault("_auto_timing", _AutoTiming())
        message = watcher.poll(folder, bool(enabled)) if folder and os.path.isdir(folder) else None
        version, video, note = watcher.latest
        if version == shown:
            return gr.update(), (gr.update(value=message) if message else gr.update()), shown
        return (gr.update(value=video) if video else gr.update()), gr.update(value=message or note), version

    @staticmethod
    def extract_checkpoint(video, frame):
        try:
            image, info, index = camera_timing.extract_frame(video, frame)
            return image, f"Loaded source frame {index+1} at {index/info['fps']:.6f}s. Set its checkpoint time in the new clip, then apply."
        except (ValueError, OSError, TypeError, subprocess.SubprocessError) as error:
            raise gr.Error(str(error)) from error

    @staticmethod
    def inspect_clip_timing(video, path_json, keyframe):
        try:
            result = camera_timing.inspect_hold(video, path_json, keyframe)
            arrival, departure = result['source_marks'][1:3]
            first, last = result['target_marks'][1:3]
            fps = result['media']['fps']
            return arrival+1, departure+1, (f"Suggested source hold: frames **{arrival+1}–{departure+1}** "
                f"({arrival/fps:.6f}–{departure/fps:.6f}s). Planned hold: **{first+1}–{last+1}** "
                f"({first/fps:.6f}–{last/fps:.6f}s). {result['note']}")
        except (ValueError, OSError, TypeError, subprocess.SubprocessError) as error:
            raise gr.Error(str(error)) from error

    def correct_clip_timing(self, video, path_json, keyframe, arrival, departure, audio, sampling):
        try:
            info = camera_timing.probe_video(video)
            first, last = camera_timing.held_path_frames(path_json, keyframe, info['frames'])
            source = [0,camera_timing._integer(arrival,'Arrival frame')-1,
                      camera_timing._integer(departure,'Departure frame')-1,info['frames']-1]
            target = [0,first,last,info['frames']-1]
            output_dir = getattr(self,'save_path',None) or Path.cwd()/'local_runtime'/'h3_camera_exports'
            output, report, result = camera_timing.export_retimed(video, source, target, output_dir, audio=audio, sampling=sampling)
            return output, report, (f"Exported **{info['frames']} frames at {info['fps']:g} fps**. "
                f"Selected arrival and departure now occupy frames **{first+1} and {last+1}**. "
                f"Audio: {result['audio_mode']}. {result['repeated_source_frames']} repeated source-frame selections. "
                "The timing report records every source-to-output frame mapping. Download the corrected clip above.")
        except (ValueError, OSError, TypeError, subprocess.SubprocessError) as error:
            raise gr.Error(str(error)) from error

    def verify_clip_candidates(self, videos, path_json, keyframe, audio, sampling):
        try:
            output_dir = getattr(self,'save_path',None) or Path.cwd()/'local_runtime'/'h3_camera_exports'
            video, report, result = camera_timing.verified_candidates(videos,path_json,keyframe,output_dir,audio,sampling)
            if video:
                marks = result['timing']['target_marks_zero_based']
                status = (f"**Timing checks passed** on candidate {len(result['attempts'])}. "
                          f"Measured hold boundaries: frames **{marks[1]+1} and {marks[2]+1}** (1-based). "
                          "Camera angles and roll are unverified; review the result. Subject motion and audio follow the selected retiming mode.")
            else:
                status = "**No candidate passed. No corrected video returned.** " + " ".join(
                    f"Candidate {index+1}: {attempt['reason']}" for index,attempt in enumerate(result['attempts']))
            return video,report,status
        except (ValueError,OSError,TypeError,subprocess.SubprocessError) as error:
            # Clear any earlier successful preview on invalid input as well.
            return None,None,f"**No result:** {error}"

    @staticmethod
    def use_preset(name):
        return PRESETS.get(name, gr.update())

    @staticmethod
    def _validated_plan(path_json, motion="animated", interpolation="smooth", close_loop=False, stabilize_roll=True):
        if not isinstance(path_json, str) or len(path_json.encode("utf-8")) > MAX_PLAN_BYTES:
            raise ValueError("The camera plan must be UTF-8 JSON smaller than 64 KB.")
        if motion not in ("animated", "frozen") or interpolation not in ("smooth", "linear") or not isinstance(close_loop, bool):
            raise ValueError("The camera plan has invalid motion, interpolation or loop settings.")
        if not isinstance(stabilize_roll, bool):
            raise ValueError("Camera roll stabilization must be true or false.")
        keyframes = validate_path(path_json)
        return {"format": PLAN_FORMAT, "keyframes": keyframes, "motion": motion,
                "interpolation": interpolation, "close_loop": close_loop, "stabilize_roll": stabilize_roll}

    @classmethod
    def load_plan(cls, path):
        if not path:
            return (gr.update(),) * 5
        try:
            source = Path(path)
            if source.stat().st_size > MAX_PLAN_BYTES:
                raise ValueError("The camera plan must be smaller than 64 KB.")
            payload = json.loads(source.read_text(encoding="utf-8-sig"))
            if isinstance(payload, list):
                plan = cls._validated_plan(json.dumps(payload))
            elif isinstance(payload, dict) and payload.get("format") == PLAN_FORMAT:
                plan = cls._validated_plan(json.dumps(payload.get("keyframes")), payload.get("motion", "animated"),
                                           payload.get("interpolation", "smooth"), payload.get("close_loop", False),
                                           payload.get("stabilize_roll", True))
            else:
                raise ValueError("Load a keyframe array or a saved WanGP H3 camera plan.")
            return (json.dumps(plan["keyframes"], indent=2), plan["motion"], plan["interpolation"],
                    plan["close_loop"], plan["stabilize_roll"])
        except (OSError, ValueError, TypeError, UnicodeError, RecursionError) as error:
            raise gr.Error(str(error)) from error

    @classmethod
    def save_plan_for_session(cls, path_json, motion, interpolation, close_loop, previous, stabilize_roll=True):
        saved = cls.save_plan(path_json, motion, interpolation, close_loop, stabilize_roll)
        try:
            cls.cleanup_export(previous)
        except OSError:
            cls.cleanup_export(saved)
            raise
        return saved, saved

    @staticmethod
    def cleanup_export(path):
        if path:
            Path(path).unlink(missing_ok=True)

    @classmethod
    def save_plan(cls, path_json, motion, interpolation, close_loop, stabilize_roll=True):
        try:
            plan = cls._validated_plan(path_json, motion, interpolation, close_loop, stabilize_roll)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", prefix="h3-camera-", delete=False) as target:
                json.dump(plan, target, indent=2, ensure_ascii=False, allow_nan=False)
                return target.name
        except (OSError, ValueError, TypeError) as error:
            raise gr.Error(str(error)) from error


class _AutoTiming:
    """Watch the output folder for renders of a one-hold camera plan and publish verified retimed copies."""

    def __init__(self):
        self.lock = threading.Lock()
        self.started = time.time()
        self.seen = {}
        self.busy = None
        self.latest = (0, None, "")

    def poll(self, folder, enabled):
        with self.lock:
            if self.busy:
                return f"Correcting hold timing of **{Path(self.busy).name}**..."
            if not enabled:
                return None
            candidate = self._next(folder)
            if candidate is None:
                return None
            self.busy = candidate
        threading.Thread(target=self._run, args=(candidate, folder), daemon=True, name="h3-camera-auto-timing").start()
        return f"Correcting hold timing of **{Path(candidate).name}**..."

    def _next(self, folder):
        now, best = time.time(), None
        for entry in os.scandir(folder):
            name = entry.name.lower()
            if not entry.is_file() or not name.endswith(".mp4") or "_timed_" in name or name.startswith("."):
                continue
            modified = entry.stat().st_mtime
            # Skip files from before this session, files still being written, and files already handled.
            if modified < self.started or now - modified < 4 or self.seen.get(entry.path) == modified:
                continue
            if best is None or modified > best[1]:
                best = (entry.path, modified)
        if best is None:
            return None
        self.seen[best[0]] = best[1]
        return best[0]

    def _run(self, source, folder):
        name = Path(source).name
        try:
            result = camera_timing.auto_correct(source, folder)
        except Exception as error:
            result = dict(status="rejected", reason=str(error))
        with self.lock:
            self.busy = None
            if result is None:
                return
            version = self.latest[0] + 1
            if result["status"] == "corrected":
                first, last = result["planned_hold_zero_based"]
                ending = (" Its final move now ends on the last frame." if result.get("measured_final_arrival_zero_based") is not None
                          and not result.get("ending_left_unchanged") else "")
                self.latest = (version, result["output"],
                               f"Hold timing corrected for **{name}**: the camera is now still on frames **{first + 1}-{last + 1}** "
                               f"as planned.{ending} Saved **{Path(result['output']).name}** with a timing report; the original is unchanged.")
            elif result["status"] == "exact":
                self.latest = (version, None, f"**{name}** already holds on the planned frames; no correction needed.")
            else:
                hint = (" H3 stayed at the hold too long for a clean retime; another seed usually times it better."
                        if "speed limit" in result.get("reason", "") or "more than" in result.get("reason", "") else "")
                self.latest = (version, None, f"Hold timing of **{name}** was left unchanged: {result.get('reason', 'not correctable')}{hint}")
