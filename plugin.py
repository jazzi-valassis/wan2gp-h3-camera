"""Visual camera planning; generation continues through WanGP's native form."""

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import gradio as gr
import numpy as np
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
try:
    from shared.deepy.hybrid import service_for
except ImportError:
    service_for = None
try:
    import preprocessing.depth_anything_v3.depth  # noqa: F401  (Depth Anything 3 renders the exact-pose views)
    DEPTH_READY = True
except ImportError:
    DEPTH_READY = False

from .camera_plan import (DEFAULT_PATH, PRESETS, add_camera_video, compile_plan, validate_path, keyframe_frame,
                          strip_camera_plan)
from .editor import BRIDGE_JS, render_editor
from .image_anchors import merge_timed_images
from . import timing as camera_timing
from . import view_anchors
from . import view_geometry


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
ANCHOR_OUTPUTS = ("image_refs", "frames_positions", "video_prompt_type", "video_prompt_type_image_refs", "image_refs_row",
                  "video_guide", "video_prompt_type_video_guide")
# Camera guidance on Apply: a camera preview video, exact-pose view anchors, or the text plan alone.
GUIDANCE_CHOICES = (("Camera preview video (recommended)", "video"), ("Exact-pose view anchors", "views"), ("Text only", "text"))
# A generated camera preview video is attached as the first Reference Video (H3 Ref2VA mode "V-U").
PREVIEW_PREFIX = "h3cam_camera_"
PREVIEW_VIDEO_MODE = "V-U"
VIDEO_GUIDE_LETTERS = "GPDEV+*-U1"
# H3 gives a reference video whole to every window and keeps its first 15 seconds.
PREVIEW_MAX_FRAMES = 362
# The preview is rendered at half the generation size, blurred and turned grey, so H3 takes its camera movement rather
# than its rough look: in colour, the smeared roof of test 1's preview came back as streaks on the rendered roof, and
# blurring alone only softened them.
PREVIEW_SCALE = 0.5
PREVIEW_BLUR = 4.0
PREVIEW_GREY = True
ANCHOR_INPUTS = ("path_json", "motion", "interpolation", "close_loop", *FORM_INPUTS, "stabilize_roll", "frames_positions")


class H3CameraPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "H3 Camera"
        self.version = "0.7.0"
        self.type = ["extension"]
        self.description = "Visual single-shot camera paths for MiniMax H3"

    def setup_ui(self):
        for name in dict.fromkeys((*FORM_INPUTS, *FORM_OUTPUTS, *LABEL_OUTPUTS, *ANCHOR_OUTPUTS, "model_choice_target")):
            self.request_component(name)
        for name in HOST_GLOBALS:
            self.request_global(name)
        self.request_global("PROMPT_TOOLS_ATTACH_JS")
        self.request_global("save_path")
        # Optional: lets view renders start on the server when no browser tab relays the queue trigger.
        self.request_global("load_queue_action")
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
            views_ready = self._wangp_session is not None and window_contexts is not None and DEPTH_READY
            with gr.Accordion("Camera guidance (recommended for orbits)", open=True):
                gr.Markdown("Text alone does not tell H3 which side to end on or how fast to move, so some seeds orbit past "
                            "the subject's back or linger at a hold. **Apply** can show H3 the camera path instead:\n\n"
                            "- **Camera preview video** (recommended): Depth Anything 3 re-renders the Start Image along "
                            "the whole path, and the result, blurred and grey, is attached as `<Video 1>`, a Reference Video H3 takes "
                            "only the camera movement from. H3 still animates the subject, so they turn in three dimensions "
                            "as the camera moves. Rendering takes one to two minutes plus one background image; the render "
                            "itself takes longer because H3 also reads the video. Single takes up to 15 seconds.\n"
                            "- **Exact-pose view anchors:** stills of every keyframe view and of pacing checkpoints, "
                            "rendered at their exact angles and injected at their times. They pin the subject's pose at "
                            "those frames, so a small subject can look flat as the camera turns around it.\n"
                            "- **Text only:** the written camera plan alone.\n\n"
                            "Both picture modes need an H3 Ref2VA model and one Start Image; otherwise Apply uses text "
                            "guidance and says why. Generated media are cached for the same image, path and seed.")
                with gr.Row():
                    guidance = gr.Radio(choices=list(GUIDANCE_CHOICES), value="video" if views_ready else "text",
                                        label="Camera guidance", interactive=views_ready)
                    view_seed = gr.Number(label="Seed for generated images", value=42, precision=0, minimum=0)
                views = gr.Gallery(label="Generated views", columns=4, height=260, interactive=False)
                if not views_ready:
                    gr.Markdown("This WanGP build lacks the plugin generation API, timed-image labels or Depth Anything 3. "
                                "Update WanGP to use the camera preview video or view anchors.")
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
                        apply.click(self.apply_camera, inputs=[guidance, view_seed, *args],
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
                guidance.interactive = False
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
                gr.update(value=values["video_prompt_type"]), gr.update(value="FI"), gr.update(visible=True),
                *self._video_updates(values))

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

    def _guidance_unavailable(self, mode, args):
        """Why the chosen camera guidance cannot be added to this form, or None."""
        if mode == "text":
            return "camera guidance is set to text only."
        if self._wangp_session is None or window_contexts is None or not DEPTH_READY:
            return "this WanGP build lacks the plugin generation API, timed-image labels or Depth Anything 3."
        try:
            values, initial = self._anchor_form(args)
        except gr.Error as error:
            return str(error.message if hasattr(error, "message") else error)
        start = view_anchors.pose_of(initial["path"][0])
        if all(view_anchors.pose_of(keyframe) == start for keyframe in initial["path"]):
            return "the path never leaves the start view, which the Start Image already shows."
        if mode == "video":
            definition = self.get_model_def(self.get_state_model_type(values["state"])) or {}
            guide = self._video_path(values["video_guide"])
            if not definition.get("reference_video_enabled"):
                return "this model takes no Reference Video; choose view anchors instead."
            if guide and not self._is_preview(guide):
                return "a Reference / Control Video is already attached; remove it or choose view anchors."
            if initial["frame_count"] > PREVIEW_MAX_FRAMES:
                return "H3 reads at most 15 seconds of a Reference Video; choose view anchors for longer takes."
            return None
        targets, _ = view_anchors.anchor_targets(initial["path"], initial["frame_count"], keyframe_frame,
                                                 initial["fps"], values["interpolation"])
        if not targets:
            return "the path never leaves the start view, which the Start Image already shows."
        return None

    def apply_camera(self, mode, seed, *args):
        """Apply the camera path with the chosen camera guidance: a camera preview video, view anchors or text.
        ``mode`` is "video", "views" or "text"; True and False mean views and text, as before."""
        # WanGP pumps the queue only for handlers whose own code uses the plugin session.
        session = self._wangp_session
        mode = {True: "views", False: "text"}.get(mode, mode)
        text_only = list(self.apply_plan(*args))  # Validates the plan; invalid input raises before any change.
        reason = self._guidance_unavailable(mode, args)
        if reason:
            text_only[-1] += f" Text guidance only: {reason}"
            yield self._as_updates((*text_only, *self._text_media_updates(args), gr.update(), str(time.time_ns())))
            return
        if mode == "video":
            steps = self._camera_video_steps(session, seed, args)
        else:
            steps = self._view_anchor_steps(session, seed, args)
        for final, updates in steps:
            yield (*updates, gr.update(value=str(time.time_ns())) if final else gr.update())

    @staticmethod
    def _video_path(item):
        if isinstance(item, dict):
            item = item.get("path") or (item.get("video") or {}).get("path")
        return str(item) if isinstance(item, (str, Path)) and str(item) else None

    @staticmethod
    def _is_preview(path):
        return bool(path) and Path(path).name.startswith(PREVIEW_PREFIX)

    def _without_generated_video(self, values):
        """Detach a camera preview video an earlier Apply attached, so another guidance mode never keeps it."""
        if not self._is_preview(self._video_path(values.get("video_guide"))):
            return values
        flags = "".join(flag for flag in values["video_prompt_type"] or "" if flag not in VIDEO_GUIDE_LETTERS)
        return dict(values, video_guide=None, video_prompt_type=flags, video_cleared=True)

    @staticmethod
    def _video_updates(values):
        """Updates for the video_guide outputs: cleared when a preview video was detached, unchanged otherwise."""
        if values.get("video_cleared"):
            return gr.update(value=None), gr.update(value="")
        return gr.update(), gr.update()

    def _text_media_updates(self, args):
        """ANCHOR_OUTPUTS updates for a text-only Apply: detach a camera preview video, leave everything else."""
        values = self._without_generated_video(dict(zip(ANCHOR_INPUTS, args)))
        if not values.get("video_cleared"):
            return (gr.update(),) * len(ANCHOR_OUTPUTS)
        return (gr.update(), gr.update(), gr.update(value=values["video_prompt_type"]), gr.update(), gr.update(),
                *self._video_updates(values))

    def _relay(self, steps):
        """Forward a _ViewRun step's progress text as status updates and return its result."""
        count = len(FORM_OUTPUTS) + 3 + len(ANCHOR_OUTPUTS) + 1
        while True:
            try:
                message = next(steps)
            except StopIteration as stop:
                return stop.value
            updates = [gr.update()] * count
            updates[len(FORM_OUTPUTS) + 2] = gr.update(value=message)
            yield False, tuple(updates)

    def _camera_video_steps(self, session, seed, args):
        """Yield (final, updates): progress messages, then the form with the camera preview video attached."""
        values, initial = self._anchor_form(args)
        try:
            seed = int(seed)
            if seed < 0:
                raise ValueError("Use a seed of 0 or more.")
            values = self._without_generated_views(self._without_generated_video(values), initial)
            positions = self._injection_positions(values, initial)
        except (ValueError, TypeError, OverflowError) as error:
            raise gr.Error(str(error)) from error
        frames, fps = initial["frame_count"], initial["fps"]
        image_path, size = self._image_file(initial["image_start"][0])
        resolution = view_anchors.view_resolution(*size)
        width, height = map(int, resolution.split("x"))
        preview_size = (max(16, round(width * PREVIEW_SCALE / 2) * 2), max(16, round(height * PREVIEW_SCALE / 2) * 2))
        image_hash = hashlib.sha1(Path(image_path).read_bytes()).hexdigest()
        plan = json.dumps(initial["path"], sort_keys=True)
        cache = self.__dict__.setdefault("_view_cache", {})
        key = ("preview", image_hash, resolution, seed, plan, frames, values["interpolation"])
        video = cache.get(key)
        if not (video and Path(video).is_file()):
            name = hashlib.sha1(repr(key).encode()).hexdigest()[:12]
            folder = Path(getattr(self, "save_path", None) or tempfile.gettempdir())
            run = self._view_run(session, values["state"], image_path, resolution, seed)
            try:
                yield from self._relay(run.start(1, cache, (image_hash, resolution, seed), "the camera preview video"))
                poses = view_anchors.frame_poses(initial["path"], frames, keyframe_frame, values["interpolation"])
                video = yield from self._relay(self._render_preview(run.scene, poses, preview_size, fps,
                                                                   folder / f"{PREVIEW_PREFIX}{name}_s{seed}.mp4"))
            finally:
                run.close()
            cache[key] = video
        flags = "".join(flag for flag in values["video_prompt_type"] or "" if flag not in VIDEO_GUIDE_LETTERS)
        flags = (flags if positions else flags.replace("F", "")) + PREVIEW_VIDEO_MODE
        result = self._compile(*[values[name] for name in ANCHOR_INPUTS])
        result["prompt"] = add_camera_video(result["prompt"])
        result["summary"] += (f" Camera preview video: {frames} frames along the path, attached as <Video 1>, seed {seed}. "
                              "H3 takes its camera movement and animates the subject itself.")
        updates = (*self._apply_result(result), gr.update(value=values["image_refs"]),
                   gr.update(value=values["frames_positions"]), gr.update(value=flags), gr.update(), gr.update(),
                   gr.update(value=video, visible=True), gr.update(value=PREVIEW_VIDEO_MODE))
        yield True, self._as_updates((*updates, []))

    @staticmethod
    def _render_preview(scene, poses, size, fps, target):
        """Render, soften and encode the camera preview video; yields progress text and returns its path."""
        import cv2
        rendered = []
        for index, frame in enumerate(scene.flythrough(poses, size), 1):
            if PREVIEW_BLUR:
                frame = cv2.GaussianBlur(frame, (0, 0), PREVIEW_BLUR)
            if PREVIEW_GREY:
                frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
            rendered.append(frame)
            if index % 24 == 0 or index == len(poses):
                yield f"Rendering the camera preview video: frame {index} of {len(poses)}..."
        return camera_timing.write_video(rendered, fps, size, target)

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
        relay = self._relay
        values, initial = self._anchor_form(args)
        try:
            seed = int(seed)
            if seed < 0:
                raise ValueError("Use a view image seed of 0 or more.")
            values = self._without_generated_views(self._without_generated_video(values), initial)
            positions = self._injection_positions(values, initial)
        except (ValueError, TypeError, OverflowError) as error:
            raise gr.Error(str(error)) from error
        fps = initial["fps"]
        targets, skipped = view_anchors.anchor_targets(initial["path"], initial["frame_count"], keyframe_frame, fps,
                                                       values["interpolation"])
        image_path, size = self._image_file(initial["image_start"][0])
        resolution = view_anchors.view_resolution(*size)
        image_hash = hashlib.sha1(Path(image_path).read_bytes()).hexdigest()
        cache = self.__dict__.setdefault("_view_cache", {})
        run = self._view_run(session, values["state"], image_path, resolution, seed)
        # A view depends on every view rendered before it, so the cache key is the whole chain so far.
        chain = (image_hash, resolution, seed)
        files, measured = [], []
        previous = (0.0, 0.0, 1.0)
        try:
            yield from relay(run.start(len(targets), cache, (image_hash, resolution, seed)))
            for number, target in enumerate(targets, 1):
                pose = target["pose"]
                chain += (pose,)
                cached = cache.get(chain)
                if isinstance(cached, tuple):
                    # This view missed its pose on an earlier Apply; the same seed would miss the same way.
                    cached, got = None, cached[1]
                elif cached and Path(cached).is_file():
                    got = yield from relay(run.reuse(cached, pose))
                else:
                    # Re-render from a view at most MAX_STEP away: far targets first get helper views along the path.
                    while view_anchors.pose_gap(run.nearest(pose), pose) > view_anchors.MAX_STEP:
                        helper = view_anchors.helper_pose(previous, pose)
                        if helper is None:
                            break
                        yield from relay(run.view(helper, f"helper view on the way to view {number} of {len(targets)}"))
                        previous = helper
                    # The view must not land nearer the poses planned before and after it than its own.
                    before = targets[number - 2]["pose"] if number > 1 else (0.0, 0.0, 1.0)
                    after = [target["pose"] for target in targets[number:number + 1]]
                    cached, got = yield from relay(run.view(pose, f"view {number} of {len(targets)}", (before, *after)))
                    cache[chain] = cached or ("missed", got)
                files.append(cached)
                measured.append(got)
                previous = pose
        finally:
            run.close()
        kept = [(target, file, got) for target, file, got in zip(targets, files, measured) if file]
        missed = [(target, got) for target, file, got in zip(targets, files, measured) if not file]
        frames = sorted(frame for target, _, _ in kept for frame in target["frames"])
        if kept:
            description = (f" Generated exact-pose view anchors at frames {' '.join(str(frame + 1) for frame in frames)} "
                           f"(1-based), seed {seed}." + view_anchors.describe([target for target, _, _ in kept], skipped,
                                                                              [got for _, _, got in kept], fps))
        else:
            description = f" No generated view landed close enough to its pose to anchor (seed {seed})."
        description += view_anchors.describe_missed(missed, fps)
        additions = {frame: file for target, file, _ in kept for frame in target["frames"]}
        gallery = [(file, f"{', '.join(f'{frame / fps:.2f}s' for frame in sorted(target['frames']))}: "
                          f"{view_anchors.pose_label(target['pose'])} (measured {view_anchors.pose_label(got)})")
                   for target, file, got in kept]
        if not additions and not positions:
            # Nothing to inject: apply the plan as text, without the views an earlier Apply injected.
            result = self._compile(*[values[name] for name in ANCHOR_INPUTS])
            result["summary"] += description
            anchors = (gr.update(value=values["image_refs"]), gr.update(value=values["frames_positions"]),
                       gr.update(value=values["video_prompt_type"]), gr.update(), gr.update(), *self._video_updates(values))
            yield True, self._as_updates((*self._apply_result(result), *anchors, gallery))
            return
        updates = self._apply_image_updates(values, initial, positions, additions, description)
        yield True, self._as_updates((*updates, gallery))

    def _view_run(self, session, state, image_path, resolution, seed):
        return _ViewRun(self, session, state, image_path, resolution, seed)

    def _start_queue_without_browser(self, state):
        """Admit a plugin job that is waiting for a browser tab to relay the queue trigger.

        Plugin jobs normally start when the page that pressed Apply forwards WanGP's queue trigger, and a
        background or unfocused tab can delay that indefinitely. WanGP's server-side generation service (the
        one Deepy uses) can load the waiting inline queue itself. Returns whether it did."""
        service = service_for(state) if service_for is not None else None
        gen = state.get("gen") if isinstance(state, dict) else None
        load = getattr(self, "load_queue_action", None)
        if service is None or not isinstance(gen, dict) or gen.get("inline_queue") is None or not callable(load):
            return False
        load(None, state, SimpleNamespace(target=1))
        service.start_generation()
        print("[H3 Camera] No browser tab relayed the queue trigger; started the view render on the server.")
        return True

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


class _ViewRun:
    """One Apply's view renders: the depth model, the registered views and their queue jobs.

    Steps are generators that yield progress text and return their result."""

    # Seconds a job may wait for the browser to relay WanGP's queue trigger before the server starts it.
    BROWSER_GRACE = 2.0

    def __init__(self, plugin, session, state, image_path, resolution, seed):
        self.plugin, self.session, self.state = plugin, session, state
        self.image_path, self.resolution, self.seed = image_path, resolution, seed
        self.size = tuple(map(int, resolution.split("x")))
        self.folder = Path(tempfile.mkdtemp(prefix="h3_camera_views_"))
        self.model = view_geometry.DepthModel()
        self.scene = None
        self.renders = 0

    def _busy(self):
        gen = self.state.get("gen") if isinstance(self.state, dict) else None
        return isinstance(gen, dict) and bool(gen.get("in_progress") or gen.get("queue"))

    def _wait_idle(self):
        """Depth estimation shares the GPU; let any running generation finish first."""
        last = 0.0
        while self._busy():
            if time.time() - last >= 5:
                last = time.time()
                yield "Waiting for the current generation to finish before estimating depth..."
            time.sleep(0.5)

    def start(self, count, cache, key, what=None):
        """Find the subject, paint it out for a background plate, and register the Start Image."""
        self.cache = cache
        what = what or f"{count} exact-pose view(s)"
        yield from self._wait_idle()
        yield f"Preparing {what}: finding the subject in the Start Image..."
        with Image.open(self.image_path) as start:
            start = start.convert("RGB").resize(self.size)
        mask = view_geometry.subject_mask(start)
        plate, grown = yield from self._plate(self.image_path, start, mask, "the Start Image")
        yield f"Preparing {what}: estimating depth..."
        self.scene = view_geometry.Scene(self.model, start, mask, plate, grown)
        self.files = [self.image_path]

    def _plate(self, file, image, mask, label):
        """Background plate of one picture: its subject painted out. Returns (plate, grown mask) or (None, None)."""
        # A subject filling most of the frame leaves nothing to paint behind it.
        if mask is None or not .005 <= mask.mean() <= .6:
            return None, None
        grown = view_geometry.plate_mask(mask)
        key = ("plate", file, self.resolution, self.seed)
        plate = self.cache.get(key)
        if not (plate and Path(plate).is_file()):
            self.renders += 1
            mask_path = self.folder / f"plate_mask_{self.renders}.png"
            Image.fromarray(grown.astype(np.uint8) * 255).save(mask_path)
            task = view_anchors.plate_task(file, str(mask_path), self.resolution, self.seed,
                                           view_anchors.VIEW_PREFIX + f"plate_s{self.seed}")
            plate = yield from self._render(task, f"the background plate of {label}",
                                            f"Painting the subject out of {label} for a background layer...")
            self.cache[key] = plate
        yield from self._wait_idle()
        with Image.open(plate) as picture:
            return picture.convert("RGB").resize(self.size), grown

    def _prepare_source(self, pose):
        """A generated view about to be re-rendered gets its subject mask (for the flat card) and its own
        background plate first."""
        source = self.scene.source_for(pose, self.size)
        if source == 0 or source in self.scene.subjects:
            return
        image = self.scene.images[source]
        mask = view_geometry.subject_mask(image)
        self.scene.set_subject(source, mask)
        plate, grown = yield from self._plate(self.files[source], image, mask, "the nearest generated view")
        self.scene.set_plate(source, plate, grown)

    def nearest(self, pose):
        return min(self.scene.measured, key=lambda measured: view_anchors.pose_gap(measured, pose))

    def reuse(self, file, pose):
        """Register a cached view; returns its measured pose."""
        yield from self._wait_idle()
        with Image.open(file) as view:
            got = self.scene.add(view, pose)
        self.files.append(file)
        return got

    def view(self, pose, label, neighbours=()):
        """Render one view at ``pose``, re-aimed once past the miss if it lands too far or nearer a ``neighbours``
        pose than its own. Returns (file, measured); file is None when both attempts miss."""
        attempts, aim = [], pose
        yield from self._prepare_source(pose)
        for attempt in range(2):
            coarse, _, _ = self.scene.coarse(aim, self.size)
            self.renders += 1
            coarse_path = self.folder / f"coarse_{self.renders}.png"
            coarse.save(coarse_path)
            task = view_anchors.view_task(self.image_path, str(coarse_path), self.resolution, self.seed + attempt,
                                          view_anchors.view_name(pose, self.seed, attempt))
            what = f"{label} ({view_anchors.pose_label(pose)}{', re-aimed' if attempt else ''})"
            file = yield from self._render(task, what)
            yield from self._wait_idle()
            with Image.open(file) as view:
                got = self.scene.add(view, pose)
            attempts.append((view_anchors.pose_gap(got, pose), attempt, file, got))
            if view_anchors.accepted(got, pose, neighbours):
                self.files.append(file)
                return file, got
            self.scene.drop_last()
            aim = view_anchors.aimed_pose(pose, got)
        # Neither attempt met the tolerance. H3 follows an injected view within a few degrees, so a view this far off
        # would pull the render off the plan by as much; leave it out and let the text guide this pose.
        return None, min(attempts)[3]

    def _render(self, task, what, first=None):
        job = self.session.submit([task])
        started = last = time.time()
        relayed = False
        yield first or f"Rendering {what} with Qwen Image 2.1 and AnyAngle..."
        while not job.done:
            time.sleep(0.25)
            now = time.time()
            if not relayed and now - started >= self.BROWSER_GRACE:
                relayed = True
                try:
                    self.plugin._start_queue_without_browser(self.state)
                except Exception as error:  # The browser can still relay the trigger.
                    print(f"[H3 Camera] Could not start the view render on the server: {error}")
            if now - last >= 5:
                last = now
                yield f"Rendering {what}... {int(now - started)}s elapsed."
        result = job.result(timeout=1.0)
        if result.cancelled:
            raise gr.Error("View generation was cancelled; the generation form is unchanged.")
        files = list(result.generated_files or [])
        if not result.success or len(files) != 1:
            errors = "; ".join(error.message for error in result.errors) or "the queue returned no image"
            raise gr.Error(f"View generation failed ({errors}); the generation form is unchanged.")
        return files[0]

    def close(self):
        self.model.close()
        shutil.rmtree(self.folder, ignore_errors=True)


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
                sound = (" Audio retimed to keep the dialogue in sync." if result.get("audio") == "retime"
                         else " Original audio kept unchanged.")
                self.latest = (version, result["output"],
                               f"Hold timing corrected for **{name}**: the camera is now still on frames **{first + 1}-{last + 1}** "
                               f"as planned.{ending}{sound} Saved **{Path(result['output']).name}** with a timing report; "
                               "the original is unchanged.")
            elif result["status"] == "exact":
                self.latest = (version, None, f"**{name}** already holds on the planned frames; no correction needed.")
            else:
                hint = (" H3 stayed at the hold too long for a clean retime; another seed usually times it better."
                        if "speed limit" in result.get("reason", "") or "more than" in result.get("reason", "") else "")
                self.latest = (version, None, f"Hold timing of **{name}** was left unchanged: {result.get('reason', 'not correctable')}{hint}")
