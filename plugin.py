"""Visual camera planning; generation continues through WanGP's native form."""

import json
import math
from pathlib import Path
import tempfile

import gradio as gr

from shared.utils.frame_scheduler import normalize_frame_count
from shared.utils.plugins import WAN2GPPlugin
from shared.utils.prompt_parser import split_prompt_units

from .camera_plan import DEFAULT_PATH, PRESETS, compile_plan, validate_path
from .editor import BRIDGE_JS, render_editor


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


class H3CameraPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "H3 Camera"
        self.version = "0.2.4"
        self.type = ["extension"]
        self.description = "Visual single-shot camera paths for MiniMax H3"

    def setup_ui(self):
        for name in dict.fromkeys((*FORM_INPUTS, *FORM_OUTPUTS, *LABEL_OUTPUTS, "model_choice_target")):
            self.request_component(name)
        for name in HOST_GLOBALS:
            self.request_global(name)
        self.request_global("PROMPT_TOOLS_ATTACH_JS")
        self.add_custom_js(f"({BRIDGE_JS})();")
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
                            ". H3 Camera requires the plugin APIs verified with WanGP 13.141.")
                return panel
            with gr.Row():
                motion = gr.Dropdown(label="Subject and environment", choices=[
                    ("Allow motion from the scene prompt", "animated"),
                    ("Freeze the scene; move only the camera", "frozen"),
                ], value="animated")
                interpolation = gr.Dropdown(label="Movement between keyframes", choices=[
                    ("Smooth ease in / ease out", "smooth"), ("Linear / steady speed", "linear"),
                ], value="smooth")
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
            args = [path, motion, interpolation, close_loop, *[components[name] for name in FORM_INPUTS]]
            preview.click(self.preview, inputs=args, outputs=[table, compiled, status], api_name="h3_camera_preview")
            apply.click(self.apply_plan, inputs=args, outputs=[*[components[name] for name in FORM_OUTPUTS],
                        table, compiled, status], api_name="h3_camera_apply").success(
                self.refresh_prompt_labels,
                inputs=[components[name] for name in ("state", "multi_prompts_gen_type", "image_mode")],
                outputs=[components[name] for name in LABEL_OUTPUTS], show_progress="hidden", api_name=False,
            ).success(
                self.refresh_video_length_label,
                inputs=[components[name] for name in ("state", "video_length", "force_fps", "video_guide", "video_source")],
                outputs=components["video_length"], show_progress="hidden", api_name=False,
            ).then(fn=None, inputs=None, outputs=None, js=self.PROMPT_TOOLS_ATTACH_JS, api_name=False)
            upload.change(self.load_plan, inputs=upload, outputs=[path, motion, interpolation, close_loop])
            save.click(self.save_plan_for_session, inputs=[path, motion, interpolation, close_loop, export_state],
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
            if not math.isfinite(fps) or fps <= 0 or not math.isfinite(frames):
                return "{}"
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
                 sliding_window_discard_last_frames, sliding_window_trim_first_frames, video_prompt_type, image_refs, image_end):
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
            fps = self.get_computed_fps(force_fps or "", model_type, video_guide, video_source)
            has_start = "S" in (image_prompt_type or "") and isinstance(image_start, list) and len(image_start) == 1 and image_start[0] is not None
            if close_loop and not has_start:
                raise ValueError("For a closed loop, select Start Image and provide exactly one image in the native Start Image gallery.")
            has_end = "E" in (image_prompt_type or "") and isinstance(image_end, list) and len(image_end) == 1 and image_end[0] is not None
            if close_loop and not has_end and "I" in (video_prompt_type or "") and image_refs:
                raise ValueError("Adding an End Image changes H3 reference numbering: it becomes <Picture 2>, shifting other images. Enable an End Image in the native form first, update the scene's reference labels, then apply the closed camera loop.")
            result = compile_plan(path_json, prompt=prompt, frame_count=frames, fps=fps,
                                  frozen=motion == "frozen", close_loop=bool(close_loop),
                                  reference_mode=bool(definition.get("reference_image_enabled")),
                                  has_start_image=has_start, interpolation=interpolation)
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
    def use_preset(name):
        return PRESETS.get(name, gr.update())

    @staticmethod
    def _validated_plan(path_json, motion="animated", interpolation="smooth", close_loop=False):
        if not isinstance(path_json, str) or len(path_json.encode("utf-8")) > MAX_PLAN_BYTES:
            raise ValueError("The camera plan must be UTF-8 JSON smaller than 64 KB.")
        if motion not in ("animated", "frozen") or interpolation not in ("smooth", "linear") or not isinstance(close_loop, bool):
            raise ValueError("The camera plan has invalid motion, interpolation or loop settings.")
        keyframes = validate_path(path_json)
        return {"format": PLAN_FORMAT, "keyframes": keyframes, "motion": motion,
                "interpolation": interpolation, "close_loop": close_loop}

    @classmethod
    def load_plan(cls, path):
        if not path:
            return (gr.update(),) * 4
        try:
            source = Path(path)
            if source.stat().st_size > MAX_PLAN_BYTES:
                raise ValueError("The camera plan must be smaller than 64 KB.")
            payload = json.loads(source.read_text(encoding="utf-8-sig"))
            if isinstance(payload, list):
                plan = cls._validated_plan(json.dumps(payload))
            elif isinstance(payload, dict) and payload.get("format") == PLAN_FORMAT:
                plan = cls._validated_plan(json.dumps(payload.get("keyframes")), payload.get("motion", "animated"),
                                           payload.get("interpolation", "smooth"), payload.get("close_loop", False))
            else:
                raise ValueError("Load a keyframe array or a saved WanGP H3 camera plan.")
            return json.dumps(plan["keyframes"], indent=2), plan["motion"], plan["interpolation"], plan["close_loop"]
        except (OSError, ValueError, TypeError, UnicodeError, RecursionError) as error:
            raise gr.Error(str(error)) from error

    @classmethod
    def save_plan_for_session(cls, path_json, motion, interpolation, close_loop, previous):
        saved = cls.save_plan(path_json, motion, interpolation, close_loop)
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
    def save_plan(cls, path_json, motion, interpolation, close_loop):
        try:
            plan = cls._validated_plan(path_json, motion, interpolation, close_loop)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", prefix="h3-camera-", delete=False) as target:
                json.dump(plan, target, indent=2, ensure_ascii=False, allow_nan=False)
                return target.name
        except (OSError, ValueError, TypeError) as error:
            raise gr.Error(str(error)) from error
