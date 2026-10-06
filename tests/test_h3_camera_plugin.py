"""Native-form integration tests; no model loading or GPU generation required."""

import importlib
import json
from pathlib import Path
import tempfile
import unittest

import gradio as gr
from PIL import Image

from shared.utils.frame_scheduler import build_frame_scheduler
from shared.utils.plugins import PluginManager
from shared.utils.prompt_parser import split_prompt_units


from plugin_under_test import load_plugin

camera = load_plugin()


def make_plugin():
    plugin = camera.H3CameraPlugin()
    plugin.setup_ui()
    definition = {"frames_minimum": 107, "frames_steps": 17, "frames_offset": 5,
                  "image_prompt_types_allowed": "TSEVL", "reference_image_enabled": True, "custom_frames_injection": True,
                  "sliding_window_defaults": {"window_min": 124, "window_max": 481}}
    plugin.get_model_def = lambda model: dict(definition, audio_only=model == "tts")
    plugin.get_model_family = lambda model: "minimax_h3" if model != "wan" else "wan"
    plugin.get_state_model_type = lambda state: state["model_type"]
    plugin.get_computed_fps = lambda force, model, guide, source: (30 if guide else 24) if force in ("", "auto") else float(force)
    plugin.refresh_prompt_labels = lambda *args: (gr.update(),) * 5
    plugin.refresh_video_length_label = lambda *args: gr.update()
    plugin.PROMPT_TOOLS_ATTACH_JS = "() => window.wangpPromptTools?.attach?.(document)"
    return plugin


def arguments(**changes):
    values = dict(prompt="A boat drifts on a lake.\noverall_soundscape: Water and birds.\nnon_diegetic_music: Soft piano.",
                  state={"model_type": "h3"}, force_fps="24", video_length=226,
                  video_guide=None, video_source=None, image_mode=0,
                  image_prompt_type="", image_start=None, multi_prompts_gen_type="PG",
                  sliding_window_size=124, sliding_window_discard_last_frames=17,
                  sliding_window_trim_first_frames=2, video_prompt_type="", image_refs=None, image_end=None)
    values.update(changes)
    return [camera.DEFAULT_PATH, "animated", "smooth", False, *[values[name] for name in camera.FORM_INPUTS]]


class NativeCameraTests(unittest.TestCase):
    def setUp(self):
        self.plugin = make_plugin()

    def build_real_plugin_ui(self):
        with gr.Blocks() as ui:
            components = {}
            with gr.Column():
                components["prompt"] = gr.Textbox("A single scene")
            for name in self.plugin.component_requests:
                if name == "prompt":
                    continue
                if name == "state":
                    component = gr.State({"model_type": "h3"})
                elif name == "image_end_row":
                    component = gr.Row()
                elif name in ("image_start", "image_end", "image_refs"):
                    component = gr.Gallery()
                elif name == "image_prompt_type_endcheckbox":
                    component = gr.Checkbox()
                elif name == "image_mode":
                    component = gr.Number(0)
                elif name in ("video_length", "sliding_window_size"):
                    component = gr.Slider(22, 481, value=226)
                elif name in ("video_guide", "video_source"):
                    component = gr.Video()
                elif name in ("prompt_info_label", "wizard_prompt_info_label"):
                    component = gr.HTML()
                else:
                    component = gr.Textbox("24" if name == "force_fps" else "")
                components[name] = component
            manager = PluginManager()
            manager.plugins = {"wan2gp-h3-camera": self.plugin}
            manager.run_component_insertion_and_setup(components)
        return ui.get_config_file(), components

    def test_apply_routes_audio_and_all_prompt_lines_as_one_native_request(self):
        result = self.plugin.apply_plan(*arguments())
        changes = dict(zip(camera.FORM_OUTPUTS, result))
        prompt = changes["prompt"]["value"]
        self.assertIn("overall_soundscape: Water and birds.", prompt)
        self.assertIn("non_diegetic_music: Soft piano.", prompt)
        self.assertEqual(changes["multi_prompts_gen_type"]["value"], "FG")
        self.assertEqual(len(split_prompt_units(prompt, "FG")), 1)
        self.assertEqual(changes["video_length"]["value"], 226)
        self.assertEqual(changes["sliding_window_size"]["value"], 226)
        self.assertEqual(changes["sliding_window_discard_last_frames"]["value"], 0)
        self.assertEqual(changes["sliding_window_trim_first_frames"]["value"], 0)
        for name in camera.FORM_OUTPUTS[6:]:
            self.assertEqual(changes[name], gr.update())

    def test_native_scheduler_keeps_one_complete_window(self):
        result = self.plugin._compile(*arguments(video_length=225))
        scheduler, error = build_frame_scheduler(split_prompt_units(result["prompt"], "FG"),
            total_frames=result["frame_count"], fps=result["fps"], window_size=result["sliding_window_size"],
            default_overlap=22, minimum=107, step=17, frame_offset=5, overlap_offset=5,
            max_overlap=56, output_frame_policy="expand", allow_new_shot=True)
        self.assertIsNone(error)
        # No slash commands means WanGP uses the normal single-window path.
        self.assertFalse(scheduler["active"])
        self.assertEqual(result["frame_count"], 226)
        self.assertGreaterEqual(result["sliding_window_size"], result["frame_count"])

    def test_minimum_duration_respects_model_and_window_grid(self):
        result = self.plugin._compile(*arguments(video_length=20))
        self.assertEqual(result["frame_count"], 107)
        self.assertEqual(result["sliding_window_size"], 124)

    def test_actual_fps_is_used_for_timestamps(self):
        result = self.plugin._compile(*arguments(force_fps="60"))
        self.assertEqual(result["fps"], 60)
        self.assertAlmostEqual(result["rows"][-1][1], 225 / 60, places=3)
        timing = json.loads(self.plugin.update_timing(226, "auto", "reference.mp4", None, {"model_type": "h3"}))
        self.assertEqual(timing["fps"], 30)

    def test_loop_uses_actual_gallery_image_and_enables_native_end_controls(self):
        start = [(Image.new("RGB", (32, 32), "red"), "start")]
        args = arguments(image_prompt_type="S", image_start=start)
        args[:4] = [camera.PRESETS["Full orbit"], "animated", "linear", True]
        changes = dict(zip(camera.FORM_OUTPUTS, self.plugin.apply_plan(*args)))
        self.assertIs(changes["image_end"]["value"], start)
        self.assertEqual(changes["image_prompt_type"]["value"], "SE")
        self.assertTrue(changes["image_prompt_type_endcheckbox"]["value"])
        self.assertTrue(changes["image_end_row"]["visible"])

    def test_loop_requires_one_active_start_image(self):
        for mode, images in (("", ["image.png"]), ("S", None), ("S", ["a.png", "b.png"])):
            with self.subTest(mode=mode, images=images), self.assertRaises(gr.Error):
                args = arguments(image_prompt_type=mode, image_start=images)
                args[3] = True
                self.plugin.apply_plan(*args)

    def test_new_loop_endpoint_cannot_silently_renumber_reference_images(self):
        args = arguments(image_prompt_type="S", image_start=["start.png"], image_refs=["person.png"],
                         video_prompt_type="I", prompt="The person in <Picture 2> smiles.")
        args[0], args[3] = camera.PRESETS["Full orbit"], True
        with self.assertRaisesRegex(gr.Error, "reference numbering"):
            self.plugin.apply_plan(*args)
        args[camera.FORM_INPUTS.index("image_prompt_type") + 4] = "SE"
        with self.assertRaisesRegex(gr.Error, "reference numbering"):
            self.plugin.apply_plan(*args)
        args[camera.FORM_INPUTS.index("image_end") + 4] = ["existing-end.png"]
        self.assertIn("<Picture 2>", self.plugin._compile(*args)["prompt"])

    def test_unsupported_or_multishot_input_is_rejected_without_form_updates(self):
        for changes in ({"state": {"model_type": "wan"}}, {"state": {"model_type": "tts"}},
                        {"image_mode": 1}, {"video_source": "clip.mp4"}, {"image_prompt_type": "V"},
                        {"video_prompt_type": "G"}, {"video_length": 482}, {"video_length": float("nan")},
                        {"prompt": "[Shot 1] One. [Shot 2] Two."},
                        {"prompt": "[/duration=5] One."},
                        {"prompt": "One\n\nTwo", "multi_prompts_gen_type": "PW"}):
            with self.subTest(changes=changes), self.assertRaises(gr.Error):
                self.plugin.apply_plan(*arguments(**changes))

    def test_reference_video_and_reference_tokens_remain_usable(self):
        result = self.plugin._compile(*arguments(prompt="<Subject 1> from <Picture 1> walks.",
                                                video_guide="reference.mp4", video_prompt_type="V-UI"))
        self.assertIn("<Picture 1>", result["prompt"])

    def test_preview_does_not_modify_inputs(self):
        args = arguments()
        state = json.dumps(args[5])
        self.plugin.preview(*args)
        self.assertEqual(json.dumps(args[5]), state)
        self.assertEqual(args[4], arguments()[4])

    def test_save_load_roundtrip_and_plain_upstream_keyframes(self):
        saved = self.plugin.save_plan(camera.DEFAULT_PATH, "frozen", "linear", False)
        try:
            loaded = self.plugin.load_plan(saved)
            self.assertEqual(json.loads(loaded[0]), json.loads(camera.DEFAULT_PATH))
            self.assertEqual(loaded[1:], ("frozen", "linear", False, True))
        finally:
            Path(saved).unlink()
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "plan.json"
            file.write_text(camera.DEFAULT_PATH, encoding="utf-8")
            self.assertEqual(self.plugin.load_plan(file)[1:], ("animated", "smooth", False, True))
            file.write_text("[]", encoding="utf-8")
            with self.assertRaises(gr.Error):
                self.plugin.load_plan(file)

    def test_model_target_visibility_uses_new_model_not_old_session(self):
        shown, _ = self.plugin.update_visibility("h3|123", 226, "24", None, None)
        hidden, timing = self.plugin.update_visibility("wan|124", 226, "24", None, None)
        self.assertTrue(shown["visible"])
        self.assertFalse(hidden["visible"])
        self.assertEqual(timing, "{}")

    def test_roll_option_survives_export_and_reaches_preview_and_apply(self):
        saved = self.plugin.save_plan(camera.DEFAULT_PATH, "animated", "smooth", False, False)
        try:
            loaded = self.plugin.load_plan(saved)
            self.assertFalse(loaded[-1])
            args = arguments() + [loaded[-1]]
            self.assertNotIn("Keep the horizon level", self.plugin.preview(*args)[1])
            self.assertNotIn("Keep the horizon level", self.plugin.apply_plan(*args)[0]["value"])
            args[-1] = True
            self.assertIn("Keep the horizon level", self.plugin.preview(*args)[1])
            self.assertIn("Keep the horizon level", self.plugin.apply_plan(*args)[0]["value"])
        finally:
            Path(saved).unlink()

    def test_legacy_plan_defaults_to_stabilized_roll_and_invalid_option_is_rejected(self):
        payload = {"format": camera.PLAN_FORMAT, "keyframes": json.loads(camera.DEFAULT_PATH)}
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "plan.json"
            file.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(self.plugin.load_plan(file)[-1])
            payload["stabilize_roll"] = "false"
            file.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(gr.Error):
                self.plugin.load_plan(file)
        self.assertEqual(self.plugin.load_plan(None), (gr.update(),) * 5)

    def test_preview_timing_matches_applied_frame_alignment_for_hold_duration(self):
        for requested in (20, 225, 239, 243):
            timing = json.loads(self.plugin.timing_value(requested, "24", None, None, "h3"))
            applied = self.plugin._compile(*arguments(video_length=requested))
            self.assertEqual(timing["frame_count"], applied["frame_count"])
            self.assertEqual(timing["seconds"], applied["end_seconds"])

    def test_export_replacement_cleans_only_the_current_sessions_previous_file(self):
        first = self.plugin.save_plan(camera.DEFAULT_PATH, "animated", "smooth", False)
        other = self.plugin.save_plan(camera.DEFAULT_PATH, "animated", "smooth", False)
        saved = None
        try:
            saved, session_file = self.plugin.save_plan_for_session(camera.DEFAULT_PATH, "frozen", "linear", False, first)
            self.assertEqual(saved, session_file)
            self.assertTrue(Path(saved).is_file())
            self.assertFalse(Path(first).exists())
            self.assertTrue(Path(other).is_file())
            self.plugin.cleanup_export(session_file)
            self.assertFalse(Path(saved).exists())
        finally:
            for name in (first, other, saved):
                if name:
                    Path(name).unlink(missing_ok=True)

    def test_invalid_export_keeps_previous_download(self):
        first = self.plugin.save_plan(camera.DEFAULT_PATH, "animated", "smooth", False)
        try:
            with self.assertRaises(gr.Error):
                self.plugin.save_plan_for_session("{}", "animated", "smooth", False, first)
            self.assertTrue(Path(first).is_file())
        finally:
            Path(first).unlink(missing_ok=True)

    def test_bridge_uses_host_javascript_registration(self):
        snippets = self.plugin.custom_js_snippets
        self.assertEqual(len(snippets), 1)
        self.assertIn("wan2gp:h3-camera:v1", snippets[0])

    def test_setup_requests_native_label_refresh_contract(self):
        self.assertTrue({"refresh_prompt_labels", "refresh_video_length_label", "PROMPT_TOOLS_ATTACH_JS"}
                        <= set(self.plugin.global_requests))
        self.assertTrue({"wizard_prompt", "prompt_info_label", "wizard_prompt_info_label"}
                        <= set(self.plugin.component_requests))

    def test_missing_global_produces_visible_compatibility_diagnostic(self):
        del self.plugin.get_model_family
        self.plugin.post_ui_setup({})
        with gr.Blocks() as ui:
            panel = self.plugin.build_panel()
        self.assertTrue(panel.visible)
        config = ui.get_config_file()
        messages = [str(item["props"].get("value", "")) for item in config["components"]]
        self.assertTrue(any("get_model_family" in text for text in messages))

    def test_preview_uses_independent_session_values(self):
        first = arguments(prompt="First person's scene.", force_fps="30")
        second = arguments(prompt="Second person's scene.", force_fps="60")
        first_result = self.plugin.preview(*first)
        second_result = self.plugin.preview(*second)
        self.assertIn("First person's scene.", first_result[1])
        self.assertNotIn("Second person's scene.", first_result[1])
        self.assertIn("Second person's scene.", second_result[1])
        self.assertNotIn("First person's scene.", second_result[1])
        self.assertEqual(self.plugin.preview(*first), first_result)

    def test_real_gradio_plugin_insertion_registers_apply_and_native_outputs(self):
        config, components = self.build_real_plugin_ui()
        apply = next(fn for fn in config["dependencies"] if fn["api_name"] == "h3_camera_apply")
        self.assertEqual(apply["outputs"][:len(camera.FORM_OUTPUTS)], [components[name]._id for name in camera.FORM_OUTPUTS])
        panel = next(item for item in config["components"] if item["props"].get("elem_id") == "h3-camera-panel")
        self.assertTrue(panel["props"]["visible"])

    def test_apply_label_refresh_chain_runs_only_after_success(self):
        config, components = self.build_real_plugin_ui()
        dependencies = config["dependencies"]
        apply = next(item for item in dependencies if item["api_name"] == "h3_camera_apply")
        prompt_refreshes = [item for item in dependencies if item.get("trigger_after") == apply["id"]]
        self.assertEqual(len(prompt_refreshes), 1)
        prompt_refresh = prompt_refreshes[0]
        self.assertTrue(prompt_refresh["trigger_only_on_success"])
        self.assertFalse(prompt_refresh["api_name"])
        self.assertEqual(prompt_refresh["show_progress"], "hidden")
        self.assertEqual(prompt_refresh["inputs"], [components[name]._id for name in
                         ("state", "multi_prompts_gen_type", "image_mode")])
        self.assertEqual(prompt_refresh["outputs"], [components[name]._id for name in
                         ("prompt", "wizard_prompt", "image_end", "prompt_info_label", "wizard_prompt_info_label")])
        frame_refreshes = [item for item in dependencies if item.get("trigger_after") == prompt_refresh["id"]]
        self.assertEqual(len(frame_refreshes), 1)
        frame_refresh = frame_refreshes[0]
        self.assertTrue(frame_refresh["trigger_only_on_success"])
        self.assertFalse(frame_refresh["api_name"])
        self.assertEqual(frame_refresh["show_progress"], "hidden")
        self.assertEqual(frame_refresh["inputs"], [components[name]._id for name in
                         ("state", "video_length", "force_fps", "video_guide", "video_source")])
        self.assertEqual(frame_refresh["outputs"], [components["video_length"]._id])
        js_refreshes = [item for item in dependencies if item.get("trigger_after") == frame_refresh["id"]
                        and item.get("js") == self.plugin.PROMPT_TOOLS_ATTACH_JS]
        self.assertEqual(len(js_refreshes), 1)


if __name__ == "__main__":
    unittest.main()
