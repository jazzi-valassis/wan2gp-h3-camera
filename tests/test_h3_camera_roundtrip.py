"""Camera plans through native prompt, window and Gallery boundaries (CPU only)."""

import importlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import gradio as gr
from PIL import Image

from models.minimax_h3.minimax_h3_handler import family_handler
from models.minimax_h3.text_encoder import MiniMaxH3TextEncoder
from shared.utils import prompt_parser
from shared.utils.frame_scheduler import build_default_window_plan, build_frame_scheduler


from plugin_under_test import load_plugin

camera = load_plugin()
ROOT = Path(os.environ["WAN2GP_TEST_HOST"])


def native_plugin(model_type):
    plugin = camera.H3CameraPlugin()
    plugin.get_state_model_type = lambda state: state["model_type"]
    plugin.get_model_family = lambda model: family_handler.query_model_family()
    plugin.get_model_def = lambda model: family_handler.query_model_def(model, {})
    plugin.get_computed_fps = lambda force, model, guide, source: float(force or 24)
    return plugin


def native_arguments(model_type, prompt, frames, start=None):
    values = dict(prompt=prompt, state={"model_type": model_type}, force_fps="30",
                  video_length=frames, video_guide=None, video_source=None, image_mode=0,
                  image_prompt_type="S" if start else "", image_start=start,
                  multi_prompts_gen_type="FG", sliding_window_size=124,
                  sliding_window_discard_last_frames=17, sliding_window_trim_first_frames=2,
                  video_prompt_type="", image_refs=None, image_end=None)
    return [camera.PRESETS["Full orbit"], "animated", "smooth", bool(start),
            *[values[name] for name in camera.FORM_INPUTS]]


class CameraNativeRoundtripTests(unittest.TestCase):
    def test_controlnet_models_are_not_offered_by_camera_planner(self):
        for model in ("minimax_h3_control", "minimax_h3_control_pruned"):
            with self.subTest(model=model):
                plugin = native_plugin(model)
                self.assertFalse(plugin._supports_camera(model))
                visible, timing = plugin.update_visibility(model + "|123", 226, "24", None, None)
                self.assertFalse(visible["visible"])
                self.assertEqual(timing, "{}")

    def test_controlnet_default_guide_cannot_be_applied_as_a_camera_plan(self):
        for model in ("minimax_h3_control", "minimax_h3_control_pruned"):
            with self.subTest(model=model):
                plugin = native_plugin(model)
                args = native_arguments(model, "One continuous scene.", 226)
                args[4 + camera.FORM_INPUTS.index("video_prompt_type")] = "PV"
                with self.assertRaisesRegex(gr.Error, "FL2VA or Ref2VA"):
                    plugin.apply_plan(*args)

    def test_standard_and_pruned_h3_video_models_remain_supported(self):
        for model in ("minimax_h3_fl2va", "minimax_h3_fl2va_pruned",
                      "minimax_h3_ref2va", "minimax_h3_ref2va_pruned"):
            with self.subTest(model=model):
                plugin = native_plugin(model)
                self.assertTrue(plugin._supports_camera(model))
                result = plugin._compile(*native_arguments(model, "One continuous scene.", 226))
                self.assertEqual(result["frame_count"], 226)
                self.assertIn("Orbit 360 degrees", result["prompt"])

    def test_real_model_defaults_survive_native_queue_prompt_processing(self):
        # Run the same template/split/serialize operations used by wgp's queue
        # validation, then reapply that processed prompt as metadata users do.
        for model in ("minimax_h3_fl2va", "minimax_h3_ref2va"):
            source = json.loads((ROOT / "defaults" / (model + ".json")).read_text(encoding="utf-8"))["prompt"]
            plugin = native_plugin(model)
            first = plugin._compile(*native_arguments(model, source, 226))
            processed, error = prompt_parser.process_template(first["prompt"])
            self.assertEqual(error, "")
            units = prompt_parser.split_prompt_units(processed, "FG")
            self.assertEqual(len(units), 1)
            serialized = prompt_parser.serialize_prompt_units(processed, units, "FG")
            self.assertEqual(serialized.count("Camera plan:"), 1)
            self.assertNotIn("# WanGP H3 camera", serialized)
            self.assertIn("<d>[English]", serialized)
            self.assertIn("overall_soundscape:", serialized)
            self.assertIn("non_diegetic_music:", serialized)
            repeated = plugin._compile(*native_arguments(model, serialized, 226))
            self.assertEqual(first["prompt"], repeated["prompt"])

    def test_real_model_geometry_preserves_single_window_with_and_without_start(self):
        for model in ("minimax_h3_fl2va", "minimax_h3_ref2va"):
            plugin = native_plugin(model)
            definition = plugin.get_model_def(model)
            defaults = definition["sliding_window_defaults"]
            for frame_count in (107, 124, 226, 481):
                for has_start in (False, True):
                    with self.subTest(model=model, frames=frame_count, start=has_start):
                        start = [(Image.new("RGB", (32, 32), "red"), None)] if has_start else None
                        plan = plugin._compile(*native_arguments(model, "One continuous scene.", frame_count, start))
                        geometry = dict(total_frames=plan["frame_count"], window_size=plan["sliding_window_size"],
                            default_overlap=defaults["overlap_default"], minimum=definition["frames_minimum"],
                            step=definition["frames_steps"], frame_offset=definition["frames_offset"],
                            overlap_offset=defaults["overlap_offset"], max_overlap=defaults["overlap_max"],
                            output_frame_policy=definition.get("frame_scheduler_output_policy"),
                            preserve_exact_output_frames=definition.get("image_end_frame_position", False),
                            discard_last_frames=plan["sliding_window_discard_last_frames"])
                        scheduler, error = build_frame_scheduler(
                            prompt_parser.split_prompt_units(plan["prompt"], "FG"), fps=plan["fps"],
                            first_window_overlap_frames=int(has_start), initial_shared_frames=int(has_start),
                            allow_new_shot=True, **geometry)
                        self.assertIsNone(error)
                        self.assertFalse(scheduler["active"])
                        windows = build_default_window_plan(first_window_overlap=int(has_start),
                            first_window_available_overlap=int(has_start), initial_shared_frames=int(has_start), **geometry)
                        self.assertEqual(len(windows), 1)
                        self.assertEqual(windows[0]["frame_num"], frame_count)
                        self.assertEqual(windows[0]["trim_last_frames"], 0)
                        self.assertEqual(windows[0]["discard_last_frames"], 0)
                        self.assertAlmostEqual(plan["rows"][-1][1], (frame_count - 1) / 30)

    def test_native_text_encoder_labels_start_end_then_references(self):
        # Exercise the real presentation method without constructing model weights.
        fake_encoder = SimpleNamespace(_token_ids=lambda text: [text])
        start, end, reference = object(), object(), object()
        prompt = "<Subject 1> from <Picture 3> speaks with <Audio 1>."
        entries = MiniMaxH3TextEncoder._presentation(fake_encoder, prompt, [
            {"type": "image", "frames": start}, {"type": "image", "frames": end},
            {"type": "image", "frames": reference}, {"type": "audio"},
        ])
        labels = [entry for entry in entries if isinstance(entry, str)]
        self.assertEqual(labels, ["<Picture 1>: ", "<Picture 2>: ", "<Picture 3>: ", "<Audio 1>: ", prompt])
        self.assertEqual([entry["frames"] for entry in entries if isinstance(entry, dict)], [start, end, reference])

    def test_closed_loop_roundtrip_through_real_gradio_galleries(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "start.png"
            Image.new("RGB", (32, 32), "red").save(image_path)
            # WanGP's native start/end Galleries use filepath values. Passing
            # those paths through preserves the original bytes and caption.
            gallery = gr.Gallery(type="filepath")
            start = gallery.preprocess(gallery.postprocess([(str(image_path), "Start")]))
            plugin = native_plugin("minimax_h3_fl2va")
            changes = dict(zip(camera.FORM_OUTPUTS, plugin.apply_plan(*native_arguments(
                "minimax_h3_fl2va", "One continuous scene.", 124, start))))
            end = gallery.preprocess(gallery.postprocess(changes["image_end"]["value"]))
            self.assertEqual(end[0][1], "Start")
            self.assertEqual(Path(end[0][0]).read_bytes(), Path(start[0][0]).read_bytes())
            self.assertEqual(changes["image_prompt_type"]["value"], "SE")


if __name__ == "__main__":
    unittest.main()
