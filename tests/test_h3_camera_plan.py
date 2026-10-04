"""CPU-only checks for camera validation, native H3 timing and prompt fidelity."""

import importlib.util
import json
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "camera_plan.py"
SPEC = importlib.util.spec_from_file_location("h3_camera_plan", MODULE_PATH)
camera = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(camera)


class CameraPlanTests(unittest.TestCase):
    def compile(self, path=None, **kwargs):
        values = dict(prompt="A singer smiles and says <d>[English] Hello.</d> Soft piano music.", frame_count=124, fps=24)
        values.update(kwargs)
        return camera.compile_plan(path or camera.DEFAULT_PATH, **values)

    def test_native_frame_grid_and_nondefault_fps(self):
        result = self.compile(frame_count=121, fps=30)
        self.assertEqual(result["frame_count"], 124)
        self.assertAlmostEqual(result["duration_seconds"], 124 / 30)
        self.assertEqual(result["rows"][0][1], 0)
        self.assertAlmostEqual(result["rows"][-1][1], 123 / 30)
        self.assertIn("4.100000s", result["prompt"])

    def test_presets_are_valid(self):
        for preset in camera.PRESETS.values():
            camera.validate_path(preset)

    def test_corrupt_json_and_wrong_shape(self):
        for bad in ("", "{broken", "null", "{}", "[]", "[{}]", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                camera.validate_path(bad)

    def test_each_numeric_field_must_be_finite(self):
        for field in camera.FIELDS:
            for bad in (float("nan"), float("inf"), float("-inf"), True, "1", None):
                path = json.loads(camera.DEFAULT_PATH)
                path[-1][field] = bad
                with self.subTest(field=field, value=bad), self.assertRaises(ValueError):
                    camera.validate_path(json.dumps(path))

    def test_bounds_origin_and_end_are_strict(self):
        for field, bad in (("time", -0.1), ("time", 1.1), ("time", .9), ("elevation", -89.1),
                           ("elevation", 89.1), ("distance", .09), ("distance", 4.1), ("azimuth", 11521)):
            path = json.loads(camera.DEFAULT_PATH)
            path[-1][field] = bad
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                camera.validate_path(json.dumps(path))
        for field in camera.FIELDS:
            path = json.loads(camera.DEFAULT_PATH)
            path[0][field] += 0.00000001
            with self.subTest(origin=field), self.assertRaises(ValueError):
                camera.validate_path(json.dumps(path))

    def test_order_count_unknown_fields_and_travel(self):
        origin, end = json.loads(camera.DEFAULT_PATH)
        invalid = [
            [origin, dict(end, time=0), end],
            [origin, dict(end, time=.8), dict(end, time=.4), end],
            [origin] + [dict(end, time=i / 24) for i in range(1, 25)],
            [origin, dict(end, unexpected=1)],
            [origin, dict(end, azimuth=10000, time=.5), dict(end, azimuth=-10000)],
        ]
        for path in invalid:
            with self.subTest(path=path), self.assertRaises(ValueError):
                camera.validate_path(json.dumps(path))

    def test_explicit_hold_has_real_timing(self):
        origin, end = json.loads(camera.DEFAULT_PATH)
        result = self.compile(json.dumps([origin, dict(end, time=.5), end]))
        self.assertAlmostEqual(result["rows"][1][1], 2.5625)
        self.assertIn("[2.562500s–5.125000s] Hold the camera at this pose.", result["prompt"])

    def test_full_loop_requires_real_turn_and_start_image(self):
        with self.assertRaisesRegex(ValueError, "Start Image"):
            self.compile(camera.PRESETS["Full orbit"], close_loop=True)
        for azimuth, elevation, distance in ((0, 0, 1), (359.5, 0, 1), (720.5, 0, 1), (360, .01, 1), (360, 0, 1.01)):
            path = json.loads(camera.PRESETS["Full orbit"])
            path[-1].update(azimuth=azimuth, elevation=elevation, distance=distance)
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.compile(json.dumps(path), close_loop=True, has_start_image=True)
        for turns in (-2, -1, 1, 2):
            path = json.loads(camera.PRESETS["Full orbit"])
            path[-1]["azimuth"] = turns * 360
            result = self.compile(json.dumps(path), close_loop=True, has_start_image=True)
            self.assertTrue(result["close_loop"])
            self.assertIn("<Picture 2>", result["prompt"])
            self.assertIn("does not guarantee", result["summary"])

    def test_plain_prompt_keeps_dialogue_music_and_subject_motion(self):
        result = self.compile()
        self.assertIn("<d>[English] Hello.</d> Soft piano music.", result["prompt"])
        self.assertIn("Allow the subject and environment to move", result["prompt"])
        self.assertNotIn("Silence", result["prompt"])
        self.assertNotIn("non_diegetic_music: N/A", result["prompt"])
        self.assertNotIn("<Picture", result["prompt"])
        self.assertEqual(result["multi_prompts_gen_type"], "FG")

    def test_frozen_scene_is_opt_in_without_muting_audio(self):
        result = self.compile(frozen=True)
        self.assertIn("Frozen scene", result["prompt"])
        self.assertIn("Preserve the requested soundtrack", result["prompt"])
        self.assertIn("Soft piano music", result["prompt"])

    def test_fl2va_and_ref2va_fields_preserve_audio_and_labels(self):
        for ref, visual in ((False, "integrated_multimodal_description"), (True, "detailed_description")):
            prefix = ("subject_definitions: <Subject 2> is the singer from <Picture 3>.\nsummary: [reference generation + audio reference] Portrait.\nretention_analysis: <Audio 1>: reference - voice.\n" if ref else "")
            original = prefix + f"{visual}: [Shot 1] The singer (S1) says <d>[French] Bonjour.</d>\n\noverall_soundscape: Rain. <Audio 1> is the voice reference.\nnon_diegetic_music: Warm cello."
            result = self.compile(prompt=original, reference_mode=ref)
            self.assertIn("overall_soundscape: Rain. <Audio 1> is the voice reference.", result["prompt"])
            self.assertIn("non_diegetic_music: Warm cello.", result["prompt"])
            self.assertIn("<d>[French] Bonjour.</d>", result["prompt"])
            self.assertNotIn("\n\n", result["prompt"])
            if ref:
                self.assertIn("<Subject 2> is the singer from <Picture 3>.", result["prompt"])
                self.assertIn("[reference generation + audio reference]", result["prompt"])

    def test_repeated_apply_is_idempotent_and_changed_plan_replaces(self):
        for ref in (False, True):
            first = self.compile(reference_mode=ref, has_start_image=True)
            second = self.compile(prompt=first["prompt"], reference_mode=ref, has_start_image=True)
            self.assertEqual(first["prompt"], second["prompt"])
            changed = self.compile(camera.PRESETS["Pull back"], prompt=second["prompt"], reference_mode=ref)
            self.assertEqual(changed["prompt"].count("Camera plan:"), 1)
            self.assertNotIn("orbit 90 degrees", changed["prompt"])
            self.assertNotIn("<Picture 1>", changed["prompt"])
            self.assertIn("dolly back", changed["prompt"].lower())

    def test_loop_alignment_is_removed_when_disabled(self):
        first = self.compile(camera.PRESETS["Full orbit"], close_loop=True, has_start_image=True)
        second = self.compile(prompt=first["prompt"], has_start_image=True)
        self.assertNotIn("<Picture 2>", second["prompt"])
        self.assertEqual(second["prompt"].count("<Picture 1>"), 1)

    def test_processed_prompt_without_comments_can_be_reapplied(self):
        for ref in (False, True):
            for closed in (False, True):
                first = self.compile(camera.PRESETS["Full orbit"], reference_mode=ref, close_loop=closed, has_start_image=True)
                processed = "\n".join(line for line in first["prompt"].splitlines() if not line.startswith("#"))
                second = self.compile(camera.PRESETS["Full orbit"], prompt=processed, reference_mode=ref, close_loop=closed, has_start_image=True)
                self.assertEqual(first["prompt"], second["prompt"])
                self.assertEqual(second["prompt"].count("Camera plan:"), 1)

    def test_reject_ambiguous_multishot_or_wrong_structured_mode(self):
        for bad in ("[/duration=124] Scene", "Scene\n---\nOther scene", "[Shot 1] Scene. [Shot 2] Cut.",
                    "integrated_multimodal_description: Scene\nintegrated_multimodal_description: Another scene"):
            with self.subTest(prompt=bad), self.assertRaises(ValueError):
                self.compile(prompt=bad)
        with self.assertRaisesRegex(ValueError, "Ref2VA"):
            self.compile(prompt="detailed_description: Scene")
        with self.assertRaisesRegex(ValueError, "FL2VA"):
            self.compile(prompt="integrated_multimodal_description: Scene", reference_mode=True)

    def test_invalid_timing_options_and_markers(self):
        for name, bad in (("fps", 0), ("fps", float("nan")), ("fps", True), ("frame_count", 4),
                          ("frame_count", 5.5), ("frame_count", float("inf")), ("interpolation", "cubic"), ("frozen", "false")):
            with self.subTest(name=name, value=bad), self.assertRaises(ValueError):
                self.compile(**{name: bad})
        with self.assertRaisesRegex(ValueError, "marker"):
            self.compile(prompt="# WanGP H3 camera plan begin\nAn incomplete block.")


if __name__ == "__main__":
    unittest.main()
