import hashlib
import importlib.util
import json
from pathlib import Path
import re
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "camera_plan.py"
SPEC = importlib.util.spec_from_file_location("h3_camera_elevation_plan", MODULE_PATH)
camera = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(camera)

SCENE_PROMPT = """subject_definitions: <Subject 1> is a blue geometric marker.
summary: [reference generation] One continuous view of <Subject 1>.
retention_analysis: Keep <Subject 1> unchanged.
detailed_description: [Shot 1] <Subject 1> remains centered and says <d>[English] Hello.</d>
overall_soundscape: Only quiet room tone.
non_diegetic_music: No music."""


def path_json(*poses):
    return json.dumps(poses)


def reported_path():
    return path_json(
        {"time": 0, "azimuth": 0, "elevation": 0, "distance": 1},
        {"time": 0.5, "azimuth": 67.5226, "elevation": 15.0955, "distance": 1.025},
        {"time": 1, "azimuth": 92.2188, "elevation": 81, "distance": 1},
    )


def compile_path(path, prompt=SCENE_PROMPT):
    return camera.compile_plan(
        path,
        prompt=prompt,
        frame_count=124,
        fps=24,
        reference_mode=True,
        interpolation="linear",
    )


def segment_lines(prompt):
    return [line for line in prompt.splitlines() if re.match(r"^\[\d", line)]


def sentence_with(segment, pattern):
    for sentence in re.split(r"(?<=[.!?])\s+", segment):
        if re.search(pattern, sentence, re.IGNORECASE):
            return sentence
    raise AssertionError(f"No sentence matched {pattern!r}: {segment}")


class ElevationCompilerTests(unittest.TestCase):
    def test_reported_rise_separates_camera_travel_lens_tilt_and_near_overhead_endpoint(self):
        first, second = segment_lines(compile_path(reported_path())["prompt"])
        for segment in (first, second):
            travel = sentence_with(segment, r"camera.*(?:rise|rises|rising|climb|climbs|crane|cranes|upward)")
            lens = sentence_with(segment, r"(?:lens.*tilt|tilt.*lens)")
            self.assertNotRegex(travel.lower(), r"\blens\b")
            self.assertRegex(lens.lower(), r"(?:lens.*down|down.*lens)")
        self.assertNotRegex(first.lower(), r"overhead|directly above")
        self.assertRegex(second.lower(), r"near[- ]overhead|almost directly above|nearly directly above")

    def test_reported_small_radial_changes_are_slight_proportional_adjustments_not_dollies(self):
        first, second = segment_lines(compile_path(reported_path())["prompt"])
        for segment, start, end in ((first, "1x", "1.025x"), (second, "1.025x", "1x")):
            lower = segment.lower()
            self.assertNotIn("dolly", lower)
            self.assertIn("slight", lower)
            self.assertRegex(lower, r"(?:%|percent)")
            self.assertIn(start, segment)
            self.assertIn(end, segment)

    def test_five_percent_radial_boundary_is_slight_but_larger_change_remains_explicit(self):
        five_percent = path_json(
            {"time": 0, "azimuth": 0, "elevation": 0, "distance": 1},
            {"time": 1, "azimuth": 0, "elevation": 20, "distance": 1.05},
        )
        larger = path_json(
            {"time": 0, "azimuth": 0, "elevation": 0, "distance": 1},
            {"time": 1, "azimuth": 0, "elevation": 20, "distance": 1.1},
        )
        slight = segment_lines(compile_path(five_percent)["prompt"])[0].lower()
        meaningful = segment_lines(compile_path(larger)["prompt"])[0].lower()
        self.assertNotIn("dolly", slight)
        self.assertIn("slight", slight)
        self.assertRegex(slight, r"(?:5(?:\.0+)?%|5(?:\.0+)? percent)")
        self.assertIn("1x", slight)
        self.assertIn("1.05x", slight)
        self.assertIn("dolly", meaningful)
        self.assertIn("1x", meaningful)
        self.assertIn("1.1x", meaningful)

    def test_physical_vertical_travel_uses_world_height_when_radius_change_reverses_angular_rise(self):
        path = path_json(
            {"time": 0, "azimuth": 0, "elevation": 0, "distance": 1},
            {"time": 0.5, "azimuth": 0, "elevation": 45, "distance": 2},
            {"time": 1, "azimuth": 0, "elevation": 60, "distance": 0.5},
        )
        second = segment_lines(compile_path(path)["prompt"])[1]
        travel = sentence_with(second, r"camera.*(?:descend|descends|descending|lower|lowers|drop|drops|downward)")
        lens = sentence_with(second, r"(?:lens.*tilt|tilt.*lens)")
        self.assertNotRegex(travel.lower(), r"\blens\b|(?:rise|rises|rising|climb|climbs|upward)")
        self.assertRegex(lens.lower(), r"(?:lens.*down|down.*lens)")

    def test_negative_elevation_moves_down_and_looks_up_without_overhead_claim(self):
        path = path_json(
            {"time": 0, "azimuth": 0, "elevation": 0, "distance": 1},
            {"time": 1, "azimuth": 30, "elevation": -25, "distance": 1},
        )
        segment = segment_lines(compile_path(path)["prompt"])[0]
        travel = sentence_with(segment, r"camera.*(?:descend|descends|descending|lower|lowers|drop|drops|downward)")
        lens = sentence_with(segment, r"(?:lens.*tilt|tilt.*lens)")
        self.assertNotRegex(travel.lower(), r"\blens\b|(?:rise|rises|rising|climb|climbs|upward)")
        self.assertRegex(lens.lower(), r"(?:lens.*up|up.*lens)")
        self.assertNotRegex(segment.lower(), r"overhead|directly above")

    def test_linear_orbit_push_pull_prompts_match_030_baseline_with_roll_option_off(self):
        cases = {
            "orbit": ([camera.ORIGIN, {"time": 1, "azimuth": 90, "elevation": 0, "distance": 1}],
                      "8450ce87c7c3423b866213cb7e50c7b599860bf256f932c6799ddad5b1c52698"),
            "push": ([camera.ORIGIN, {"time": 1, "azimuth": 0, "elevation": 0, "distance": 0.65}],
                     "4ec467edba213583fcdbc86b5781caacd1e2b49ec1c43cb6350a8a27e33373a7"),
            "pull": ([camera.ORIGIN, {"time": 1, "azimuth": 0, "elevation": 0, "distance": 1.5}],
                     "0a92dd1c6ec5fa4cffd0c20a3d4a38b42c7750679cf6b0f089240d36a99cbaf1"),
        }
        for name, (path, expected) in cases.items():
            with self.subTest(name=name):
                result = camera.compile_plan(
                    json.dumps(path), prompt="A geometric sculpture remains centered.",
                    frame_count=124, fps=24, interpolation="linear", stabilize_roll=False,
                )
                actual = hashlib.sha256(result["prompt"].encode("utf-8")).hexdigest()
                self.assertEqual(actual, expected)

    def test_elevation_compile_preserves_scene_values_and_timeline(self):
        result = compile_path(reported_path())
        self.assertEqual(result["path"], [
            {"time": 0.0, "azimuth": 0.0, "elevation": 0.0, "distance": 1.0},
            {"time": 0.5, "azimuth": 67.5226, "elevation": 15.0955, "distance": 1.025},
            {"time": 1.0, "azimuth": 92.2188, "elevation": 81.0, "distance": 1.0},
        ])
        self.assertEqual(result["rows"], [
            [1, 0.0, 0.0, 0.0, 1.0],
            [2, 2.5625, 67.5226, 15.0955, 1.025],
            [3, 5.125, 92.2188, 81.0, 1.0],
        ])
        self.assertIn("[0.000000s–2.562500s]", result["prompt"])
        self.assertIn("[2.562500s–5.125000s]", result["prompt"])
        for line in SCENE_PROMPT.splitlines():
            self.assertIn(line, result["prompt"])
        self.assertNotRegex(result["prompt"].lower(), r"\bcat\b|\bstool\b")

    def test_legacy_comment_free_elevation_prompt_reapplies_without_duplicate_plan(self):
        legacy = """integrated_multimodal_description: [Shot 1] A blue marker remains centered.
Camera plan: one continuous take, 124 frames at 24 fps; last frame at 5.125000s. No cuts.
Camera coordinates are relative to the starting view, looking toward the main subject: azimuth 0 degrees, elevation 0 degrees, distance 1x.
Allow the subject and environment to move naturally according to the scene description.
Use a constant rate within each segment, changing direction at its keyframes.
[0.000000s–5.125000s] Raise camera elevation from 0 to 81 degrees; maintain distance 1x.
overall_soundscape: Quiet room tone.
non_diegetic_music: No music."""
        result = camera.compile_plan(
            reported_path(), prompt=legacy, frame_count=124, fps=24, interpolation="linear",
        )
        self.assertEqual(result["prompt"].count("Camera plan:"), 1)
        self.assertNotIn("Raise camera elevation from 0 to 81 degrees", result["prompt"])
        self.assertIn("A blue marker remains centered.", result["prompt"])
        self.assertIn("overall_soundscape: Quiet room tone.", result["prompt"])

    def test_new_elevation_output_is_idempotent(self):
        first = compile_path(reported_path())
        second = compile_path(reported_path(), prompt=first["prompt"])
        self.assertEqual(second["prompt"], first["prompt"])
        self.assertEqual(second["path"], first["path"])
        self.assertEqual(second["rows"], first["rows"])
        comment_free = "\n".join(
            line for line in first["prompt"].splitlines()
            if not line.startswith("# WanGP H3 camera ")
        )
        third = compile_path(reported_path(), prompt=comment_free)
        self.assertEqual(third["prompt"], first["prompt"])
        self.assertEqual(third["path"], first["path"])
        self.assertEqual(third["rows"], first["rows"])


if __name__ == "__main__":
    unittest.main()
