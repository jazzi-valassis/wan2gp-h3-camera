"""Orbit angles are absolute: holding an angle stops the orbit and turning back reverses it."""

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "camera_plan.py"
SPEC = importlib.util.spec_from_file_location("h3_camera_orbit_plan", MODULE_PATH)
camera = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(camera)


def pose(time, azimuth, elevation=0, distance=1):
    return {"time": time, "azimuth": azimuth, "elevation": elevation, "distance": distance}


def segments(*poses, prompt="A woman stands on a railway platform."):
    result = camera.compile_plan(json.dumps([camera.ORIGIN, *poses]), prompt=prompt,
                                 frame_count=124, fps=24, interpolation="linear")
    return [line for line in result["prompt"].splitlines() if re.match(r"^\[\d", line)]


class OrbitDirectionTests(unittest.TestCase):
    def test_reported_orbit_then_overhead_crane_stops_orbiting(self):
        first, second = segments(pose(.5, 90), pose(1, 90, 89))
        self.assertIn("arcs a quarter turn (90 degrees) to the right around the subject", first)
        self.assertIn("It settles at eye level, on a side view of the subject.", first)
        self.assertIn("The camera stops arcing and", second)
        self.assertIn("rises straight up, staying on a side view of the subject", second)
        self.assertNotRegex(second, r"continu|to the (?:left|right)|around the (?:main )?subject")
        self.assertIn("almost directly above the subject", second)

    def test_rest_wording_only_precedes_a_held_angle(self):
        hold_first, _ = segments(pose(.5, -180), pose(1, -180))
        self.assertIn("orbits a half turn (180 degrees) to the left around the subject", hold_first)
        self.assertIn("It settles at eye level, on the opposite side of the subject.", hold_first)
        odd_first, _ = segments(pose(.5, 45, 10), pose(1, 45, 30))
        self.assertIn("It settles at a slightly raised angle", odd_first)
        self.assertIn("on a three-quarter view of the subject", odd_first)
        for first in (segments(pose(.5, 90), pose(1, 180))[0], segments(pose(.5, 90), pose(1, 0))[0],
                      segments(pose(1, 90))[0]):
            self.assertNotIn("settles", first)

    def test_returning_to_start_angle_is_a_reversal_not_a_continuation(self):
        second = segments(pose(.5, 90), pose(1, 0, 89))[1]
        self.assertIn("reverses direction and arcs a quarter turn (90 degrees) back to the left around the subject", second)
        self.assertNotIn("continu", second)

    def test_same_direction_elevation_orbit_still_continues(self):
        first, second = segments(pose(.5, 45), pose(1, 90, 40))
        self.assertNotIn("continu", first)
        self.assertIn("continues its arc around the subject to the right, adding an eighth of a turn (45 degrees)", second)
        self.assertNotIn("revers", second)

    def test_level_reversal_and_hold_before_reversal(self):
        self.assertTrue(segments(pose(.5, 90), pose(1, -30))[1].startswith(
            "[2.562500s–5.125000s] The camera quickly reverses direction and arcs 120 degrees back to the left around the subject."))
        first, hold, back = segments(pose(.3, -90), pose(.6, -90), pose(1, 0))
        self.assertIn("to the left", first)
        self.assertIn("Hold the camera completely stationary", hold)
        self.assertIn("reverses direction and arcs a quarter turn (90 degrees) back to the right", back)

    def test_dolly_after_orbit_states_that_the_orbit_stops(self):
        second = segments(pose(.5, 90), pose(1, 90, distance=.65))[1]
        self.assertIn("The camera stops arcing and slowly dollies in from 1x to 0.65x the starting distance "
                      "(about 35%), staying on a side view of the subject.", second)

    def test_rise_without_orbit_names_only_the_rise(self):
        only = segments(pose(1, 0, 20))[0]
        self.assertIn("The camera slowly rises straight up.", only)
        self.assertNotRegex(only, r"orbit|arc|stops")

    def test_render_validated_023_linear_path_matches_030_baseline_with_roll_option_off(self):
        scene = (Path(__file__).with_name("test_h3_camera_elevation.py").read_text(encoding="utf-8")
                 .split('SCENE_PROMPT = """')[1].split('"""')[0])
        path = json.dumps([camera.ORIGIN, pose(.5, 67.5226, 15.0955, 1.025), pose(1, 92.2188, 81)])
        result = camera.compile_plan(path, prompt=scene, frame_count=124, fps=24,
                                     reference_mode=True, interpolation="linear", stabilize_roll=False)
        self.assertEqual(hashlib.sha256(result["prompt"].encode("utf-8")).hexdigest(),
                         "9d6d8f5d2444b9f1e8389a5e5f96176cd3508df48d575e56b96f0fcd4c459b9f")

    def test_new_wording_reapplies_without_duplicates(self):
        path = json.dumps([camera.ORIGIN, pose(.5, 90), pose(1, 0, 89)])
        first = camera.compile_plan(path, prompt="A scene.", frame_count=124, fps=24)
        comment_free = "\n".join(line for line in first["prompt"].splitlines() if not line.startswith("#"))
        for prompt in (first["prompt"], comment_free):
            again = camera.compile_plan(path, prompt=prompt, frame_count=124, fps=24)
            self.assertEqual(again["prompt"], first["prompt"])


if __name__ == "__main__":
    unittest.main()
