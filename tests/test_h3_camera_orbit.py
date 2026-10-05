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
        self.assertIn("Orbit 90 degrees toward camera right (azimuth 0 to 90 degrees) and come to rest "
                      "a quarter turn from the starting view, without orbiting past azimuth 90 degrees", first)
        self.assertIn("The camera stops orbiting and cranes upward", second)
        self.assertIn("holding azimuth at 90 degrees", second)
        self.assertIn("does not circle the subject or travel sideways", second)
        self.assertIn("the subject keeps the same side toward the camera", second)
        self.assertNotRegex(second, r"continuing|toward camera (?:left|right)|around the main subject")
        self.assertIn("almost directly above the subject", second)

    def test_rest_anchor_only_precedes_a_held_angle(self):
        hold_first, _ = segments(pose(.5, -180), pose(1, -180))
        self.assertIn("come to rest a half turn from the starting view, without orbiting past azimuth -180 degrees", hold_first)
        odd_first, _ = segments(pose(.5, 45, 10), pose(1, 45, 30))
        self.assertIn("come to rest 45 degrees from the starting view", odd_first)
        for first in (segments(pose(.5, 90), pose(1, 180))[0], segments(pose(.5, 90), pose(1, 0))[0],
                      segments(pose(1, 90))[0]):
            self.assertNotIn("come to rest", first)

    def test_returning_to_start_angle_is_a_reversal_not_a_continuation(self):
        second = segments(pose(.5, 90), pose(1, 0, 89))[1]
        self.assertIn("reversing its orbit to travel 90 degrees toward camera left (azimuth 90 to 0 degrees)", second)
        self.assertNotIn("continuing", second)

    def test_same_direction_elevation_orbit_still_continues(self):
        second = segments(pose(.5, 45), pose(1, 90, 40))[1]
        self.assertIn("continuing 45 degrees toward camera right", second)
        self.assertNotIn("revers", second)

    def test_level_reversal_and_hold_before_reversal(self):
        self.assertTrue(segments(pose(.5, 90), pose(1, -30))[1].startswith(
            "[2.562500s–5.125000s] Reverse direction and orbit 120 degrees toward camera left (azimuth 90 to -30 degrees)"))
        first, hold, back = segments(pose(.3, -90), pose(.6, -90), pose(1, 0))
        self.assertIn("toward camera left", first)
        self.assertTrue(hold.endswith("Hold the camera at this pose."))
        self.assertIn("Reverse direction and orbit 90 degrees toward camera right", back)

    def test_dolly_after_orbit_states_that_the_orbit_stops(self):
        second = segments(pose(.5, 90), pose(1, 90, distance=.65))[1]
        self.assertIn("Stop orbiting and hold azimuth at 90 degrees; dolly in from 1x to 0.65x", second)

    def test_rise_without_orbit_says_no_sideways_orbit(self):
        only = segments(pose(1, 0, 20))[0]
        self.assertIn("keeping azimuth at 0 degrees without orbiting sideways", only)
        self.assertNotIn("stops orbiting", only)

    def test_render_validated_023_path_is_byte_identical(self):
        scene = (Path(__file__).with_name("test_h3_camera_elevation.py").read_text(encoding="utf-8")
                 .split('SCENE_PROMPT = """')[1].split('"""')[0])
        path = json.dumps([camera.ORIGIN, pose(.5, 67.5226, 15.0955, 1.025), pose(1, 92.2188, 81)])
        result = camera.compile_plan(path, prompt=scene, frame_count=124, fps=24,
                                     reference_mode=True, interpolation="linear")
        self.assertEqual(hashlib.sha256(result["prompt"].encode("utf-8")).hexdigest(),
                         "76ada2bdf073d3940b0d80876e5526039dd05d73d38b8b60da4d1dd99e898943")

    def test_new_wording_reapplies_without_duplicates(self):
        path = json.dumps([camera.ORIGIN, pose(.5, 90), pose(1, 0, 89)])
        first = camera.compile_plan(path, prompt="A scene.", frame_count=124, fps=24)
        comment_free = "\n".join(line for line in first["prompt"].splitlines() if not line.startswith("#"))
        for prompt in (first["prompt"], comment_free):
            again = camera.compile_plan(path, prompt=prompt, frame_count=124, fps=24)
            self.assertEqual(again["prompt"], first["prompt"])


if __name__ == "__main__":
    unittest.main()
