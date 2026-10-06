"""Regressions for the 0.3.0 report: a 90-degree orbit with a 48-degree rise rendered as a spinning top-down shot.

The 0.2.9 plan named moves it wanted to prevent (full turns, overhead, twisting,
rotation around the lens axis, "no orbit ... or roll") and placed the 48-degree
view "above the subject". H3 performed those moves. These tests keep the plan
limited to the requested motion and bound the whole take.
"""

import importlib.util
import json
from pathlib import Path
import re
import unittest


SPEC = importlib.util.spec_from_file_location("h3_camera_spin_plan", Path(__file__).resolve().parents[1] / "camera_plan.py")
camera = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(camera)

LEGACY_029 = (
    'integrated_multimodal_description: A woman sits on a crate.\n'
    'Camera plan: one continuous take, 124 frames at 24 fps; last frame at 5.125000s. No cuts.\n'
    'Camera coordinates are relative to the starting view, looking toward the main subject: azimuth 0 degrees, elevation 0 degrees, distance 1x. Positive azimuth means the camera travels to its right around the subject; angles stay unwrapped across full turns. Move the camera through the scene with natural parallax while keeping the subject framed. Stabilize camera roll throughout the take: keep the camera upright relative to world up, with no banking, Dutch angle or rotation around the lens axis. Use pan and tilt for framing corrections without adding roll. As the view approaches overhead, preserve the screen orientation without twisting.\n'
    'Allow the subject and environment to move naturally according to the scene description, with requested speech and action synchronized to the audio.\n'
    'Ease smoothly into and out of each segment; reach each keyframe and stop before starting the next segment. Smooth the speed within each segment, without rounding off path corners or blending adjacent moves. Pause only for explicitly timed hold intervals.\n'
    '[0.000000s–2.562500s] The camera cranes upward through a rising arc around the main subject, continuing 45 degrees toward camera right (azimuth 0 to 45 degrees). Its viewing elevation changes from 0 to 48 degrees while the lens tilts downward to keep the subject framed. At the end of this segment, the view is elevated above the subject, looking down toward it. Dolly in from 1x to 0.75x the starting distance (about 25%). Keep the focal length fixed throughout this move.\n'
    '[2.562500s–5.125000s] The camera descends through a falling arc around the main subject, continuing 45 degrees toward camera right (azimuth 45 to 90 degrees). Its viewing elevation changes from 48 to 0 degrees while the lens tilts upward to keep the subject framed. At the end of this segment, the view returns to the starting elevation. Dolly in from 0.75x to 0.5x the starting distance (about 33.3%). Keep the focal length fixed throughout this move.\n'
    "overall_soundscape: Follow the scene's audio and dialogue instructions; do not add unrequested sound.\n"
    "non_diegetic_music: Follow the scene's music instructions; do not add unrequested music."
)

PRIMERS = r"overhead|above the subject|full turn|unwrapped|twist|rotation|\broll\b|banking|dutch|no orbit|without orbiting|circle"


def pose(time, azimuth, elevation=0, distance=1):
    return dict(time=time, azimuth=azimuth, elevation=elevation, distance=distance)


def plan(path, **changes):
    options = dict(prompt="[Shot 1] A woman sits on a crate on a rooftop at night.", frame_count=239, fps=24,
                   reference_mode=True, has_start_image=True)
    options.update(changes)
    prompt = camera.compile_plan(json.dumps(path), **options)["prompt"]
    return prompt.split("# WanGP H3 camera plan begin\n")[1].split("\n# WanGP H3 camera plan end")[0]


def reported_path():
    return [camera.ORIGIN, pose(.5, 45, 48, .75), pose(5.541666666666666 / 10.083333333333334, 45, 48, .75),
            pose(1, 90, 0, .5)]


def segments(text):
    return [line for line in text.splitlines() if re.match(r"^\[\d", line)]


class SpinReportTests(unittest.TestCase):
    def test_reported_path_names_only_requested_motion(self):
        text = plan(reported_path())
        self.assertNotRegex(text.lower(), PRIMERS)
        self.assertIn("In total the camera arcs a quarter turn (90 degrees) to the right around the subject, "
                      "ending on a side view of the subject.", text)
        self.assertIn("Its highest viewpoint is a high angle, looking down at about 48 degrees.", text)
        first, hold, last = segments(text)
        self.assertIn("The camera slowly arcs an eighth of a turn (45 degrees) to the right around the subject while rising.", first)
        self.assertNotIn("continu", first)
        self.assertIn("It settles at a high angle, looking down at about 48 degrees, on a three-quarter view", first)
        self.assertIn("Hold the camera completely stationary for 0.500000 seconds", hold)
        self.assertIn("continues its arc around the subject to the right, adding an eighth of a turn (45 degrees) "
                      "while descending", last)
        self.assertIn("It ends at eye level, on a side view of the subject.", last)

    def test_dense_keyframes_keep_the_total_bound_and_do_not_stop_between_them(self):
        path = [camera.ORIGIN]
        for second in range(1, 11):
            elevation = 48 * min(second, 10 - second) / 5
            path.append(pose(second / 10, 9 * second, elevation, 1 - .05 * second))
        text = plan(path)
        self.assertNotRegex(text.lower(), PRIMERS + r"|\bstop")
        self.assertEqual(text.count("In total the camera arcs a quarter turn (90 degrees) to the right"), 1)
        lines = segments(text)
        self.assertEqual(len(lines), 10)
        self.assertNotIn("continu", lines[0])
        for line in lines[1:]:
            self.assertIn("continues its arc around the subject to the right, adding 9 degrees", line)
            self.assertIn("The camera slowly", line)

    def test_overhead_is_only_named_when_the_path_goes_overhead(self):
        self.assertNotIn("overhead", plan([camera.ORIGIN, pose(1, 0, 74)]).lower())
        steep = plan([camera.ORIGIN, pose(1, 0, 75)])
        self.assertIn("Near the overhead view, keep the picture's orientation steady.", steep)
        self.assertIn("almost directly above the subject", steep)
        self.assertNotIn("Near the overhead view", plan([camera.ORIGIN, pose(1, 0, 75)], stabilize_roll=False))

    def test_reversal_and_multi_turn_totals(self):
        back = plan([camera.ORIGIN, pose(.5, 90), pose(1, 0)])
        self.assertIn("In total the camera travels 180 degrees around the subject, ending on the starting side of the subject.", back)
        loop = plan([camera.ORIGIN, pose(1, 720)])
        self.assertIn("orbits two full turns (720 degrees) to the right around the subject", loop)
        self.assertIn("ending on the starting side of the subject again", loop)

    def test_comment_free_029_plan_is_replaced_once(self):
        path = [camera.ORIGIN, pose(.5, 45, 48, .75), pose(1, 90, 0, .5)]
        result = camera.compile_plan(json.dumps(path), prompt=LEGACY_029, frame_count=124, fps=24)["prompt"]
        self.assertEqual(result.count("Camera plan:"), 1)
        self.assertNotRegex(result.lower(), PRIMERS)
        self.assertIn("integrated_multimodal_description: A woman sits on a crate.", result)
        self.assertIn("non_diegetic_music: Follow the scene's music instructions", result)

    def test_view_and_angle_names(self):
        self.assertEqual(camera._view(-45), "a three-quarter view of the subject")
        self.assertEqual(camera._view(270), "a side view of the subject")
        self.assertEqual(camera._view(180), "the opposite side of the subject")
        self.assertEqual(camera._angle(-30), "a low angle, looking up at about 30 degrees")
        self.assertEqual(camera._angle(65), "a steep high angle, looking down at about 65 degrees")
        self.assertEqual(camera._angle(0), "eye level")


if __name__ == "__main__":
    unittest.main()
