"""Regressions for orbit/hold/crane timing, independent scene motion and camera roll."""

import importlib.util
import json
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location("h3_camera_motion_plan", Path(__file__).resolve().parents[1] / "camera_plan.py")
camera = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(camera)


def pose(time, azimuth=90, elevation=0, distance=1):
    return dict(time=time, azimuth=azimuth, elevation=elevation, distance=distance)


def compile_path(path, **changes):
    options = dict(prompt="The robot turns its head and says hello.\noverall_soundscape: Quiet machinery.",
                   frame_count=243, fps=24)
    options.update(changes)
    return camera.compile_plan(json.dumps(path), **options)


def segment_lines(result):
    return [line for line in result["prompt"].splitlines() if line.startswith("[")]


class MotionBoundaryTests(unittest.TestCase):
    def test_reported_one_percent_hold_keeps_timing_and_stops_orbit_through_ascent(self):
        path = [camera.ORIGIN, pose(.5), pose(.51), pose(1, elevation=89)]
        result = compile_path(path)
        first, hold, rise = segment_lines(result)
        self.assertEqual(result["path"], path)
        self.assertAlmostEqual(result["rows"][2][1] - result["rows"][1][1], 2.42 / 24)
        self.assertIn("stop completely before the hold", first)
        self.assertIn("stationary for 0.100833 seconds", hold)
        self.assertIn("azimuth 90 degrees, elevation 0 degrees and distance 1x", hold)
        self.assertIn("keeps its orbit stopped", rise)
        self.assertNotIn("around the main subject", rise)
        self.assertNotIn("the subject keeps the same side", rise)
        self.assertIn("[5.142500s–10.083333s]", rise)
        self.assertEqual(len(result["warnings"]), 1)
        self.assertIn("2.42 frame intervals", result["summary"])
        self.assertIn("robot turns its head and says hello", result["prompt"])
        self.assertIn("overall_soundscape: Quiet machinery.", result["prompt"])

    def test_hold_advice_uses_seconds_at_selected_fps_without_retiming(self):
        for fps in (24, 30, 60):
            end = 242 / fps
            for duration, expected in ((.083333, True), (.499, True), (.5, False), (1, False)):
                with self.subTest(fps=fps, duration=duration):
                    path = [camera.ORIGIN, pose(.25), pose(.25 + duration / end), pose(1, elevation=89)]
                    result = compile_path(path, fps=fps)
                    self.assertEqual(bool(result["warnings"]), expected)
                    self.assertEqual(result["path"], path)
                    self.assertAlmostEqual(result["rows"][2][1] - result["rows"][1][1], duration)

    def test_orbit_context_survives_multiple_holds_before_dolly_or_crane(self):
        for end in (pose(1, elevation=89), pose(1, distance=.65)):
            result = compile_path([camera.ORIGIN, pose(.3), pose(.4), pose(.5), end])
            last = segment_lines(result)[-1]
            self.assertIn("orbit stopped", last)
            self.assertIn("azimuth at 90 degrees", last)
            self.assertNotIn("around the main subject", last)

    def test_stationary_hold_does_not_freeze_scene_or_misclassify_small_motion(self):
        for end in (pose(1, elevation=.000001), pose(1, distance=1.000001), pose(1, azimuth=90.000001)):
            last = segment_lines(compile_path([camera.ORIGIN, pose(.999), end]))[-1]
            self.assertNotIn("stationary", last)
        for frozen in (False, True):
            result = compile_path([camera.ORIGIN, pose(.3), pose(.5), pose(1, elevation=89)], frozen=frozen)
            self.assertEqual("Frozen scene:" in result["prompt"], frozen)
            self.assertEqual(result["warnings"], [])

    def test_smooth_speed_keeps_segment_boundaries_and_no_implicit_dwell(self):
        path = [camera.ORIGIN, pose(.5), pose(1, elevation=89)]
        result = compile_path(path)
        self.assertIn("without rounding off path corners or blending adjacent moves", result["prompt"])
        self.assertIn("Pause only for explicitly timed hold intervals", result["prompt"])
        self.assertEqual(len(segment_lines(result)), 2)
        self.assertNotIn("stationary for", result["prompt"])
        self.assertEqual(result["rows"][1][1], 121 / 24)

    def test_roll_toggle_replaces_marked_and_processed_plans_in_both_modes(self):
        path = [camera.ORIGIN, pose(.5), pose(.6), pose(1, elevation=89)]
        for reference in (False, True):
            first = compile_path(path, reference_mode=reference)
            self.assertIn("Stabilize camera roll", first["prompt"])
            self.assertIn("Use pan and tilt for framing corrections without adding roll", first["prompt"])
            for prompt in (first["prompt"], "\n".join(line for line in first["prompt"].splitlines() if not line.startswith("#"))):
                disabled = compile_path(path, prompt=prompt, reference_mode=reference, stabilize_roll=False)
                self.assertNotIn("Stabilize camera roll", disabled["prompt"])
                self.assertEqual(disabled["prompt"].count("Camera plan:"), 1)
                restored = compile_path(path, prompt=disabled["prompt"], reference_mode=reference)
                self.assertEqual(first["prompt"], restored["prompt"])

    def test_zero_azimuth_rise_never_suggests_circling_subject(self):
        path = [camera.ORIGIN, pose(.5, azimuth=0), pose(1, azimuth=0, elevation=89)]
        rise = segment_lines(compile_path(path))[-1]
        self.assertIn("without orbiting sideways", rise)
        self.assertNotIn("around the main subject", rise)


if __name__ == "__main__":
    unittest.main()
