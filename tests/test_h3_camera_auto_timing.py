"""Automatic hold timing: plan parsing, correction outcomes and the output-folder watcher."""
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from test_h3_camera_timing import timing
from test_h3_camera_plugin import make_plugin, camera


PLAN = ("detailed_description: [Shot 1] A woman sits on a crate.\n"
        "Camera plan: one continuous take, 243 frames at 24 fps; last frame at 10.083333s. No cuts.\n"
        "[0.000000s–5.041667s] The camera slowly arcs an eighth of a turn (45 degrees) to the right around the subject.\n"
        "[5.041667s–5.541667s] Hold the camera completely stationary for 0.500000 seconds on a three-quarter view.\n"
        "[5.541667s–10.083333s] The camera slowly continues its arc to the right.\n")


def evidence(source_marks, frames=243):
    first, last = source_marks[1:3]
    return dict(media=dict(frames=frames), source_marks=source_marks, target_marks=[0, 121, 133, frames - 1],
                motion=[dict(frame=i, pixels=.01 if first <= i < last else 3) for i in range(frames - 1)])


class PlannedHoldTests(unittest.TestCase):
    def test_reads_the_single_interior_hold_of_a_compiled_plan(self):
        self.assertEqual(timing.planned_hold(PLAN, 243), (121, 133))
        self.assertIsNone(timing.planned_hold(PLAN, 226))
        two = PLAN + "[8.000000s–8.500000s] Hold the camera completely stationary for 0.500000 seconds.\n"
        self.assertIsNone(timing.planned_hold(two, 243))
        self.assertIsNone(timing.planned_hold("A cat sleeps.", 243))
        self.assertIsNone(timing.planned_hold(PLAN.replace("[5.041667s", "[0.000000s"), 243))

    def test_skips_clips_without_a_plan_before_probing(self):
        with patch.object(timing, "read_generation_prompt", return_value="A cat sleeps."), \
                patch.object(timing, "probe_video", side_effect=AssertionError("probed")):
            self.assertIsNone(timing.auto_correct("cat.mp4", "."))

    def test_exact_and_rejected_outcomes_leave_the_source_alone(self):
        info = dict(path="clip.mp4", frames=243)
        with patch.object(timing, "read_generation_prompt", return_value=PLAN), patch.object(timing, "probe_video", return_value=info), \
                patch.object(timing, "export_retimed", side_effect=AssertionError("exported")):
            with patch.object(timing, "inspect_frames", return_value=evidence([0, 121, 133, 242])):
                self.assertEqual(timing.auto_correct("clip.mp4", ".")["status"], "exact")
            with patch.object(timing, "inspect_frames", return_value=evidence([0, 40, 200, 242])):
                result = timing.auto_correct("clip.mp4", ".")
            self.assertEqual(result["status"], "rejected")
            self.assertIn("speed limit", result["reason"])
            self.assertIsNone(result["output"])


def moving_pairs(frames, still, frozen=True):
    """Motion evidence: camera moving except in ``still`` ranges; whole frames frozen there unless told otherwise."""
    pairs = []
    for i in range(frames - 1):
        quiet = any(a <= i < b for a, b in still)
        pairs.append(dict(frame=i, pixels=.01 if quiet else 3, change=(.2 if frozen else 5) if quiet else 9))
    return pairs


class EndingTests(unittest.TestCase):
    def test_settled_tail_needs_a_frozen_still_ending_after_the_hold(self):
        self.assertEqual(timing.settled_tail(moving_pairs(243, [(107, 136), (224, 242)]), 136, 243), 224)
        self.assertIsNone(timing.settled_tail(moving_pairs(243, [(107, 136), (224, 242)], frozen=False), 136, 243))
        self.assertIsNone(timing.settled_tail(moving_pairs(243, [(107, 136), (238, 242)]), 136, 243))
        self.assertIsNone(timing.settled_tail(moving_pairs(243, [(107, 136)]), 136, 243))

    def correct(self, still, prompt=PLAN):
        info = dict(path="clip.mp4", frames=243)
        evidence = dict(media=dict(frames=243), motion=moving_pairs(243, still),
                        source_marks=[0, *still[0], 242], target_marks=[0, 121, 133, 242])
        exported = {}

        def export(source, source_marks, target_marks, staging, audio, *args):
            exported.update(source=source_marks, target=target_marks, audio=audio)
            video, report = Path(staging) / "clip_timed_x.mp4", Path(staging) / "clip_timed_x.timing.json"
            video.write_bytes(b"v")
            report.write_text("{}")
            return str(video), str(report), dict(repeated_source_frames=0)

        after = dict(media=dict(frames=243), motion=moving_pairs(243, [(121, 133)]),
                     source_marks=[0, 121, 133, 242], target_marks=[0, 121, 133, 242])
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(timing, "read_generation_prompt", return_value=prompt), \
                patch.object(timing, "probe_video", return_value=info), \
                patch.object(timing, "export_retimed", side_effect=export), \
                patch.object(timing, "inspect_frames", side_effect=[evidence, after]):
            result = timing.auto_correct("clip.mp4", folder)
        return result, exported

    def test_an_early_frozen_ending_is_moved_to_the_last_frame(self):
        result, exported = self.correct([(107, 136), (224, 242)])
        self.assertEqual(result["status"], "corrected")
        self.assertEqual(exported, dict(source=[0, 107, 136, 224, 242], target=[0, 121, 133, 241, 242], audio="preserve"))
        self.assertEqual(result["audio"], "preserve")

    def test_dialogue_keeps_its_ending_and_retimes_audio_for_lip_sync(self):
        spoken = PLAN.replace("A woman sits on a crate.", "A woman sits on a crate and says <d>[English] Hello.</d>")
        result, exported = self.correct([(107, 136), (224, 242)], spoken)
        self.assertEqual(exported, dict(source=[0, 107, 136, 242], target=[0, 121, 133, 242], audio="retime"))
        self.assertEqual(result["audio"], "retime")

    def test_the_ending_is_left_alone_when_moving_it_would_slow_the_move_too_much(self):
        # Ending at 241 would play frames 180-230 over 133-241 (0.46x); the hold-only retime stays at 0.57x.
        result, exported = self.correct([(98, 180), (230, 242)])
        self.assertEqual(result["status"], "corrected")
        self.assertEqual(exported, dict(source=[0, 98, 180, 242], target=[0, 121, 133, 242], audio="preserve"))
        self.assertIn("speed limit", result["ending_left_unchanged"])

    def test_retimed_audio_is_trimmed_past_four_times_and_faded_at_joins(self):
        chains = timing.audio_filter([0, 98, 174, 242], [0, 121, 133, 242], 24, 243, 32000).split(";")
        self.assertIn("atempo", chains[0])
        self.assertNotIn("atempo", chains[1])
        self.assertIn("atempo", chains[2])
        # Short fades at every join between sections (the last displayed frame is its own section),
        # none at the clip's own start or end.
        self.assertNotIn("afade", chains[0].split("areverse")[0])
        self.assertTrue(all("afade" in chain for chain in chains[:4]))
        self.assertNotIn("areverse", chains[3])


class WatcherTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.watcher = camera._AutoTiming()
        self.watcher.started = time.time() - 600

    def tearDown(self):
        self.folder.cleanup()

    def clip(self, name, age):
        path = Path(self.folder.name) / name
        path.write_bytes(b"video")
        moment = time.time() - age
        os.utime(path, (moment, moment))
        return str(path)

    def wait(self):
        for _ in range(100):
            if not self.watcher.busy:
                return
            time.sleep(.02)
        self.fail("correction did not finish")

    def test_each_new_render_is_corrected_once_and_outputs_are_ignored(self):
        render = self.clip("render.mp4", 30)
        self.clip("render_timed_0123456789.mp4", 20)
        self.clip("writing.mp4", 1)
        self.clip("old.mp4", 3600)
        output = str(Path(self.folder.name) / "render_timed_abc.mp4")
        corrected = dict(status="corrected", output=output, planned_hold_zero_based=[121, 133])
        with patch.object(camera.camera_timing, "auto_correct", return_value=corrected) as correct:
            self.assertIn("render.mp4", self.watcher.poll(self.folder.name, True))
            self.wait()
            self.assertIsNone(self.watcher.poll(self.folder.name, True))
        correct.assert_called_once_with(render, self.folder.name)
        version, video, note = self.watcher.latest
        self.assertEqual((version, video), (1, output))
        self.assertIn("frames **122-134**", note)
        self.assertIn("original is unchanged", note)

    def test_disabled_watcher_and_non_plan_clips_publish_nothing(self):
        self.clip("render.mp4", 30)
        with patch.object(camera.camera_timing, "auto_correct", return_value=None) as correct:
            self.assertIsNone(self.watcher.poll(self.folder.name, False))
            correct.assert_not_called()
            self.watcher.poll(self.folder.name, True)
            self.wait()
        self.assertEqual(self.watcher.latest, (0, None, ""))

    def test_tick_shows_each_result_once(self):
        plugin = make_plugin()
        plugin.save_path = self.folder.name
        plugin._auto_timing = self.watcher
        self.watcher.latest = (3, "clip_timed.mp4", "Corrected.")
        video, status, shown = plugin.auto_timing_tick(True, 0)
        self.assertEqual((video["value"], status["value"], shown), ("clip_timed.mp4", "Corrected.", 3))
        video, status, shown = plugin.auto_timing_tick(True, 3)
        self.assertNotIn("value", video)
        self.assertEqual(shown, 3)


if __name__ == "__main__":
    unittest.main()
