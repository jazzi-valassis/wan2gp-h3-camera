"""Generated view anchors: pose-to-view prompts, keyframe targets and the queue-backed apply."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import gradio as gr
from PIL import Image

from test_h3_camera_plugin import make_plugin, arguments, camera


SPEC = importlib.util.spec_from_file_location("h3_camera_view_anchors", Path(__file__).resolve().parents[1] / "view_anchors.py")
views = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(views)
PLAN = importlib.util.spec_from_file_location("h3_camera_view_plan", Path(__file__).resolve().parents[1] / "camera_plan.py")
plan = importlib.util.module_from_spec(PLAN)
PLAN.loader.exec_module(plan)


def pose(time, azimuth, elevation=0, distance=1):
    return dict(time=time, azimuth=azimuth, elevation=elevation, distance=distance)


REPORTED = [pose(0, 0), pose(.5, 45, 48, .75), pose(5.541666666666666 / 10.083333333333334, 45, 48, .75), pose(1, 90, 0, .5)]


class ViewPromptTests(unittest.TestCase):
    def test_reported_path_keyframes_map_to_lora_views(self):
        self.assertEqual(views.view_prompt(REPORTED[1]), "<sks> front-right quarter view high-angle shot medium shot")
        self.assertEqual(views.view_prompt(REPORTED[3]), "<sks> right side view eye-level shot close-up")

    def test_orbit_direction_wrapping_and_distance(self):
        self.assertIn("left side view", views.view_prompt(pose(1, -90)))
        self.assertIn("left side view", views.view_prompt(pose(1, 270)))
        self.assertIn("front view", views.view_prompt(pose(1, 360)))
        self.assertIn("back view", views.view_prompt(pose(1, 180)))
        self.assertIn("back view", views.view_prompt(pose(1, -175)))
        self.assertIn("back-left quarter view", views.view_prompt(pose(1, 225)))
        self.assertIn("low-angle shot", views.view_prompt(pose(1, 0, -25)))
        self.assertTrue(views.view_prompt(pose(1, 0, 0, .3)).endswith("close-up"))
        self.assertTrue(views.view_prompt(pose(1, 0, 0, 2)).endswith("wide shot"))

    def test_views_between_lora_bins_are_not_anchored(self):
        for unsupported in (pose(1, 20), pose(1, 45, 80), pose(1, 90, -50), pose(1, 112), pose(1, 84.6), pose(1, 35.7, 38)):
            with self.subTest(pose=unsupported):
                self.assertIsNone(views.view_prompt(unsupported))

    def test_targets_share_one_view_across_a_hold_and_skip_in_between_keyframes(self):
        targets, skipped = views.anchor_targets(REPORTED, 243, plan.keyframe_frame)
        self.assertEqual(sorted(targets), [121, 133, 242])
        self.assertEqual(targets[121], targets[133])
        self.assertEqual(skipped, [])
        dense = [pose(0, 0), pose(.25, 20, 24, .9), *REPORTED[1:]]
        targets, skipped = views.anchor_targets(dense, 243, plan.keyframe_frame)
        self.assertEqual(sorted(targets), [121, 133, 242])
        self.assertEqual(skipped, [2])

    def test_one_keyframe_per_second_path_anchors_only_exact_views(self):
        end = 10.083333333333334
        dense = [pose(0, 0)] + [pose(t / end, a, e, d) for t, a, e, d in (
            (1, 8.93, 9.52, .95), (2, 17.85, 19.04, .901), (3, 26.78, 28.56, .851), (4, 35.7, 38.08, .802),
            (5.041666666666667, 45, 48, .75), (5.541666666666667, 45, 48, .75), (6.541666666666667, 54.91, 37.43, .695),
            (7.541666666666667, 64.82, 26.86, .64), (8.541666666666667, 74.72, 16.29, .585), (9.541666666666667, 84.63, 5.72, .53),
            (end, 90, 0, .5))]
        targets, skipped = views.anchor_targets(dense, 243, plan.keyframe_frame)
        self.assertEqual(sorted(targets), [121, 133, 242])
        self.assertEqual(skipped, [2, 3, 4, 5, 8, 9, 10, 11])

    def test_start_view_is_not_reanchored_and_shared_views_keep_the_closest_pose(self):
        targets, skipped = views.anchor_targets([pose(0, 0), pose(.5, 3), pose(1, 0, 0, .3)], 243, plan.keyframe_frame)
        self.assertEqual(targets, {242: "<sks> front view eye-level shot close-up"})
        self.assertEqual(skipped, [2])
        targets, skipped = views.anchor_targets([pose(0, 0), pose(.5, 88), pose(.7, 88), pose(1, 92.5)], 243, plan.keyframe_frame)
        self.assertEqual(sorted(targets), [121, 169])
        self.assertEqual(skipped, [4])

    def test_resolution_and_tasks(self):
        self.assertEqual(views.view_resolution(704, 1280), "704x1280")
        self.assertEqual(views.view_resolution(1920, 1080), "1264x704")
        task, = views.view_tasks(["<sks> front view eye-level shot medium shot"], "start.png", "704x1280", 7)
        self.assertEqual(task["model_type"], "qwen_image_edit_plus2_20B")
        self.assertEqual((task["num_inference_steps"], task["guidance_scale"], task["seed"]), (8, 1, 7))
        self.assertEqual(task["video_prompt_type"], "KI")
        self.assertEqual(task["image_refs"], ["start.png"])
        self.assertEqual(task["activated_loras"], [views.ANGLES_LORA, views.LIGHTNING_LORA])
        self.assertEqual(task["loras_multipliers"], "0.9 1")


class FakeResult:
    def __init__(self, files, success=True, errors=()):
        self.generated_files, self.success, self.errors = list(files), success, list(errors)

    @property
    def cancelled(self):
        return False


class FakeJob:
    done = True

    def __init__(self, result):
        self._result = result

    def result(self, timeout=None):
        return self._result


class FakeSession:
    def __init__(self, folder, success=True):
        self.folder, self.success, self.submissions = folder, success, []

    def submit(self, tasks):
        self.submissions.append(tasks)
        files = []
        for task in tasks:
            target = Path(self.folder) / f"{task['output_filename']}.png"
            Image.new("RGB", (64, 128), "blue").save(target)
            files.append(str(target))
        return FakeJob(FakeResult(files if self.success else [], self.success))


NAMES = (*camera.FORM_OUTPUTS, "table", "compiled", "summary", *camera.ANCHOR_OUTPUTS, "views", "applied")
HOLD = "h3cam_view_front-right_quarter_view_high-angle_shot_medium_shot_s42.png"
END = "h3cam_view_right_side_view_eye-level_shot_close-up_s42.png"


class ApplyWithViewAnchorsTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.plugin = make_plugin()
        self.start = str(Path(self.folder.name) / "start.png")
        Image.new("RGB", (44, 80), "red").save(self.start)
        self.ref = Image.new("RGB", (64, 128), "green")

    def tearDown(self):
        self.folder.cleanup()

    def args(self, path=REPORTED, **changes):
        options = dict(prompt="A woman from <Picture 2> sits on a crate.\noverall_soundscape: Rain.", video_length=239,
                       image_prompt_type="S", image_start=[self.start], video_prompt_type="I", image_refs=[self.ref])
        options.update(changes)
        args = arguments(**options)
        args[0] = json.dumps(path)
        return [*args, True, ""]

    def apply(self, args, use_views=True):
        steps = list(self.plugin.apply_camera(use_views, 42, *args))
        # Streamed outputs must all be update dicts; mixing raw values breaks Gradio's diffs in the browser.
        for step in steps:
            self.assertEqual(len(step), len(NAMES))
            self.assertTrue(all(isinstance(item, dict) and item.get("__type__") == "update" for item in step))
        return [dict(zip(NAMES, step)) for step in steps]

    @staticmethod
    def follow(args, final):
        args = list(args)
        for name in camera.FORM_INPUTS:
            if "value" in final.get(name, {}):
                args[4 + camera.FORM_INPUTS.index(name)] = final[name]["value"]
        args[-1] = final["frames_positions"]["value"]
        return args

    def path(self, name):
        return str(Path(self.folder.name) / name)

    def test_apply_generates_one_view_per_pose_and_injects_them(self):
        self.plugin._wangp_session = session = FakeSession(self.folder.name)
        steps = self.apply(self.args())
        tasks, = session.submissions
        self.assertEqual([task["prompt"] for task in tasks],
                         ["<sks> front-right quarter view high-angle shot medium shot", "<sks> right side view eye-level shot close-up"])
        self.assertEqual((tasks[0]["resolution"], tasks[0]["image_refs"]), ("704x1280", [self.start]))
        first, final = steps[0], steps[-1]
        self.assertIn("Generating 2 view image(s)", first["summary"]["value"])
        self.assertNotIn("value", first["applied"])
        self.assertTrue(final["applied"]["value"])
        self.assertEqual(final["frames_positions"]["value"], "122 134 243")
        self.assertEqual(final["image_refs"]["value"], [self.path(HOLD), self.path(HOLD), self.path(END), self.ref])
        self.assertEqual(final["video_prompt_type"]["value"], "FI")
        prompt = final["prompt"]["value"]
        self.assertIn("A woman from <Picture 5> sits on a crate.", prompt)
        self.assertIn("Keep the camera composition from <Picture 3> through <Picture 4>", prompt)
        self.assertIn("Reach the camera composition shown in <Picture 2> at the end of this segment.", prompt)
        self.assertIn("In total the camera arcs a quarter turn (90 degrees)", prompt)
        summary = final["summary"]["value"]
        self.assertIn("frames 122 134 243 (1-based)", summary)
        self.assertIn("keyframes 2, 3: front-right quarter view high-angle shot medium shot", summary)
        self.assertIn("keyframe 4: right side view eye-level shot close-up", summary)
        self.assertEqual(final["table"]["value"][3], [4, 10.083333333333334, 90.0, 0.0, 0.5])
        self.assertEqual([caption for _, caption in final["views"]["value"]],
                         ["front-right quarter view high-angle shot medium shot", "right side view eye-level shot close-up"])

    def test_reapply_reuses_cached_views_and_replaces_stale_anchors(self):
        self.plugin._wangp_session = session = FakeSession(self.folder.name)
        args = self.args()
        final = self.apply(args)[-1]
        moved = [REPORTED[0], dict(REPORTED[1], time=.4), dict(REPORTED[2], time=.45), REPORTED[3]]
        again = self.follow(args, final)
        again[0] = json.dumps(moved)
        steps = self.apply(again)
        self.assertEqual(len(session.submissions), 1)
        self.assertEqual(len(steps), 1)
        final = steps[0]
        self.assertEqual(final["frames_positions"]["value"], "98 110 243")
        self.assertEqual(final["image_refs"]["value"], [self.path(HOLD), self.path(HOLD), self.path(END), self.ref])
        self.assertIn("A woman from <Picture 5> sits on a crate.", final["prompt"]["value"])
        self.assertEqual(final["prompt"]["value"].count("Camera plan:"), 1)

    def test_text_guidance_with_a_reason_when_views_are_off_or_unavailable(self):
        self.plugin._wangp_session = session = FakeSession(self.folder.name)
        cases = ((False, self.args(), "view anchors are turned off."),
                 (True, self.args(image_start=None), "Start Image"),
                 (True, self.args(path=[pose(0, 0), pose(1, 20, 10)]), "no keyframe matches"))
        for use_views, args, reason in cases:
            with self.subTest(reason=reason):
                step, = self.apply(args, use_views)
                self.assertIn("Text guidance only: ", step["summary"]["value"])
                self.assertIn(reason, step["summary"]["value"])
                self.assertNotIn("value", step["image_refs"])
                self.assertIn("Camera plan:", step["prompt"]["value"])
                self.assertTrue(step["applied"]["value"])
        self.assertEqual(session.submissions, [])

    def test_failed_generation_leaves_the_form_unchanged(self):
        self.plugin._wangp_session = FakeSession(self.folder.name, success=False)
        steps = self.plugin.apply_camera(True, 42, *self.args())
        self.assertNotIn("value", next(steps)[0])
        with self.assertRaisesRegex(gr.Error, "form is unchanged"):
            next(steps)


if __name__ == "__main__":
    unittest.main()
