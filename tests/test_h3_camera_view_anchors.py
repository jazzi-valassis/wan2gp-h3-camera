"""Exact-pose view anchors: targets and pacing checkpoints, re-aiming, helper views and the queue-backed Apply."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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


END = 10.083333333333334
REPORTED = [pose(0, 0), pose(.5, 45, 48, .75), pose(5.541666666666666 / END, 45, 48, .75), pose(1, 90, 0, .5)]


def targets_of(path, frames=243, interpolation="smooth"):
    return views.anchor_targets(path, frames, plan.keyframe_frame, 24, interpolation)


class ViewTargetTests(unittest.TestCase):
    def test_reported_path_gets_exact_views_a_shared_hold_and_pacing_checkpoints(self):
        targets, skipped = targets_of(REPORTED)
        self.assertEqual([(t["pose"], t["frames"], t["kind"], t["keyframes"]) for t in targets], [
            ((22.5, 24.0, .875), [60], "checkpoint", []),
            ((45.0, 48.0, .75), [121, 133], "keyframe", [2, 3]),
            ((67.5, 24.0, .625), [188], "checkpoint", []),
            ((90.0, 0.0, .5), [242], "keyframe", [4]),
        ])
        self.assertEqual(skipped, [])

    def test_keyframes_between_the_old_45_degree_bins_are_anchored_at_their_pose(self):
        path = [pose(0, 0), pose(.2, 20, 10, .9), pose(.4, 35.7, 38, .8), pose(.6, -112, 80, 1.3), pose(1, 84.6, -50, 2)]
        targets, _ = targets_of(path)
        keyframes = [t["pose"] for t in targets if t["kind"] == "keyframe"]
        self.assertEqual(keyframes, [(20.0, 10.0, .9), (35.7, 38.0, .8), (-112.0, 80.0, 1.3), (84.6, -50.0, 2.0)])

    def test_checkpoints_follow_the_planned_easing_and_spacing(self):
        path = [pose(0, 0), pose(1, 180)]
        linear, _ = targets_of(path, interpolation="linear")
        self.assertEqual([(t["frames"], t["pose"][0]) for t in linear],
                         [([60], 45.0), ([121], 90.0), ([182], 135.0), ([242], 180.0)])
        smooth, _ = targets_of(path)
        self.assertEqual([round(t["pose"][0], 3) for t in smooth], [28.125, 90.0, 151.875, 180.0])

    def test_short_or_small_moves_need_no_checkpoint(self):
        short, _ = targets_of([pose(0, 0), pose(.25, 90), pose(1, 90)])
        self.assertEqual([(t["kind"], t["frames"]) for t in short], [("keyframe", [61, 242])])
        small, _ = targets_of([pose(0, 0), pose(1, 8)])
        self.assertEqual([t["kind"] for t in small], ["keyframe"])

    def test_a_return_to_the_start_view_is_left_to_the_start_image(self):
        targets, skipped = targets_of([pose(0, 0), pose(.4, 30), pose(.5, 30), pose(.6, 0), pose(1, 0)])
        self.assertEqual([(t["pose"], t["kind"]) for t in targets], [((15.0, 0.0, 1.0), "checkpoint"), ((30.0, 0.0, 1.0), "keyframe")])
        self.assertEqual(skipped, [4, 5])

    def test_helper_and_aimed_poses(self):
        self.assertIsNone(views.helper_pose((0, 0, 1), (45, 0, 1)))
        azimuth, elevation, distance = views.helper_pose((0, 0, 1), (90, 0, .5))
        self.assertAlmostEqual(views.pose_gap((0, 0, 1), (azimuth, elevation, distance)), views.MAX_STEP)
        self.assertLess(views.pose_gap((azimuth, elevation, distance), (90, 0, .5)),
                        views.pose_gap((0, 0, 1), (90, 0, .5)))
        self.assertIsNone(views.helper_pose((170, 0, 1), (190, 0, 1)))
        # Azimuths are unwrapped: from 170 to -170 the path turns back 340 degrees, so the helper turns left.
        self.assertEqual(round(views.helper_pose((170, 0, 1), (-170, 0, 1))[0]), 120)
        aimed = views.aimed_pose((67.5, 24, .625), (57.5, 22, .68))
        self.assertEqual([round(x, 4) for x in aimed], [77.5, 26.0, round(.625 * .625 / .68, 4)])

    def test_a_view_is_judged_on_its_angle_and_its_distance_separately(self):
        # Test 1: a clean view measured 20.3r 20.1u 0.711x for a 22.5r 24u 0.875x plan, right in angle but framed
        # 19% closer; one combined score rejected it.
        self.assertTrue(views.accepted((20.3, 20.1, .711), (22.5, 24, .875)))
        self.assertTrue(views.accepted((42, 50.3, .884), (45, 48, .75)))
        self.assertFalse(views.accepted((100.4, 10.5, .501), (45, 0, .5)))
        self.assertFalse(views.accepted((-11, -6.9, 1.33), (45, 0, .5)))
        self.assertFalse(views.accepted((45, 0, .7), (45, 0, .5)))  # 40% farther
        self.assertTrue(views.accepted((-178, 0, 1), (179, 0, 1)))  # orbit angles wrap

    def test_labels_names_and_task(self):
        self.assertEqual(views.pose_label((45, 48, .75)), "45r 48u 0.75x")
        self.assertEqual(views.pose_label((-22.5, -10, 1.2)), "22.5l 10d 1.2x")
        name = views.view_name((45, 48, .75), 42)
        self.assertEqual(name, "h3cam_view_45r_48u_0p75x_s42")
        self.assertEqual(views.view_name((45, 48, .75), 42, 1), name + "_1")
        self.assertTrue(views.is_generated_view(f"D:/outputs/{name}.jpg"))
        self.assertFalse(views.is_generated_view("D:/outputs/start.png"))
        self.assertEqual(views.view_resolution(704, 1280), "704x1280")
        self.assertEqual(views.view_resolution(1920, 1080), "1264x704")
        plate = views.plate_task("start.png", "mask.png", "704x1280", 7, "h3cam_view_plate_s7")
        self.assertEqual((plate["image_guide"], plate["image_mask"], plate["model_mode"]), ("start.png", "mask.png", 0))
        self.assertEqual(plate["activated_loras"], [views.TURBO_LORA])
        task = views.view_task("start.png", "coarse.png", "704x1280", 7, name)
        self.assertEqual(task["model_type"], "qwen_image_21_7B")
        self.assertEqual(task["prompt"], "Change the camera angle from <image2> to <image1>.")
        self.assertEqual(task["image_refs"], ["start.png", "coarse.png"])
        self.assertEqual((task["num_inference_steps"], task["guidance_scale"], task["sample_solver"], task["seed"]),
                         (8, 1, "pruna", 7))
        self.assertEqual(task["activated_loras"], [views.ANYANGLE_LORA, views.TURBO_LORA])


class FakeModel:
    def close(self):
        FakeScene.closed += 1


class FakeScene:
    """Stands in for Depth Anything 3: measures each view as its target plus a scripted miss."""
    closed = 0
    misses = []

    def __init__(self, model, start, mask=None, plate=None, plate_mask=None):
        self.measured, self.aims, self.images, self.plates, self.sources = [(0.0, 0.0, 1.0)], [], [start], {}, []
        self.mask, self.plate, self.plate_mask = mask, plate, plate_mask
        self.subjects = {0: mask}

    def source_for(self, target, size=None):
        if views.pose_gap(self.measured[0], target) <= 55:
            return 0
        return min(range(len(self.measured)), key=lambda index: views.pose_gap(self.measured[index], target))

    def set_plate(self, index, plate, plate_mask):
        if plate is not None:
            self.plates[index] = plate

    def set_subject(self, index, mask):
        self.subjects[index] = mask

    def coarse(self, aim, size):
        self.aims.append(tuple(aim))
        self.sources.append(self.source_for(aim))
        return Image.new("RGB", size, "gray"), self.sources[-1], .5

    def add(self, image, target):
        miss = FakeScene.misses.pop(0) if FakeScene.misses else (0, 0, 1)
        self.measured.append((target[0] + miss[0], target[1] + miss[1], target[2] * miss[2]))
        self.images.append(image)
        return self.measured[-1]

    def drop_last(self):
        self.measured.pop()
        self.images.pop()


class FakeResult:
    def __init__(self, files, success=True, errors=()):
        self.generated_files, self.success, self.errors = list(files), success, list(errors)

    @property
    def cancelled(self):
        return False


class FakeJob:
    def __init__(self, result, done=True):
        self._result, self.done = result, done

    def result(self, timeout=None):
        return self._result


class FakeSession:
    def __init__(self, folder, success=True, done=True):
        self.folder, self.success, self.done, self.submissions, self.jobs = folder, success, done, [], []

    def submit(self, tasks):
        self.submissions.append(tasks)
        files = []
        for task in tasks:
            target = Path(self.folder) / f"{task['output_filename']}.png"
            Image.new("RGB", (64, 128), "blue").save(target)
            files.append(str(target))
        job = FakeJob(FakeResult(files if self.success else [], self.success), self.done)
        self.jobs.append(job)
        return job


NAMES = (*camera.FORM_OUTPUTS, "table", "compiled", "summary", *camera.ANCHOR_OUTPUTS, "views", "applied")


class ApplyWithViewAnchorsTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.plugin = make_plugin()
        self.start = str(Path(self.folder.name) / "start.png")
        Image.new("RGB", (44, 80), "red").save(self.start)
        self.ref = Image.new("RGB", (64, 128), "green")
        FakeScene.misses, FakeScene.closed = [], 0
        self.scenes = []

        def scene(model, start, mask=None, plate=None, plate_mask=None):
            self.scenes.append(FakeScene(model, start, mask, plate, plate_mask))
            return self.scenes[-1]

        self.patches = [patch.object(camera.view_geometry, "Scene", side_effect=scene),
                        patch.object(camera.view_geometry, "DepthModel", FakeModel),
                        patch.object(camera.view_geometry, "subject_mask", side_effect=lambda image: self.mask)]
        self.mask = None
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        for patcher in self.patches:
            patcher.stop()
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

    def view(self, label, attempt=0):
        return str(Path(self.folder.name) / (f"h3cam_view_{label}_s42" + (f"_{attempt}" if attempt else "") + ".png"))

    def test_apply_renders_each_view_at_its_pose_and_injects_them(self):
        self.plugin._wangp_session = session = FakeSession(self.folder.name)
        FakeScene.misses = [(-1.5, 2, 1), (3, -4, 1), (0, 0, 1.04), (-1, 1, 1)]
        steps = self.apply(self.args())
        self.assertEqual(len(session.submissions), 4)
        tasks = [task for submission in session.submissions for (task,) in [submission]]
        self.assertTrue(all(task["prompt"] == views.VIEW_PROMPT and task["image_refs"][0] == self.start for task in tasks))
        self.assertTrue(all(Path(task["image_refs"][1]).name.startswith("coarse_") for task in tasks))
        self.assertEqual(self.scenes[0].aims, [(22.5, 24.0, .875), (45.0, 48.0, .75), (67.5, 24.0, .625), (90.0, 0.0, .5)])
        self.assertEqual(FakeScene.closed, 1)
        first, final = steps[0], steps[-1]
        self.assertIn("Preparing 4 exact-pose view(s)", first["summary"]["value"])
        self.assertNotIn("value", first["applied"])
        self.assertTrue(final["applied"]["value"])
        self.assertEqual(final["frames_positions"]["value"], "61 122 134 189 243")
        hold, end = self.view("45r_48u_0p75x"), self.view("90r_0u_0p5x")
        self.assertEqual(final["image_refs"]["value"],
                         [self.view("22p5r_24u_0p875x"), hold, hold, self.view("67p5r_24u_0p625x"), end, self.ref])
        self.assertEqual(final["video_prompt_type"]["value"], "FI")
        prompt = final["prompt"]["value"]
        self.assertIn("A woman from <Picture 7> sits on a crate.", prompt)
        self.assertIn("Keep the camera composition from <Picture 4> through <Picture 5>", prompt)
        self.assertIn("At 2.500000s, pass through the camera composition shown in <Picture 3> without stopping", prompt)
        self.assertIn("At 7.833333s, pass through the camera composition shown in <Picture 6> without stopping", prompt)
        self.assertIn("Reach the camera composition shown in <Picture 2> at the end of this segment.", prompt)
        summary = final["summary"]["value"]
        self.assertIn("frames 61 122 134 189 243 (1-based)", summary)
        self.assertIn("pacing checkpoint at 2.50s: 22.5r 24u 0.875x (measured 21r 26u 0.875x)", summary)
        self.assertIn("keyframes 2, 3 at 5.04s and 5.54s: 45r 48u 0.75x (measured 48r 44u 0.75x)", summary)
        self.assertIn("keyframe 4 at 10.08s: 90r 0u 0.5x (measured 89r 1u 0.5x)", summary)
        self.assertEqual(final["views"]["value"][1][1], "5.04s, 5.54s: 45r 48u 0.75x (measured 48r 44u 0.75x)")

    def test_the_subject_is_painted_out_once_for_a_background_plate(self):
        self.plugin._wangp_session = session = FakeSession(self.folder.name)
        import numpy as np
        self.mask = np.zeros((1280, 704), bool)
        self.mask[400:900, 250:450] = True
        args = self.args()
        final = self.apply(args)[-1]
        plate, = session.submissions[0]
        self.assertEqual((plate["image_mode"], plate["video_prompt_type"], plate["image_guide"]), (2, "VAG", self.start))
        self.assertEqual(plate["prompt"], views.PLATE_PROMPT)
        self.assertEqual(len(session.submissions), 8)
        scene = self.scenes[0]
        self.assertEqual(scene.plate.size, (704, 1280))
        self.assertTrue(scene.plate_mask[392, 245] and not scene.plate_mask[100, 100])  # grown around the subject
        self.assertNotIn("plate", " ".join(map(str, final["image_refs"]["value"])))
        self.apply(self.follow(args, final))
        self.assertEqual(len(session.submissions), 8)  # plates and views come from the cache

    def test_a_generated_view_gets_its_own_plate_before_it_is_re_rendered(self):
        self.plugin._wangp_session = session = FakeSession(self.folder.name)
        import numpy as np
        self.mask = np.zeros((1280, 704), bool)
        self.mask[400:900, 250:450] = True
        self.apply(self.args())
        kinds = [tasks[0]["image_mode"] for tasks in session.submissions]
        # The fake re-renders from the Start Image within 55 degrees and otherwise from the nearest view,
        # which is painted out first: start plate, 22.5, plate of 22.5, 45, plate of 45, 67.5, plate of 67.5, 90.
        self.assertEqual(kinds, [2, 1, 2, 1, 2, 1, 2, 1])
        self.assertEqual(session.submissions[2][0]["image_guide"], self.view("22p5r_24u_0p875x"))
        self.assertEqual(self.scenes[0].sources, [0, 1, 2, 3])
        self.assertIsNotNone(self.scenes[0].plate)  # the Start Image's own plate
        self.assertEqual(sorted(self.scenes[0].plates), [1, 2, 3])
        # Each re-rendered view's subject is recorded too, so it is flattened like the Start Image's.
        self.assertEqual(sorted(self.scenes[0].subjects), [0, 1, 2, 3])

    def test_a_missed_view_is_reaimed_once(self):
        self.plugin._wangp_session = session = FakeSession(self.folder.name)
        path = [pose(0, 0), pose(.2, 40), pose(1, 40)]
        FakeScene.misses = [(-20, 0, 1), (2, 0, 1)]
        final = self.apply(self.args(path))[-1]
        self.assertEqual(self.scenes[0].aims, [(40.0, 0.0, 1.0), (60.0, 0.0, 1.0)])
        self.assertEqual([tasks[0]["seed"] for tasks in session.submissions], [42, 43])
        self.assertEqual(final["image_refs"]["value"][:2], [self.view("40r_0u_1x", 1)] * 2)
        self.assertEqual(self.scenes[0].measured, [(0.0, 0.0, 1.0), (42.0, 0.0, 1.0)])

    def test_a_view_that_misses_twice_is_left_out_and_text_guides_that_pose(self):
        # Test 1 (2026-10-16): the end view of a 45-degree orbit measured 100 degrees on both attempts, and keeping
        # the closer one would have steered H3 to a face-on ending. The other views still anchor their poses.
        self.plugin._wangp_session = session = FakeSession(self.folder.name)
        FakeScene.misses = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (55, 10, 1), (40, 8, 1)]
        args = self.args()
        final = self.apply(args)[-1]
        self.assertEqual(len(session.submissions), 5)
        self.assertEqual(final["frames_positions"]["value"], "61 122 134 189")
        hold = self.view("45r_48u_0p75x")
        self.assertEqual(final["image_refs"]["value"],
                         [self.view("22p5r_24u_0p875x"), hold, hold, self.view("67p5r_24u_0p625x"), self.ref])
        self.assertNotIn(self.view("90r_0u_0p5x"), final["image_refs"]["value"])
        self.assertEqual(len(self.scenes[0].measured), 4)  # the missed view is not registered for later re-renders
        summary = final["summary"]["value"]
        self.assertIn("frames 61 122 134 189 (1-based)", summary)
        self.assertIn("Left out because neither attempt came within 15 degrees and 30% distance of the plan, so text alone guides these "
                      "poses: keyframe 4 at 10.08s: 90r 0u 0.5x (closest attempt measured 130r 8u 0.5x).", summary)
        self.assertIn("It ends at eye level, on a side view of the subject.", final["prompt"]["value"])
        self.assertEqual([view for view, _ in final["views"]["value"]],
                         [self.view("22p5r_24u_0p875x"), hold, self.view("67p5r_24u_0p625x")])
        # Re-applying the same path reuses the kept views and does not render the missed one again.
        steps = self.apply(self.follow(args, final))
        self.assertEqual(len(session.submissions), 5)
        self.assertEqual(steps[-1]["frames_positions"]["value"], "61 122 134 189")
        self.assertIn("Left out because neither attempt", steps[-1]["summary"]["value"])

    def test_when_every_view_misses_the_plan_applies_as_text_and_drops_stale_views(self):
        self.plugin._wangp_session = session = FakeSession(self.folder.name)
        path = [pose(0, 0), pose(.2, 40), pose(1, 40)]
        final = self.apply(self.args(path))[-1]
        self.assertEqual(final["image_refs"]["value"][:2], [self.view("40r_0u_1x")] * 2)
        FakeScene.misses = [(-30, 0, 1), (25, 0, 1)]
        final = self.apply(self.follow(self.args([pose(0, 0), pose(.2, 50), pose(1, 50)]), final))[-1]
        self.assertEqual(len(session.submissions), 3)
        self.assertEqual(final["image_refs"]["value"], [self.ref])
        self.assertEqual(final["frames_positions"]["value"], "")
        self.assertNotIn("F", final["video_prompt_type"]["value"])
        summary = final["summary"]["value"]
        self.assertIn("No generated view landed close enough to its pose to anchor (seed 42).", summary)
        self.assertIn("keyframes 2, 3 at 2.00s and 10.08s: 50r 0u 1x (closest attempt measured 75r 0u 1x)", summary)
        self.assertNotIn("<Picture", final["prompt"]["value"].split("Camera plan:", 1)[1])
        self.assertEqual(final["views"]["value"], [])

    def test_far_targets_are_reached_through_helper_views(self):
        self.plugin._wangp_session = session = FakeSession(self.folder.name)
        final = self.apply(self.args([pose(0, 0), pose(.2, 180), pose(1, 180)]))[-1]
        self.assertEqual([round(aim[0]) for aim in self.scenes[0].aims], [50, 100, 150, 180])
        self.assertEqual(len(session.submissions), 4)
        self.assertEqual(final["frames_positions"]["value"], "49 243")
        self.assertEqual(final["image_refs"]["value"][:2], [self.view("180r_0u_1x")] * 2)

    def test_reapply_reuses_cached_views_and_replaces_stale_anchors(self):
        self.plugin._wangp_session = session = FakeSession(self.folder.name)
        args = self.args()
        final = self.apply(args)[-1]
        steps = self.apply(self.follow(args, final))
        self.assertEqual(len(session.submissions), 4)
        final = steps[-1]
        self.assertEqual(final["frames_positions"]["value"], "61 122 134 189 243")
        self.assertEqual(len(final["image_refs"]["value"]), 6)
        self.assertIn("A woman from <Picture 7> sits on a crate.", final["prompt"]["value"])
        self.assertEqual(final["prompt"]["value"].count("Camera plan:"), 1)

    def test_text_guidance_with_a_reason_when_views_are_off_or_unavailable(self):
        self.plugin._wangp_session = session = FakeSession(self.folder.name)
        cases = ((False, self.args(), "view anchors are turned off."),
                 (True, self.args(image_start=None), "Start Image"),
                 (True, self.args(path=[pose(0, 0), pose(1, 0)]), "never leaves the start view"))
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
            list(steps)
        self.assertEqual(FakeScene.closed, 1)


class FakeService:
    def __init__(self, session=None):
        self.started, self.session = 0, session

    def start_generation(self):
        self.started += 1
        for job in (self.session.jobs if self.session else []):
            job.done = True


class QueueWithoutBrowserTests(unittest.TestCase):
    def test_a_waiting_inline_queue_is_loaded_and_started_on_the_server(self):
        plugin = make_plugin()
        loaded = []
        plugin.load_queue_action = lambda filepath, state, event: loaded.append(state["gen"].pop("inline_queue"))
        service = FakeService()
        state = {"gen": {"inline_queue": [{"id": 0}]}}
        with patch.object(camera, "service_for", return_value=service):
            self.assertTrue(plugin._start_queue_without_browser(state))
            self.assertEqual((loaded, service.started), ([[{"id": 0}]], 1))
            # The browser already relayed it, or there is no hybrid service: nothing to do.
            self.assertFalse(plugin._start_queue_without_browser(state))
        with patch.object(camera, "service_for", return_value=None):
            self.assertFalse(plugin._start_queue_without_browser({"gen": {"inline_queue": [{}]}}))

    def test_view_renders_finish_without_a_browser_relay(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        plugin = make_plugin()
        session = FakeSession(folder.name, done=False)
        service = FakeService(session)
        plugin.load_queue_action = lambda filepath, state, event: state["gen"].pop("inline_queue", None)
        state = {"model_type": "h3", "gen": {}}
        original = session.submit

        def submit(tasks):
            state["gen"]["inline_queue"] = [{"id": 0}]
            return original(tasks)

        session.submit = submit
        start = str(Path(folder.name) / "start.png")
        Image.new("RGB", (44, 80), "red").save(start)
        run = camera._ViewRun(plugin, session, state, start, "704x1280", 42)
        run.BROWSER_GRACE = 0
        task = views.view_task(start, start, "704x1280", 42, "h3cam_view_test")
        with patch.object(camera, "service_for", return_value=service):
            file = list_result(run._render(task, "view 1 of 1"))
        run.close()
        self.assertEqual(service.started, 1)
        self.assertTrue(Path(file).is_file())


def list_result(steps):
    while True:
        try:
            next(steps)
        except StopIteration as stop:
            return stop.value


if __name__ == "__main__":
    unittest.main()
