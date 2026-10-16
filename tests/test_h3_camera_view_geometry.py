"""Exact-pose geometry: orbit conventions, re-rendering, gap filling and view registration."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


SPEC = importlib.util.spec_from_file_location("h3_camera_view_geometry", Path(__file__).resolve().parents[1] / "view_geometry.py")
geometry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(geometry)

K = (100.0, 100.0, 32.0, 32.0)  # 64x64 image, 35.5-degree field of view


def pose_matrix(rotation, centre):
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, centre
    return matrix


def stripes():
    """A 64x64 picture with vertical stripes whose colour encodes the column."""
    image = np.zeros((64, 64, 3), np.uint8)
    image[..., 0] = np.arange(64)[None, :] * 4
    image[..., 1] = 128
    return Image.fromarray(image)


class OrbitTests(unittest.TestCase):
    def test_orbit_and_measurement_agree(self):
        pivot = np.array([0.0, 0.0, 3.0])
        for wanted in ((0, 0, 1), (30, 20, .8), (-75, -15, 1.4), (135, 60, .5), (90, 0, .5)):
            with self.subTest(pose=wanted):
                rotation, centre = geometry.orbit(pivot, *wanted)
                measured = geometry.pose_of(pose_matrix(rotation, centre), pivot)
                np.testing.assert_allclose(measured, wanted, atol=1e-9)

    def test_measured_elevation_is_relative_to_a_start_view_looking_down_at_its_subject(self):
        pivot = np.array([.4, .9, 3.0])  # subject right of and below the frame centre
        for wanted in ((0, 0, 1), (30, 20, .8), (60, 45, 1.2)):
            with self.subTest(pose=wanted):
                measured = geometry.pose_of(pose_matrix(*geometry.orbit(pivot, *wanted)), pivot)
                np.testing.assert_allclose(measured, wanted, atol=1e-9)
        np.testing.assert_allclose(geometry.pose_of(np.eye(4), pivot), (0, 0, 1), atol=1e-9)

    def test_positive_azimuth_moves_the_camera_right_and_elevation_raises_it(self):
        pivot = np.array([0.0, 0.0, 3.0])
        _, centre = geometry.orbit(pivot, 30, 0, 1)
        self.assertGreater(centre[0], 0)  # x points right
        _, centre = geometry.orbit(pivot, 0, 30, 1)
        self.assertLess(centre[1], 0)  # y points down
        rotation, _ = geometry.orbit(pivot, 0, 30, 1)
        self.assertGreater(rotation[1, 2], 0)  # the camera looks down toward the pivot

    def test_far_scenery_is_pushed_back_while_the_subject_keeps_its_depth(self):
        depth = np.full((10, 10), 4.0)
        depth[3:7, 3:7] = 1.0
        expanded = geometry.expand_far(depth, near=1.0, far=50)
        self.assertAlmostEqual(float(expanded[5, 5]), 1.0, places=6)
        self.assertGreater(float(expanded[0, 0]), 40)


class RenderTests(unittest.TestCase):
    def test_a_plane_re_renders_unchanged_from_its_own_camera_and_scenery_behind_the_pivot_slides_right(self):
        image = stripes()
        cloud = geometry.point_cloud(image, np.full((64, 64), 6.0), K)
        pivot = np.array([0.0, 0.0, 3.0])
        same, covered = geometry.render(cloud, K, *geometry.orbit(pivot, 0, 0, 1), (64, 64))
        self.assertGreater(covered.mean(), .99)
        np.testing.assert_array_equal(same[:, :, 0], np.asarray(image)[:, :, 0])
        # Orbiting right turns the camera left to keep the pivot centred: the wall behind it shifts right,
        # here by 100 * 0.521 / 5.95 = 8.8 pixels.
        turned, covered = geometry.render(cloud, K, *geometry.orbit(pivot, 10, 0, 1), (64, 64), fill=False)
        self.assertLess(covered.mean(), .99)
        row = turned[32, :, 0].astype(int)
        hits = np.flatnonzero(np.abs(row - int(np.asarray(image)[32, 32, 0])) <= 2)
        self.assertTrue(len(hits))
        self.assertAlmostEqual(float(hits.mean()), 32 + 8.8, delta=1.5)

    def test_flat_depth_is_left_alone_by_the_far_correction(self):
        depth = np.full((8, 8), 3.0)
        np.testing.assert_array_equal(geometry.expand_far(depth, near=3.0), depth)

    def test_cracks_inside_the_source_outline_are_filled_and_larger_gaps_stay_grey(self):
        image = np.zeros((100, 100, 3), np.uint8)
        image[:, :] = geometry.BACKGROUND
        covered = np.zeros((100, 100), bool)
        covered[10:60, 10:60] = True
        image[covered] = (200, 50, 50)
        covered[30:32, 30:32] = False  # a splat crack
        covered[15:35, 40:55] = False  # background a subject used to hide
        image[~covered] = geometry.BACKGROUND
        footprint = np.zeros((100, 100), bool)
        footprint[10:60, 10:60] = True
        filled = geometry.fill_interior_holes(image, covered, footprint)
        self.assertGreater(int(filled[30, 30, 0]), 150)
        self.assertEqual(tuple(filled[25, 47]), geometry.BACKGROUND)
        self.assertEqual(tuple(filled[90, 90]), geometry.BACKGROUND)

    def test_push_pull_keeps_known_pixels_and_blends_gaps(self):
        image = np.zeros((32, 32, 3), np.uint8)
        image[:, :16] = 200
        known = np.ones((32, 32), bool)
        known[10:20, 14:18] = False
        filled = geometry.push_pull(image, known)
        np.testing.assert_array_equal(filled[known], image[known])
        self.assertTrue(0 < int(filled[15, 15, 0]) <= 200)

    def test_a_thin_near_post_survives_the_depth_edge_filter(self):
        depth = np.full((64, 64), 9.0)
        depth[:, 31:33] = 3.0  # a two-pixel post in front of a far wall
        points, colors, source, size, outline, spans = geometry.point_cloud(stripes(), depth, K)
        self.assertEqual(tuple(spans.shape), (len(points), 2, 3))
        self.assertTrue(bool(((source - 3.0).abs() < 1e-9).sum() >= 64))
        self.assertEqual(size, (64, 64))
        self.assertEqual(outline.shape[1], 3)

    def test_an_oblique_surface_stays_closed_when_seen_face_on(self):
        # Test 1 (2026-10-16): a rooftop seen nearly edge-on from the start broke into rows of dots when the camera
        # rose above it, because splats were sized by distance alone. Sized by where neighbouring pixels land,
        # the surface stays closed; with the spans zeroed (distance sizing only) it does not.
        import cv2
        rows = (np.arange(64)[:, None] + .5 - 32) / 100
        cloud = geometry.point_cloud(Image.new("RGB", (64, 64), "gray"), (5 / (1 - 2.5 * rows)) * np.ones((64, 64)), K)
        camera = geometry.orbit(np.array([0.0, 0.0, 5.0]), 0, -60, 1)

        def closed(cloud):
            covered = geometry.render(cloud, K, *camera, (128, 128), fill=False)[1]
            hull = cv2.convexHull(np.argwhere(covered)[:, ::-1].astype(np.int32))
            inside = np.zeros(covered.shape, np.uint8)
            cv2.fillPoly(inside, [hull], 1)
            return float(covered[inside.astype(bool)].mean())

        self.assertGreater(closed(cloud), .98)
        self.assertLess(closed((*cloud[:5], cloud[5] * 0)), .7)

    def test_a_small_subject_is_flattened_to_a_card_and_a_large_one_keeps_its_relief(self):
        depth = np.full((100, 100), 9.0)
        depth[40:60, 40:50] = np.linspace(2.0, 4.0, 10)  # 2% of the frame, depth 2-4
        small = np.zeros((100, 100), bool)
        small[40:60, 40:50] = True
        flat = geometry.flatten_subject(depth, small)
        self.assertTrue(np.allclose(flat[small], np.median(depth[small])))
        self.assertTrue(np.array_equal(flat[~small], depth[~small]))
        large = np.zeros((100, 100), bool)
        large[10:90, 20:80] = True  # 48% of the frame
        self.assertIs(geometry.flatten_subject(depth, large), depth)
        middle = np.zeros((100, 100), bool)
        middle[25:75, 35:65] = True  # 15% of the frame: half its relief
        relief = geometry.flatten_subject(depth, middle)
        median = np.median(depth[middle])
        self.assertTrue(np.allclose(relief[middle] - median, .5 * (depth[middle] - median)))
        self.assertIs(geometry.flatten_subject(depth, None), depth)

    def test_parts_the_mask_missed_join_the_card_but_the_scenery_behind_does_not(self):
        depth = np.full((200, 200), 9.0)  # the card reaches 3% of the frame (3 pixels) beyond the mask
        depth[40:60, 40:50] = np.linspace(2.0, 4.0, 10)
        depth[55:60, 50:53] = 3.2  # an outstretched leg next to the mask, at the subject's depth
        mask = np.zeros((200, 200), bool)
        mask[40:60, 40:50] = True
        flat = geometry.flatten_subject(depth, mask)
        median = np.median(depth[mask])
        self.assertTrue(np.allclose(flat[55:60, 50:53], median))
        self.assertTrue(np.allclose(flat[45:50, 50:53], 9.0))  # wall right beside her, far behind
        self.assertTrue(np.allclose(flat[0:20], 9.0))

    def test_the_pivot_sits_on_a_thin_subject_rather_than_the_wall_behind_it(self):
        depth = np.full((64, 64), 9.0)
        depth[16:56, 28:36] = 3.0
        mask = np.zeros((64, 64), bool)
        mask[16:56, 28:36] = True
        pivot = geometry.pivot_point(depth, K, mask)
        self.assertAlmostEqual(pivot[2], 3.0)
        self.assertAlmostEqual(pivot[1], (36.0 - 32) / 100 * 3.0, places=6)  # median row 35.5, pixel centre 36
        # Without a mask, the nearest substantial depth in the centre stands in for the subject.
        self.assertLess(geometry.pivot_point(depth, K)[2], 9.0)


class FakeDepth:
    """Reconstructs every image as a plane at depth 3 whose camera moved 0.1 to the right per extra view."""

    def __init__(self):
        self.calls = 0

    def reconstruct(self, images):
        self.calls += 1
        n = len(images)
        poses = np.repeat(np.eye(4)[None], n, axis=0)
        poses[:, 0, 3] = np.arange(n) * .1
        return np.full((n, 64, 64), 3.0), poses, np.repeat(np.array([K]), n, axis=0)


class SceneTests(unittest.TestCase):
    def test_views_register_measure_and_undo_without_solving_again(self):
        model = FakeDepth()
        scene = geometry.Scene(model, stripes())
        self.assertEqual(scene.measured, [(0.0, 0.0, 1.0)])
        measured = scene.add(stripes(), (2, 0, 1))
        self.assertGreater(measured[0], 0)
        self.assertEqual(model.calls, 2)
        scene.drop_last()
        self.assertEqual((len(scene.images), len(scene.measured), model.calls), (1, 1, 2))
        image, source, coverage = scene.coarse((5, 0, 1), (64, 64))
        self.assertEqual((image.size, source), ((64, 64), 0))
        self.assertGreater(coverage, .5)


    def test_the_start_image_is_the_source_while_it_sees_enough_of_the_view(self):
        # Test 1: the end view re-rendered from the generated hold view (which saw two thirds of it) broke up,
        # while the Start Image, seeing about a quarter, gave a clean view on target.
        scene = geometry.Scene(FakeDepth(), stripes())
        scene.add(stripes(), (2, 0, 1))
        coverage = {0: .25, 1: .9}
        with patch.object(geometry, "render", side_effect=lambda cloud, *args, **kwargs: (None, np.full((4, 4), coverage[cloud]))),                 patch.object(scene, "_cloud", side_effect=lambda index: (index, 0.0)):
            self.assertEqual(scene.source_for((40, 0, 1)), 0)
            coverage[0] = .15  # too little of the view: the view that sees most of it wins
            self.assertEqual(scene.source_for((40, 0, 1)), 1)
            coverage[1] = .2  # unless the Start Image is close behind (START_BONUS)
            self.assertEqual(scene.source_for((40, 0, 1)), 0)

    def test_the_plate_marks_what_the_subject_sits_on_as_its_support(self):
        # Test 1: the plate kept the crate. At the subject's depth in the plate it is the support; where the plate
        # shows the wall behind the subject (much farther), the pixels stay part of the subject.
        class PlateDepth(FakeDepth):
            def reconstruct(self, images):
                depths, poses, k = super().reconstruct(images)
                if len(images) == 2:
                    depths[1][:40] = 12.0  # behind the person: painted-in wall
                return depths, poses, k

        mask = np.zeros((64, 64), bool)
        mask[10:60, 20:44] = True
        plate_mask = geometry.plate_mask(mask)
        scene = geometry.Scene(PlateDepth(), stripes(), mask, stripes(), plate_mask)
        support = scene.supports[0]
        self.assertFalse(support[10:40].any())
        self.assertTrue(support[40:60, 20:44].all())
        # The card already draws the support, so the plate layer covers only the rest of the subject's area.
        calls = []
        real = geometry.point_cloud
        with patch.object(geometry, "point_cloud", side_effect=lambda *args, **kwargs: calls.append(kwargs.get("only")) or real(*args, **kwargs)):
            scene.coarse((10, 0, 1), (64, 64))
        layer = next(only for only in calls if only is not None)
        self.assertTrue(np.array_equal(layer, plate_mask & ~support))

if __name__ == "__main__":
    unittest.main()
