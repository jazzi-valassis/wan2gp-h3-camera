"""Exact-pose geometry: orbit conventions, re-rendering, gap filling and view registration."""
import importlib.util
from pathlib import Path
import unittest

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
        points, colors, source, size, outline = geometry.point_cloud(stripes(), depth, K)
        self.assertTrue(bool(((source - 3.0).abs() < 1e-9).sum() >= 64))
        self.assertEqual(size, (64, 64))
        self.assertEqual(outline.shape[1], 3)

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


if __name__ == "__main__":
    unittest.main()
