"""Timed images must agree with native frame indices without altering the path."""
import json
import unittest

from test_h3_camera_plan import camera


def path():
    return [camera.ORIGIN, dict(time=.5, azimuth=90, elevation=0, distance=1),
            dict(time=133/242, azimuth=90, elevation=0, distance=1),
            dict(time=1, azimuth=90, elevation=89, distance=1)]


class ImageAnchorPlanTests(unittest.TestCase):
    def test_movement_checkpoint_preserves_segments_and_requests_no_extra_stop(self):
        requested = path()
        result = camera.compile_plan(json.dumps(requested), prompt='Scene', frame_count=243, fps=24,
                                     image_anchors=[dict(frame=157,picture=4)])
        self.assertEqual(result['path'], requested)
        self.assertEqual(len(result['rows']),4)
        lines=[line for line in result['prompt'].splitlines() if line.startswith('[')]
        self.assertEqual(len(lines),3)
        self.assertIn('At 6.541667s, pass through the camera composition shown in <Picture 4> without stopping',lines[2])
        processed='\n'.join(line for line in result['prompt'].splitlines() if not line.startswith('#'))
        again=camera.compile_plan(json.dumps(requested),prompt=processed,frame_count=243,fps=24,image_anchors=[dict(frame=157,picture=4)])
        self.assertEqual(again['prompt'],result['prompt'])

    def test_native_checkpoint_inside_a_hold_does_not_request_motion(self):
        result=camera.compile_plan(json.dumps(path()),prompt='Scene',frame_count=243,fps=24,
                                   image_anchors=[dict(frame=127,picture=4)])
        self.assertIn('At 5.291667s, maintain the stationary camera composition shown in <Picture 4>',result['prompt'])
        self.assertNotIn('pass through',result['prompt'])

    def test_images_anchor_the_turn_hold_and_departure_without_changing_path(self):
        requested = path()
        result = camera.compile_plan(json.dumps(requested), prompt='A robot.\noverall_soundscape: Workshop.',
                                     frame_count=239, fps=24, image_anchors=[dict(frame=121,picture=2), dict(frame=133,picture=3)])
        self.assertEqual(result['path'], requested)
        self.assertEqual(result['frame_count'], 243)
        lines = [line for line in result['prompt'].splitlines() if line.startswith('[')]
        self.assertIn('Reach the camera composition shown in <Picture 2>', lines[0])
        self.assertIn('from <Picture 2> through <Picture 3>', lines[1])
        self.assertIn('Start from the camera composition shown in <Picture 3>', lines[2])
        self.assertIn('subject motion still follows the scene instructions', lines[1])
        self.assertIn('Workshop.', result['prompt'])

    def test_reapply_owned_and_comment_stripped_anchors_is_idempotent(self):
        kwargs = dict(frame_count=243, fps=24, image_anchors=[dict(frame=121,picture=2), dict(frame=133,picture=3)])
        result = camera.compile_plan(json.dumps(path()), prompt='A scene with <Picture 4>.', **kwargs)
        processed = '\n'.join(line for line in result['prompt'].splitlines() if not line.startswith('#'))
        for prompt in (result['prompt'], processed):
            again = camera.compile_plan(json.dumps(path()), prompt=prompt, **kwargs)
            self.assertEqual(again['prompt'], result['prompt'])
        removed = camera.compile_plan(json.dumps(path()), prompt=processed, frame_count=243, fps=24)
        self.assertNotIn('camera composition shown in <Picture 2>', removed['prompt'])
        self.assertIn('A scene with <Picture 4>', removed['prompt'])

    def test_invalid_anchors_fail_and_half_frames_round_consistently(self):
        self.assertEqual(camera.keyframe_frame(dict(time=10.5/242),243),11)
        for anchors in ([dict(frame=243,picture=2)], [dict(frame=True,picture=2)],
                        [dict(frame=121,picture=0)], [dict(frame=121,picture=2)]*2):
            with self.subTest(anchors=anchors), self.assertRaises(ValueError):
                camera.compile_plan(json.dumps(path()),prompt='Scene',frame_count=243,fps=24,image_anchors=anchors)

if __name__ == '__main__':
    unittest.main()
