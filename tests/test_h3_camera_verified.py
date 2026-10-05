"""Acceptance checks must fail closed and never expose a failed candidate."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_h3_camera_timing import timing, plugin_tests, make_plugin, PATH


def evidence():
    return dict(media=dict(frames=41),source_marks=[0,18,22,40],target_marks=[0,18,22,40],
                motion=[dict(frame=i,pixels=.01 if 18<=i<22 else 3) for i in range(40)])


class VerifiedTimingTests(unittest.TestCase):
    def test_pass_does_not_claim_geometry_or_roll(self):
        result=timing.assess_timing(evidence(),exact=True)
        self.assertFalse(result['geometry_verified'])
        self.assertFalse(result['roll_verified'])

    def test_static_untracked_and_moving_hold_rejected(self):
        for kind in ('static','untracked','hold','order','missing'):
            data=evidence()
            if kind=='static':
                for p in data['motion']: p['pixels']=0
            elif kind=='untracked':
                for p in data['motion'][:10]: p['pixels']=None
            elif kind=='hold': data['motion'][19]['pixels']=2
            elif kind=='order': data['motion'][2]['frame']=3
            else: data['motion'].pop()
            with self.subTest(kind=kind),self.assertRaises(ValueError): timing.assess_timing(data)

    def test_extreme_speed_and_inexact_output_rejected(self):
        data=evidence()
        data['target_marks']=[0,18,19,40]
        with self.assertRaises(ValueError): timing.assess_timing(data)
        data['target_marks']=[0,19,23,40]
        with self.assertRaises(ValueError): timing.assess_timing(data,exact=True)

    def test_failed_output_is_not_published_and_next_candidate_is_tried(self):
        with tempfile.TemporaryDirectory() as folder:
            def export(source,source_marks,target_marks,staging,*args):
                video=Path(staging)/'candidate.mp4'
                video.write_bytes(b'checked video')
                report=Path(staging)/'candidate.json'
                report.write_text('{}')
                return str(video),str(report),dict(output=str(video),target_marks_zero_based=target_marks)
            with patch.object(timing,'inspect_hold',side_effect=[evidence(),ValueError('Output failed'),evidence(),evidence()]),patch.object(timing,'export_retimed',side_effect=export):
                video,report,result=timing.verified_candidates(['bad.mp4','good.mp4'],PATH,2,folder)
            self.assertEqual(Path(video).read_bytes(),b'checked video')
            self.assertEqual([a['status'] for a in result['attempts']],['rejected','passed_timing_checks'])
            self.assertEqual(len(list(Path(folder).glob('*.mp4'))),1)
            self.assertFalse(list(Path(folder).glob('.h3-verified-*')))
            self.assertEqual(json.loads(Path(report).read_text())['output'],video)

    def test_all_rejected_returns_report_and_no_video(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(timing,'inspect_hold',side_effect=ValueError('No hold')),patch.object(timing,'export_retimed') as export:
            video,report,result=timing.verified_candidates(['one.mp4','two.mp4'],PATH,2,folder)
            self.assertIsNone(video)
            self.assertEqual(result['status'],'rejected')
            self.assertEqual(len(result['attempts']),2)
            self.assertTrue(Path(report).is_file())
            export.assert_not_called()

    def test_limit_and_invalid_input_clear_previous_result(self):
        plugin=make_plugin()
        for sources in (None,[],['a']*4):
            video,report,status=plugin.verify_clip_candidates(sources,PATH,2,'retime','nearest')
            self.assertIsNone(video)
            self.assertIsNone(report)
            self.assertIn('No result',status)

    def test_native_api_registration(self):
        fixture=plugin_tests.NativeCameraTests()
        fixture.setUp()
        config,_=fixture.build_real_plugin_ui()
        event=next(e for e in config['dependencies'] if e.get('api_name')=='h3_camera_verify_candidates')
        self.assertEqual(len(event['inputs']),5)
        self.assertEqual(len(event['outputs']),3)
