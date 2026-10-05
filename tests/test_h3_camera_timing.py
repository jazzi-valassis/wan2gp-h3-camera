"""Verify timing boundaries, real encoded frames, audio modes and source safety."""
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import test_h3_camera_plugin as plugin_tests
from test_h3_camera_plugin import camera, make_plugin
from test_h3_camera_anchor_plugin import PATH

timing = importlib.import_module(camera.__package__+'.timing')


class TimingMapTests(unittest.TestCase):
    def test_selected_frames_are_exact_and_mapping_never_reverses(self):
        mapping = timing.frame_map(243,[0,95,136,242],[0,121,133,242])
        self.assertEqual([mapping[index] for index in (0,121,133,242)],[0,95,136,242])
        self.assertTrue(all(a < b for a,b in zip(mapping,mapping[1:])))
        self.assertEqual(len(mapping),243)

    def test_identity_mapping_does_not_change_a_frame(self):
        self.assertEqual(timing.frame_map(41,[0,10,20,40],[0,10,20,40]),list(range(41)))

    def test_bad_or_unordered_marks_are_rejected(self):
        for source,target in [([0,10,10,40],[0,20,24,40]),([0,30,20,40],[0,20,24,40]),
                              ([0,10,20,40],[0,24,20,40]),([1,10,20,40],[0,20,24,40]),
                              ([0,10,20,41],[0,20,24,40]),([0,True,20,40],[0,20,24,40]),
                              ([0,10.5,20,40],[0,20,24,40]),([0,None,20,40],[0,20,24,40]),
                              ([0,float('nan'),20,40],[0,20,24,40]),([0,10,40],[0,20,24,40])]:
            with self.subTest(source=source,target=target),self.assertRaises(ValueError):
                timing.frame_map(41,source,target)

    def test_planned_hold_uses_actual_clip_geometry(self):
        self.assertEqual(timing.held_path_frames(PATH,2,243),(121,133))
        self.assertEqual(timing.held_path_frames(PATH,2,227),(113,124))
        with self.assertRaises(ValueError):
            timing.held_path_frames(PATH,1,243)
        with self.assertRaises(ValueError):
            timing.held_path_frames(PATH,True,243)

    def test_motion_suggestions_require_valid_tracks_overlapping_hold(self):
        pairs = [dict(frame=i,pixels=.01 if 10<=i<20 else 3) for i in range(40)]
        self.assertEqual(timing.find_stationary_interval(pairs,12,16),(10,20))
        with self.assertRaises(ValueError):
            timing.find_stationary_interval(pairs,22,25)
        with self.assertRaises(ValueError):
            timing.find_stationary_interval([dict(frame=i,pixels=None) for i in range(40)],12,16)

    def test_timing_controls_register_separate_native_apis(self):
        fixture = plugin_tests.NativeCameraTests()
        fixture.setUp()
        config,_ = fixture.build_real_plugin_ui()
        events = {event.get('api_name'):event for event in config['dependencies']}
        for name,inputs,outputs in [('h3_camera_extract_checkpoint',2,2),('h3_camera_inspect_timing',3,3),('h3_camera_correct_timing',7,3)]:
            self.assertEqual(len(events[name]['inputs']),inputs)
            self.assertEqual(len(events[name]['outputs']),outputs)

    def test_missing_video_fails_without_export(self):
        plugin = make_plugin()
        with patch.object(timing,'export_retimed') as export:
            with self.assertRaises(camera.gr.Error):
                plugin.correct_clip_timing(None,PATH,2,2,3,'retime','nearest')
            export.assert_not_called()


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required for media integration')
class TimingMediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='h3-camera-timing-test-')
        cls.root = Path(cls.directory.name)
        cls.source = cls.root/'source.mkv'
        pictures = []
        for i in range(41):
            picture = np.full((64,96,3),i*6,dtype=np.uint8)
            for bit in range(6):
                picture[16:48,bit*16:(bit+1)*16] = 235 if i & (1<<bit) else 16
            pictures.append(picture.tobytes())
        frames = b''.join(pictures)
        subprocess.run(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s','96x64','-r','10','-i','pipe:0',
                        '-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=4.1',
                        '-map','0:v:0','-map','1:a:0','-c:v','ffv1','-c:a','pcm_s16le','-t','4.1',str(cls.source)],
                       input=frames,check=True,capture_output=True,timeout=30,
                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        cls.source_hash = hashlib.sha256(cls.source.read_bytes()).hexdigest()

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_extracted_frame_is_one_based_and_out_of_range_is_rejected(self):
        picture,info,frame = timing.extract_frame(self.source,11)
        self.assertEqual(frame,10)
        self.assertEqual(info['frames'],41)
        self.assertEqual(self.frame_number(np.array(picture)),10)
        for value in (0,42,True,1.5):
            with self.subTest(value=value),self.assertRaises(ValueError):
                timing.extract_frame(self.source,value)

    def test_real_export_keeps_exact_boundary_frames_and_audio(self):
        for audio,sampling in [('retime','nearest'),('preserve','blend'),('mute','nearest')]:
            with self.subTest(audio=audio,sampling=sampling):
                output,report,record = timing.export_retimed(self.source,[0,10,20,40],[0,20,24,40],self.root,audio=audio,sampling=sampling)
                self.assertNotEqual(Path(output),self.source)
                info = timing.probe_video(output)
                self.assertEqual((info['frames'],info['fps'],info['width'],info['height']),(41,10,96,64))
                self.assertEqual(info['has_audio'],audio!='mute')
                self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(),self.source_hash)
                capture = cv2.VideoCapture(output)
                try:
                    for index,expected in [(0,0),(20,10),(24,20),(40,40)]:
                        capture.set(cv2.CAP_PROP_POS_FRAMES,index)
                        ok,pixels = capture.read()
                        self.assertTrue(ok)
                        self.assertEqual(self.frame_number(pixels),expected)
                finally:
                    capture.release()
                self.assertEqual(json.loads(Path(report).read_text())['target_marks_zero_based'],[0,20,24,40])
                self.assertEqual(record['full_decode'],'passed')
                if audio!='mute':
                    decoded = subprocess.check_output(['ffmpeg','-v','error','-i',output,'-map','0:a:0','-f','f32le','-ar','48000','-ac','1','-'],timeout=30)
                    waveform = np.frombuffer(decoded,dtype=np.float32)
                    self.assertAlmostEqual(len(waveform)/48000,4.1,delta=.04)
                    spectrum = np.abs(np.fft.rfft(waveform[:48000]))
                    self.assertAlmostEqual(float(np.argmax(spectrum)),440,delta=3)

    @staticmethod
    def frame_number(pixels):
        return sum((1<<bit) for bit in range(6) if pixels[20:44,bit*16+4:bit*16+12].mean()>128)

    def test_export_rejects_bad_options_before_creating_any_output(self):
        before = set(self.root.iterdir())
        with self.assertRaises(ValueError):
            timing.export_retimed(self.source,[0,10,20,40],[0,20,24,40],self.root,audio='unknown')
        self.assertEqual(set(self.root.iterdir()),before)

    def test_irregular_timestamps_are_rejected_despite_matching_rate_metadata(self):
        data = dict(streams=[dict(codec_type='video',avg_frame_rate='10/1',r_frame_rate='10/1',
                                 nb_read_frames='3',width=96,height=64)],
                    frames=[dict(media_type='video',best_effort_timestamp_time=str(value)) for value in (0,.1,.25)])
        with patch.object(timing,'_run',return_value=SimpleNamespace(stdout=json.dumps(data))),self.assertRaisesRegex(ValueError,'timestamps'):
            timing.probe_video(self.source)


if __name__ == '__main__':
    unittest.main()
