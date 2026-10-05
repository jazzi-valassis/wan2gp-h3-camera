"""Native image ordering and repeated hold application preserve user inputs."""
import json
import unittest
from unittest.mock import patch
import gradio as gr
from PIL import Image

from test_h3_camera_plugin import make_plugin, arguments, camera


PATH = json.dumps([dict(time=0,azimuth=0,elevation=0,distance=1),dict(time=.5,azimuth=90,elevation=0,distance=1),
                   dict(time=133/242,azimuth=90,elevation=0,distance=1),dict(time=1,azimuth=90,elevation=89,distance=1)])

class HoldImagePluginTests(unittest.TestCase):
    def setUp(self):
        self.plugin = make_plugin()
        self.start = Image.new('RGB',(128,64),'red')
        self.side = Image.new('RGB',(128,64),'blue')
        self.ref = Image.new('RGB',(128,64),'green')

    def args(self, **changes):
        options = dict(prompt='A robot from <Picture 2>.\noverall_soundscape: A quiet hum.', video_length=239,
                       image_prompt_type='S',image_start=[self.start],video_prompt_type='I',image_refs=[self.ref])
        options.update(changes)
        args = arguments(**options)
        args[0] = PATH
        return [*args,True,'']

    def outputs(self, result):
        return dict(zip((*camera.FORM_OUTPUTS,'table','compiled','summary',*camera.ANCHOR_OUTPUTS),result))

    def updated_args(self,args,result):
        args=list(args)
        for name in camera.FORM_INPUTS:
            update=result.get(name,{})
            if 'value' in update:
                args[4+camera.FORM_INPUTS.index(name)]=update['value']
        args[-1]=result['frames_positions']['value']
        return args

    def test_checkpoint_preserves_hold_references_and_is_repeatable(self):
        args=self.args()
        hold=self.outputs(self.plugin.apply_hold_image(self.side,2,*args))
        args=self.updated_args(args,hold)
        checkpoint=Image.new('RGB',(128,64),'yellow')
        added=self.outputs(self.plugin.apply_checkpoint_image(checkpoint,157/24,*args))
        self.assertEqual(added['frames_positions']['value'],'122 134 158')
        self.assertEqual(added['image_refs']['value'],[self.side,self.side,checkpoint,self.ref])
        self.assertIn('A robot from <Picture 5>',added['prompt']['value'])
        self.assertIn('from <Picture 2> through <Picture 3>',added['prompt']['value'])
        self.assertIn('At 6.541667s, pass through the camera composition shown in <Picture 4>',added['prompt']['value'])
        self.assertEqual(added['table'],hold['table'])
        args=self.updated_args(args,added)
        repeated=self.outputs(self.plugin.apply_checkpoint_image(checkpoint,157/24,*args))
        self.assertEqual(repeated['image_refs']['value'],added['image_refs']['value'])
        self.assertEqual(repeated['prompt']['value'],added['prompt']['value'])
        rehold=self.outputs(self.plugin.apply_hold_image(self.side,2,*args))
        self.assertEqual(rehold['image_refs']['value'],added['image_refs']['value'])
        self.assertEqual(rehold['prompt']['value'],added['prompt']['value'])
        normal=dict(zip(camera.FORM_OUTPUTS,self.plugin.apply_plan(*args)))
        self.assertEqual(normal['prompt']['value'],added['prompt']['value'])

    def test_earlier_checkpoint_renumbers_existing_image_roles_in_time_order(self):
        args=self.args()
        hold=self.outputs(self.plugin.apply_hold_image(self.side,2,*args))
        args=self.updated_args(args,hold)
        prompt_index=4+camera.FORM_INPUTS.index('prompt')
        args[prompt_index]+='\n# Hold image <Picture 2>, identity <Picture 4>.'
        added=self.outputs(self.plugin.apply_checkpoint_image(self.ref,2,*args))
        self.assertEqual(added['frames_positions']['value'],'122 134 49')
        self.assertIn('Hold image <Picture 3>, identity <Picture 5>',added['prompt']['value'])
        self.assertIn('from <Picture 3> through <Picture 4>',added['prompt']['value'])
        self.assertIn('At 2.000000s, pass through the camera composition shown in <Picture 2>',added['prompt']['value'])

    def test_checkpoint_replaces_last_duplicate_and_keeps_native_end_label(self):
        args=self.args(prompt='End <Picture 2>, middle <Picture 3>, identity <Picture 4>.',
                       video_prompt_type='FI',image_refs=[self.side,self.side,self.side,self.ref])
        args[-1]='158 L 158'
        result=self.outputs(self.plugin.apply_checkpoint_image(self.start,157/24,*args))
        self.assertEqual(result['image_refs']['value'],[self.side,self.side,self.start,self.ref])
        self.assertEqual(result['frames_positions']['value'],'158 243 158')
        self.assertIn('End <Picture 2>, middle <Picture 3>, identity <Picture 4>',result['prompt']['value'])

    def test_checkpoint_rejects_invalid_times_without_changing_inputs(self):
        for seconds in (None,True,float('nan'),-1,0,121/24,127/24,133/24,242/24,11):
            args=self.args()
            before=list(args)
            with self.subTest(seconds=seconds),self.assertRaises(gr.Error):
                self.plugin.apply_checkpoint_image(self.side,seconds,*args)
            self.assertEqual(args,before)
        with self.assertRaises(gr.Error):
            self.plugin.apply_checkpoint_image(None,6.5,*self.args())
        with self.assertRaises(gr.Error):
            self.plugin.apply_checkpoint_image(self.side,6.5,*self.args(video_prompt_type=''))

    def test_invalid_native_fps_is_rejected_before_timed_image_mapping(self):
        args=self.args(video_prompt_type='FI',image_refs=[self.side,self.side,self.ref])
        args[-1]='122 134'
        for fps in (0,-24,float('nan'),float('inf')):
            self.plugin.get_computed_fps=lambda *unused,value=fps:value
            with self.subTest(fps=fps),self.assertRaises(gr.Error):
                self.plugin.apply_checkpoint_image(self.side,6.5,*args)

    def test_checkpoint_rejects_incomplete_or_other_window_injections(self):
        for positions in ('X','500','122 134'):
            args=self.args(video_prompt_type='FI')
            args[-1]=positions
            with self.subTest(positions=positions),self.assertRaises(gr.Error):
                self.plugin.apply_checkpoint_image(self.side,6.5,*args)

    def test_hold_helper_uses_native_indices_and_preserves_other_references(self):
        args = self.args()
        result = self.outputs(self.plugin.apply_hold_image(self.side,2,*args))
        self.assertEqual(result['frames_positions']['value'],'122 134')
        self.assertEqual(result['video_prompt_type']['value'],'FI')
        self.assertEqual(result['image_refs']['value'],[self.side,self.side,self.ref])
        self.assertIn('A robot from <Picture 4>',result['prompt']['value'])
        self.assertIn('from <Picture 2> through <Picture 3>',result['prompt']['value'])
        self.assertIn('A quiet hum.',result['prompt']['value'])
        self.assertEqual(result['video_length']['value'],243)
        self.assertEqual(args[4+camera.FORM_INPUTS.index('image_refs')],[self.ref])

    def test_repeated_helper_and_normal_apply_do_not_duplicate_images_or_labels(self):
        args = self.args()
        first = self.outputs(self.plugin.apply_hold_image(self.side,2,*args))
        for name in camera.FORM_INPUTS:
            update = first.get(name,{})
            if 'value' in update:
                args[4+camera.FORM_INPUTS.index(name)] = update['value']
        args[-1] = first['frames_positions']['value']
        again = self.outputs(self.plugin.apply_hold_image(self.side,2,*args))
        self.assertEqual(again['image_refs']['value'],first['image_refs']['value'])
        self.assertEqual(again['prompt']['value'],first['prompt']['value'])
        normal = dict(zip(camera.FORM_OUTPUTS,self.plugin.apply_plan(*args)))
        self.assertEqual(normal['prompt']['value'],first['prompt']['value'])

    def test_existing_end_image_keeps_its_number(self):
        args = self.args(prompt='Start <Picture 1>, end <Picture 2>, robot <Picture 3>.',image_prompt_type='SE',image_end=[self.ref])
        result = self.outputs(self.plugin.apply_hold_image(self.side,2,*args))
        self.assertIn('Start <Picture 1>, end <Picture 2>, robot <Picture 5>',result['prompt']['value'])
        self.assertIn('from <Picture 3> through <Picture 4>',result['prompt']['value'])
        self.assertIn('Reach the camera composition shown in <Picture 2> at the end of this segment.',result['prompt']['value'])

    def test_injected_end_is_labeled_as_native_end_image(self):
        mapped=self.plugin.native_image_anchors(243,24,[self.start],None,[self.side,self.ref], '122 L','FI')
        self.assertEqual(mapped,[dict(frame=242,picture=2),dict(frame=121,picture=3)])

    def test_invalid_input_does_not_change_refs_or_prompt(self):
        for image, number, changes in [(None,2,{}),(self.side,3,{}),(self.side,2,{'image_prompt_type':''})]:
            args = self.args(**changes)
            before = list(args)
            with self.assertRaises(gr.Error):
                self.plugin.apply_hold_image(image,number,*args)
            self.assertEqual(args,before)
        args=self.args(video_prompt_type='FI')
        args[-1]='30'
        with self.assertRaises(gr.Error):
            self.plugin.apply_hold_image(self.side,2,*args)

    def test_native_mapping_sorts_frames_and_uses_the_last_duplicate(self):
        mapped=self.plugin.native_image_anchors(243,24,[self.start],None,[self.side,self.ref,self.side], '134 122 122','FI')
        self.assertEqual(mapped,[dict(frame=121,picture=2),dict(frame=133,picture=3)])

    def test_missing_host_helper_leaves_ordinary_camera_apply_available(self):
        with patch.object(camera, 'window_contexts', None):
            self.plugin.apply_plan(*self.args())
            with self.assertRaises(gr.Error):
                self.plugin.apply_hold_image(self.side,2,*self.args())

    def test_helper_rejects_fl2va_and_ambiguous_end_images(self):
        definition=self.plugin.get_model_def('h3')
        self.plugin.get_model_def=lambda model: dict(definition,reference_image_enabled=False)
        with self.assertRaises(gr.Error):
            self.plugin.apply_hold_image(self.side,2,*self.args())
        self.plugin.get_model_def=lambda model: definition
        with self.assertRaises(gr.Error):
            self.plugin.apply_hold_image(self.side,2,*self.args(image_prompt_type='SE',image_end=[self.ref,self.ref]))
        self.plugin.get_model_def=lambda model: dict(definition,custom_frames_injection=False)
        with self.assertRaises(gr.Error):
            self.plugin.apply_hold_image(self.side,2,*self.args())

if __name__ == '__main__':
    unittest.main()
