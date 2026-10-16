# Attribution

H3 Camera for WanGP is maintained by Jazzi:

https://github.com/jazzi-valassis/wan2gp-h3-camera

Copyright (c) 2026 Jazzi. The plugin package is distributed under the MIT license in `LICENSE`.

H3 Camera's workflow and normalized camera-coordinate schema were inspired by NyckM's `3d-Camera-control-H3-Minimax` project:

https://github.com/NyckM/3d-Camera-control-H3-Minimax

The Python compiler, Gradio integration and browser editor are original implementations. This distribution does not include the reference project's ComfyUI nodes or the host checkout's `h3cam_ref` folder.

WanGP is a separate application by DeepBeepMeep:

https://github.com/deepbeepmeep/Wan2GP

This plugin imports WanGP's installed plugin API, scheduler and prompt parser. No WanGP core or model source is distributed in this archive. The license here applies to the plugin package; WanGP and models retain their own licenses.

Generated view anchors use models that WanGP downloads on first use; none are distributed in this archive:

- Depth Anything 3 (bundled with WanGP's preprocessing) estimates depth and camera poses.
- rembg's u2net (bundled with WanGP) finds the main subject.
- Qwen Image 2.1 by the Qwen team renders the views and paints the subject out of the background plate.
- AnyAngle LoRA for Qwen Image 2.1 by lilylilith, Apache 2.0: https://huggingface.co/lilylilith/QI_2.1_AnyAngle
- Pruna 8-step LoRA for Qwen Image 2.1, as hosted by DeepBeepMeep: https://huggingface.co/DeepBeepMeep/Qwen_image_2
