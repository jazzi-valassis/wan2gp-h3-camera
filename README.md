# H3 Camera for WanGP

A visual single-shot camera planner for MiniMax H3 in [WanGP / Wan2GP](https://github.com/deepbeepmeep/Wan2GP). Sketch an orbit, rise, push or pull, preview the path, then apply its camera instructions to the native generation form.

**Author and maintainer:** [Jazzi](https://github.com/jazzi-valassis)

**Version:** 0.4.0 · **License:** [MIT](LICENSE) · **Plugin type:** extension

The plugin adds no model downloads or GPU allocations. Camera movement is prompt guidance: the diagram does not impose an exact 3D trajectory on the model. Optional timing tools use FFmpeg/FFprobe on PATH and WanGP's existing OpenCV, NumPy and Pillow packages.

## Features

- Eight camera presets, editable keyframes, numeric controls, and a scrub/play preview.
- Native frame-count and FPS timing, smooth or linear motion, and optional frozen-scene instructions.
- Preview without changing the generation form; Apply replaces the previous camera plan while retaining scene, audio and reference instructions.
- Portable JSON plans and optional closed-loop conditioning using the existing Start Image.
- Elevation instructions describe physical camera movement, lens tilt, and the requested endpoint view.
- Optional roll stabilization, explicit stationary holds, and a button to insert a half-second hold.
- Generated view anchors: one click renders each keyframe view from the Start Image (Qwen Image Edit 2511 + Multiple-Angles LoRA) and injects it at the keyframe, so H3 lands on the planned side and height.
- Native image anchors at both ends of a hold, with automatic Picture numbering and reference preservation.
- Timed image checkpoints within a move, with pass-through instructions that do not add stops or alter the saved path.
- Extract a checkpoint image directly from a reviewed video frame.
- Optional timing correction for an existing clip: align selected arrival/departure frames to one planned hold, with explicit audio and frame-sampling choices.

## Install

This is a standalone **extension** plugin. Its Python modules, browser assets and tests are all inside this directory. No files need to be copied into WanGP's `shared/`, `models/`, or `defaults/` directories.

### Install through WanGP

In WanGP, open **Plugins**. Under **Install New Plugin**, paste:

```text
https://github.com/jazzi-valassis/wan2gp-h3-camera
```

Click **Download and Install Plugin**, enable **H3 Camera**, and click **Save Settings**. Restart WanGP after existing work finishes. For supported H3 video models, **H3 Camera - single-shot planner** appears below the main prompt in **Media Generator**.

The repository is public; no GitHub account is required to install or download it. Keep only one installed copy. For an existing installation, use the [update instructions](#update-an-existing-installation).

### Clone with Git

From the **WanGP root directory**, run:

```sh
git clone https://github.com/jazzi-valassis/wan2gp-h3-camera.git plugins/wan2gp-h3-camera
```

An existing `plugins/wan2gp-h3-camera` directory must be backed up outside `plugins/` before cloning into that location. Do not keep a second enabled copy under another name.

Enable **H3 Camera**, save settings, and restart WanGP as above.

### Install from a ZIP

Download the prepared plugin ZIP from the [releases page](https://github.com/jazzi-valassis/wan2gp-h3-camera/releases). Extract it into WanGP's `plugins` directory, then enable the plugin and restart as above. Each prepared release includes a SHA-256 checksum and per-file manifest. To package this checkout as `wan2gp-h3-camera-0.4.0.zip`, run `python scripts/build_release.py`. Use the Git installation method for the latest source version; downloadable releases may lag behind main.

The final layout must be:

```text
Wan2GP/
  plugins/
    wan2gp-h3-camera/
      __init__.py
      plugin.py
      plugin_info.json
      camera_plan.py
      editor.py
      image_anchors.py
      timing.py
      web/
```

GitHub's **Code > Download ZIP** source archive uses a branch suffix such as `wan2gp-h3-camera-main`. Rename that extracted folder to `wan2gp-h3-camera` before placing it under `plugins/`. The prepared release ZIP already uses the correct name.

### Update an existing installation

For an existing installation, keep the folder name `wan2gp-h3-camera`. Back up the old folder outside `plugins/` or name the backup with a `.bak` suffix. Replace it with the new package. Keep your existing enabled-plugin setting and restart when convenient. Installing a second copy under another name creates conflicting editor IDs.

For a Git installation, use the update action in WanGP's **Plugins** tab, or fetch updates from the WanGP root with:

```sh
git -C plugins/wan2gp-h3-camera pull --ff-only
```

Review and preserve local edits before updating. Restart WanGP to load changed Python and browser assets. A ZIP installation has no Git remote; update it by replacing the plugin folder.

The repository root contains the plugin files directly, so the public URL can be used with WanGP's GitHub installer.

## Compatibility

Version **0.4.0** is tested with the installed **WanGP v17.01**, host HEAD `0e58385fbde7ff102d276e4a9e490845de76b4ea` with existing local host edits, and **Gradio 5.29.0**. Image anchoring needs the native `shared.prompt_enhancer.images.window_contexts` and `resolve_injected_positions` helpers and an H3 Ref2VA model supporting Inject Frames. If those helpers are absent, the image-anchor buttons are disabled; ordinary prompt planning remains available when the other requested host APIs exist. Timing correction needs FFmpeg/FFprobe on PATH. Older releases are not certified for the new image workflow. See [VALIDATION.md](VALIDATION.md) for the exact local checks.

The plugin uses `WAN2GPPlugin`, component/global requests, `insert_after`, and `add_custom_js`, plus the installed frame scheduler, prompt parser and native image-label helper. These are WanGP dependencies, not additional files to ship. It does not patch the pipeline, launch another server, submit its own generation jobs, or import `h3cam_ref` or MiniMaxH3Mod.

Missing required form controls or injected host functions produce a compatibility message when WanGP can render the insertion point. The plugin also requests WanGP's native prompt/duration label helpers and prompt-tool attachment script to keep labels synchronized after Apply. The static version field is not a promise that every older or future host layout works.

## Use

1. Enable **H3 Camera** in the Plugin Manager and restart WanGP if it is not already enabled.
2. Select an H3 **FL2VA** or **Ref2VA** video model. Compatible H3 Multishot variants also work for one shot.
3. Write one scene in the main prompt, including any desired dialogue and sound. Set the native frame count and FPS.
4. Open **H3 Camera - single-shot planner** below the prompt. Choose a preset, edit keyframes and preview the camera move. Time is normalized across the selected clip, with the final pose at its last frame.
5. Choose whether the scene can animate (default) or is frozen, and smooth or linear camera movement. **Stabilize camera roll** is on by default; turn it off for intentional banking or lens-axis rotation. **Preview camera prompt** changes no generation settings.
6. Click **Apply camera path to generation form**, then use the normal **Generate** or **Add to Queue** button. Apply again after editing the path, scene, or duration.

Select a keyframe in the strip, then drag the diagram horizontally to orbit or vertically to change elevation. The first drag direction locks that axis until release. Focus the diagram and use the mouse wheel to change distance, or use its arrow keys (hold Shift for larger steps). Numeric controls provide precise edits. **Add** inserts a keyframe between the selected pose and the next pose; the first pose stays fixed. Scrub or play to preview the planned move.

### Orbit angle and segment turn

**Orbit angle** is the keyframe's absolute position around the subject, measured from the start view. It is not an amount added at that keyframe. **Segment turn** shows how far the camera orbits since the previous keyframe; the keyframe strip shows it in parentheses. To orbit 90 degrees and then crane overhead without further sideways movement, give the last keyframe the same orbit angle as the previous one (`90`, turn `0`) and raise only its elevation. Setting that keyframe's orbit angle to `0` instead returns the camera to the start side, which the prompt describes as a reversed orbit.

Editing **Segment turn** moves the selected keyframe and shifts every later keyframe by the same amount, so their own turns stay unchanged. Editing **Orbit angle**, dragging, or using the arrow keys moves only the selected keyframe.

The compiled prompt states when a segment stops orbiting and when it reverses the latest orbit direction. An orbit followed by a held angle names its rest point, and later segments at that angle retain the stopped-orbit instruction, including after a pause. This constrains the camera's side without instructing an animated subject to stop turning. Keyframes keep the same absolute format.

H3 can still overshoot a requested orbit. In the 0.2.4 test renders a 90-degree orbit sometimes reached about 180 degrees before the camera held its angle. The local 0.2.5 comparison also overshot, including with 0.5- and 1-second holds. Review the render; a hold is not a guaranteed correction. A smaller planned angle is a separate experiment and changes the intended path.

The 0.2.6 image-anchor workflow below reached the intended side view and held it in the tested robot scene at two seeds. It still permits timing drift: the camera can dwell beyond the requested hold before rising. Text-only Apply has no geometric enforcement, and merely installing this version does not add image conditioning to existing jobs.

Version 0.2.7 adds a movement checkpoint to guide departure from the hold. It also includes native images between camera keyframes in the compiled instructions; 0.2.6 only linked images coinciding with keyframes. The new checkpoint is a view to pass through, so it does not introduce another stop or change the original path times.

With the selected image checkpoint, both tested seeds resumed sustained background motion at 5.625 seconds after a hold ending at 5.541667 seconds. The previous same-seed comparison resumed at 6.625 seconds. This corrects the tested late departure; arrival at the side view can still occur early. See [VALIDATION.md](VALIDATION.md) for measurements and the configured local example.

Applying updates the prompt, makes all lines part of one prompt, aligns the frame count to H3's native `17k+5` grid, selects one sliding window, and sets both trim controls to zero. Existing reference media, audio, LoRAs, model selection, inference settings and memory settings remain in the normal form. Applying again replaces the prior generated camera instructions rather than adding duplicates.

For a **closed loop**, choose a path ending at its original camera pose (for example Full orbit), activate the native **Start Image** and provide exactly one image. Enable the loop checkbox before applying. The plugin copies that image into **End Image**, replacing its previous contents, and enables endpoint conditioning. With no loop requested, the plugin leaves the existing End Image alone. Matching endpoints does not guarantee that H3 completes an orbit.

If reference images are active, adding an End Image changes their H3 picture numbers. The plugin stops this operation with an explanation: first enable and fill the native End Image, then update the scene's picture labels (Start Image is Picture 1, End Image is Picture 2, other images follow). An already active, populated End Image keeps its existing position when replaced.

Use **Camera path JSON and saved plans** to save or load portable `.json` plans. Saved plans include keyframes, scene-motion choice, interpolation, loop choice and roll stabilization. Older plans and bare keyframe arrays load with stabilization on. They do not overwrite the main scene prompt or duration. Bare keyframe arrays from the original editor are supported when they satisfy the normalized schema. The ordinary WanGP queue stores the compiled prompt and generation settings, so queued jobs do not need the planner to run.

### Generated view anchors (recommended for orbits)

Prompt wording cannot say which side of the subject an orbit should finish on, and some seeds circle past the subject's back to the far side. **Generate view anchors and apply camera path** gives H3 a picture of each target view instead:

1. Select an H3 **Ref2VA** model and add exactly one active **Start Image**. Keep the identity reference in **Reference Images** if you use one.
2. Plan the path. A keyframe gets a view when its orbit angle is within 5 degrees of a 45-degree step (±45, ±90, ±135, 180) and its elevation within 15 degrees of -30, 0, 30 or 60. The start view is never re-anchored (the Start Image already shows it). When several keyframes round to the same view, only the closest pose keeps it, together with its exact repeats (a hold): the same still at two different poses would read as a stop. Other keyframes keep text guidance only, so a dense path with a keyframe every second is anchored only at its exact views.
3. Click **Generate view anchors and apply camera path**. The plugin queues one Qwen Image Edit Plus (2511) image per distinct keyframe view, rendered from the Start Image with fal's Multiple-Angles LoRA and the 8-step Lightning LoRA. It then injects each view at its keyframe frame (a hold shares one view at both ends), renumbers your `<Picture N>` labels and applies the camera path. Generate normally afterwards.

The views use WanGP's normal queue, so they appear in the gallery and need the Media Generator tab to stay focused while they render (about 15-30 seconds per view plus a model switch). The model and both LoRAs download on first use. **View image seed** changes the generated views; apply again to replace them. Distances map to the LoRA's shot sizes (close-up at 0.6x or closer, wide at 1.6x or farther, medium otherwise), so the anchored framing is approximate and depends on how wide the Start Image is.

Views are generated stills: they fix the camera side, height and the endpoint view, and they also fix the subject's pose and the background at those frames. Review them before generating; regenerate with another seed if a view is wrong.

### Image anchors for a camera hold

1. Select H3 **Ref2VA**, add exactly one active **Start Image**, and make a hold using two identical poses at different times.
2. Open **Image anchors for a camera hold**. Supply an image of the desired held view, such as a reviewed side-view frame from a previous render. Use the Start Image's aspect ratio: native Inject Frames takes its canvas from the first injected image.
3. Set **Hold starts at keyframe** to the first of those equal poses (usually 2 for orbit, hold, rise). Click **Anchor hold and apply camera path**, then Generate normally.

The button puts two copies of that image at the nearest displayed frames to the hold boundaries, using native 1-based frame positions. It preserves normalized keyframe times, retains the remaining reference images, and updates their `<Picture N>` labels in the scene. Start and active End Image numbers remain unchanged. The prompt links the incoming move, hold and departure to the native images; an active end image is also linked to the final camera segment. Repeated application at the same boundaries does not duplicate images or renumber them again.

This helper handles one interior hold per clip. Conflicting positions for the first two injected images are rejected without changing the form; later movement checkpoints are preserved. After changing hold timing or clip duration, update the first two native frame positions to the new boundaries before using the helper again. Ordinary Apply reads existing injection positions; it does not move them. An endpoint hold should use the native Start/End Image controls.

Images constrain appearance and subject pose at their anchor frames as well as the camera view. They do not enforce a calibrated 3D angle, exact departure time, or roll lock. Saved camera-plan JSON does not embed media or injection settings; save the native WanGP generation settings/queue to keep those inputs together.

### Guide departure with a movement checkpoint

1. Apply the held-view image as above. Open **Movement image checkpoints**.
2. Supply a view that already shows the next camera movement. For the orbit/hold/rise example, use an elevated side view from a reviewed render, with the desired screen orientation and the Start Image's aspect ratio. To guide arrival, use a view still approaching the stop and place it shortly before the hold.
3. Set **Checkpoint time (seconds)** shortly after the hold. The tested half-second hold ends at 5.541667 seconds and uses a checkpoint at 6.541667 seconds.
4. Click **Add or replace movement checkpoint and apply**, then Generate normally.

The time is rounded to the nearest displayed frame. The checkpoint must lie strictly inside a moving segment, away from its camera keyframes and holds. The plugin adds its image through native Inject Frames, updates existing Picture references according to the host's actual image order, and asks the camera to pass through that view without stopping. The original keyframes and times remain unchanged. Ordinary Apply also recognizes manually configured images inside segments.

**Extract a checkpoint from an earlier render** avoids saving a PNG manually. Choose the video and a 1-based source frame, then click **Use this video frame as the checkpoint image**. Review the extracted view and set its time in the new clip before applying. The source frame's time and the new checkpoint time are separate controls; extraction does not alter the camera path or generation form.

### Correct the timing of an existing clip

**Automatically check and correct candidates** accepts one to three existing renders in preference order. It measures each source, corrects its timing, then independently measures the encoded output. Only a passing output is returned; otherwise it returns a rejection report and clears the video preview. No GPU generation or automatic rerender is launched.

Automatic acceptance requires at least 90% valid background tracking in each segment, movement on at least 60% of tracked pairs before and after the hold, a stationary hold of at least three frame intervals, correction speeds within 0.5x to 2x, and output hold boundaries matching the planned displayed frames exactly. These conservative thresholds can reject otherwise usable clips. Use the manual controls below for reviewed exceptions. A passing result certifies these motion/timing checks only: camera angles, viewpoint correctness, roll, scene consistency and audio quality still need visual/listening review. The JSON report records every attempted candidate and its rejection reason or measurements.

1. Load the matching camera plan. Open **Correct timing in a generated clip** and supply a video that already contains the wanted orbit, stationary hold and rise.
2. Choose the keyframe starting the planned hold. Click **Suggest arrival and departure frames**. Review the clip and adjust those 1-based source frame numbers if needed.
3. Choose audio handling. **Retime with video** applies the same section boundaries with pitch-preserving tempo changes. **Keep original audio timeline** is useful when ambience or a fixed music track should remain unchanged. **Remove audio** exports video only. Review dialogue and sharp sound events after retiming.
4. Choose **Nearest frame** to preserve individual source images, accepting repeated/dropped frames, or **Blend neighboring frames** to blend fractional positions, accepting possible ghosting.
5. Click **Export clip with corrected timing**. Download the new clip and its timing report. The source remains intact.

The selected source arrival/departure frames map exactly to the chosen hold boundaries. Boundaries round to displayed frames using the uploaded clip's actual frame count and FPS. The first/last frame, duration, frame count and dimensions remain fixed. All motion and other events inside each section are retimed together. This tool aligns one interior hold; it does not recover camera geometry or repair an incorrect viewpoint. Background-motion suggestions can fail with moving scenery, a large foreground subject or little texture, so the source marks remain editable.

Exports support constant-frame-rate single shots of 3–481 frames with even dimensions. They are written under WanGP's configured video output directory, or `local_runtime/h3_camera_exports` if that host global is unavailable. Filenames are unique. A separate `.timing.json` report records the source hash, exact boundary mapping, every sampled source position, audio handling and verification. Original generation metadata is not copied onto the edited video as though it were an unedited render; the report identifies its source.

Applying again at the same frame replaces that checkpoint image without adding another. A different time adds another checkpoint; use native injection positions to move an existing one, and review Picture numbering if images are reordered. The hold button retains added checkpoints when reapplied at the same hold boundaries. Image helpers reject unused Reference Images, incomplete injections and positions outside the one-shot timeline rather than silently activating or dropping those inputs.

### Stops and roll stabilization

A hold uses two identical camera poses at different times. Select a pose before the last keyframe and click **Add 0.5s hold** to duplicate it half a second later. Edit the new keyframe's time to adjust the pause. Existing keyframe times stay fixed, so the following move gets less time. If there is not enough room before the next keyframe, the editor asks you to adjust the timing first. Timing uses the same native frame alignment as Apply.

The prompt describes a hold as a stationary camera with fixed position, viewing direction and focal length, for its full duration. Subject and environment movement still follow the selected scene-motion setting. Smooth easing requests smooth acceleration/deceleration and a change of direction at each keyframe without rounding path corners. It does not ask the camera to stop at keyframes, so extra intermediate keyframes do not turn a move into a series of stops. Only timed holds pause the camera.

Holds shorter than 0.5 seconds show advice in the editor and Preview/Apply summary. This is a suggested starting point for a visible-stop test, not a measured H3 minimum. For example, the reported 50%–51% hold at 243 frames and 24 fps lasts about 0.101 seconds (2.42 frame intervals). In the local one-seed comparison, extending that hold to 0.5 or 1 second did not produce a reliable stop; see [VALIDATION.md](VALIDATION.md). All entered times are preserved; the plugin does not lengthen holds automatically.

**Stabilize camera roll** asks for a level horizon and an upright camera, using pan and tilt for framing. Paths reaching 75 degrees or higher also ask for a steady picture orientation near overhead. This checkbox adds prompt guidance; it is not a geometric camera lock. Turn it off if the scene deliberately calls for roll, and press Apply again after changing it.

### How moves are worded

Since 0.3.0, the prompt names only the motion you planned. Earlier versions also listed motions to avoid: "no orbit, crane, dolly, pan, tilt or roll", "without orbiting past", "twisting" and "rotation around the lens axis". They also mentioned full turns and overhead views on every path. H3 tended to perform the moves it read about. A 90-degree orbit with a 48-degree rise rendered as a full orbit, a top-down view and a spin.

Each move is now described as a turn fraction with its angle, such as "an eighth of a turn (45 degrees)". Its endpoint is named as a view (three-quarter, side or opposite side) and a camera height (eye level, high angle, near overhead). A whole-take line states the total orbit and the highest viewpoint, which bounds how far the camera should travel. Speed words (slowly, steadily, quickly) follow each segment's angular rate. These are wording choices that improved the tested renders; they do not impose exact camera geometry.

### Elevation moves

When elevation changes, the compiled segment separates movement of the camera from rotation of its lens and describes the endpoint view. High endpoints at 75 degrees or above receive near-overhead wording. Small distance changes of up to 5% within an elevation segment are described as slight percentage adjustments, retaining both distances. Larger distance changes remain explicit dolly instructions. These thresholds select wording; they are not model controls. Elevation at a fixed azimuth is described as rising or lowering straight up or down on the current view, including after a hold.

The diagram uses spherical distance and elevation around the subject. A large distance change can make camera height fall even while its elevation angle increases. The compiler accounts for this when describing camera travel. It keeps all keyframe values and does not rewrite the scene's subject actions, dialogue, gaze, or audio.

Use **Write/Enhance before Apply**. Disable automatic prompt enhancement when testing the compiled path, since enhancement can rewrite the instructions or reuse a stored original caption that has no camera plan. Apply again after using Write/Enhance. H3 Camera does not change the host's enhancer setting.

After updating, load your saved plan and press **Apply** again to obtain the new wording. Existing compiled prompts and queued jobs keep their old text. Improved wording still does not enforce exact camera angles, distances, timing, or subject behavior; review the generated video. See [VALIDATION.md](VALIDATION.md) for the actual render comparisons and their limits.

## Limits

- Camera coordinates become text instructions; this is approximate prompt guidance, not an enforced 3D camera track.
- The first pose is fixed at time 0, azimuth/elevation 0, distance 1. The last time is 1. Use 2–24 keyframes.
- The planner supports one shot/window, up to 481 frames. Dedicated H3 ControlNet models, multi-shot prompts, scheduler slash commands, video continuation, Control Video editing, still-image mode and audio-only models are rejected before applying.
- Audio sections and reference tokens in the scene prompt are retained. Frozen mode deliberately overrides subject/environment movement; audio stays governed by the scene prompt.
- The viewport uses local embedded assets and a sandboxed iframe, with no CDN or external service.

## Verification

Run these commands from the **plugin repository root**. The compiler tests require only Python:

```sh
python scripts/run_tests.py --suite plan
```

To run the complete suite, use WanGP's Python environment and point at its root:

```sh
python scripts/run_tests.py --host /path/to/Wan2GP
```

On Windows, select the environment you normally use for WanGP, for example:

```powershell
D:\Wan2GP\env_venv\Scripts\python.exe scripts\run_tests.py --host D:\Wan2GP
```

The tests load **this package**, even when another H3 Camera copy is already installed. `--suite integration` selects form, loader, and native-label tests. `--suite native` selects the model-handler, text-encoder and gallery tests; those require the complete WanGP Python dependencies. No test starts a generation. See [VALIDATION.md](VALIDATION.md) for the checks actually performed for this release and their limits.

Build a release with standard Python:

```sh
python scripts/build_release.py
```

This produces the installable ZIP, its SHA-256 checksum, and a per-file hash manifest under `dist/`. The archive excludes Git metadata, Python caches, and local environments.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Download or clone fails | Check the repository URL and your connection to GitHub. Git must be installed for Plugin Manager and command-line installation; the release ZIP is an alternative. |
| Plugin is missing from Plugins | Confirm the folder layout above; `plugin.py` must sit directly inside `plugins/wan2gp-h3-camera/`. |
| Plugin is enabled but the planner is hidden | Restart WanGP, select a supported H3 video model, and open Media Generator. This extension does not add a separate top-level tab. |
| A compatibility message lists missing controls or functions | Compare your host with the tested [WanGP revision](#compatibility). Do not copy host core files into the plugin. |
| Invalid JSON disables the editor | Correct the path JSON or choose a camera preset to reset the path. |
| Close loop reports reference numbering changes | Populate the native End Image first, update the scene's picture labels, and apply the loop again. |

For a reproducible issue, use this repository's [issue tracker](https://github.com/jazzi-valassis/wan2gp-h3-camera/issues). Include the plugin version, WanGP revision, model type, steps to reproduce, and relevant error text.

## Removal and saved plans

Disable the plugin, save, and restart to remove its interface and browser bridge. Uninstall removes the plugin folder. Plans you explicitly downloaded elsewhere remain yours. Save creates one temporary JSON export per browser session; the next successful save removes the previous source file, and Gradio's session cleanup removes the last one. Gradio manages its served download cache separately. Download a plan to retain it. Already compiled prompts and queued tasks run through WanGP normally without the planner.

## Author and attribution

Maintained by **Jazzi**, [@jazzi-valassis](https://github.com/jazzi-valassis). Copyright (c) 2026 Jazzi.

The workflow and normalized coordinate schema were inspired by [NyckM's MiniMax H3 camera editor](https://github.com/NyckM/3d-Camera-control-H3-Minimax). The Python compiler, Gradio integration and visual editor here are original implementations; upstream ComfyUI nodes and dependencies are not vendored. WanGP is developed separately by DeepBeepMeep.

See [NOTICE.md](NOTICE.md), [LICENSE](LICENSE), and [CHANGELOG.md](CHANGELOG.md).
