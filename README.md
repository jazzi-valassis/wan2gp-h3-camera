# H3 Camera for WanGP

A visual single-shot camera planner for MiniMax H3 in [WanGP / Wan2GP](https://github.com/deepbeepmeep/Wan2GP). Sketch an orbit, rise, push or pull, preview the path, then apply its camera instructions to the native generation form.

**Author and maintainer:** [Jazzi](https://github.com/jazzi-valassis)

**Version:** 0.2.2 · **License:** [MIT](LICENSE) · **Plugin type:** extension

The plugin adds no model downloads, GPU allocations or extra Python dependencies. Camera movement is prompt guidance: the diagram does not impose an exact 3D trajectory on the model.

## Features

- Eight camera presets, editable keyframes, numeric controls, and a scrub/play preview.
- Native frame-count and FPS timing, smooth or linear motion, and optional frozen-scene instructions.
- Preview without changing the generation form; Apply replaces the previous camera plan while retaining scene, audio and reference instructions.
- Portable JSON plans and optional closed-loop conditioning using the existing Start Image.

## Install

This is a standalone **extension** plugin. Its Python modules, browser assets and tests are all inside this directory. No files need to be copied into WanGP's `shared/`, `models/`, or `defaults/` directories.

### Clone the private repository

This repository is currently private. Use a GitHub account with access and an authenticated GitHub CLI on the machine where WanGP is installed. From the **WanGP root directory**, run:

```sh
gh repo clone jazzi-valassis/wan2gp-h3-camera plugins/wan2gp-h3-camera
```

An existing `plugins/wan2gp-h3-camera` directory must be backed up outside `plugins/` before cloning into that location. Do not keep a second enabled copy under another name.

In WanGP, open **Plugins**, enable **H3 Camera**, and save. Restart WanGP after existing work finishes. For supported H3 video models, **H3 Camera - single-shot planner** appears below the main prompt in **Media Generator**.

### Install from a ZIP

Download the prepared plugin ZIP from the [releases page](https://github.com/jazzi-valassis/wan2gp-h3-camera/releases) while signed in to an account with access, or build `wan2gp-h3-camera-0.2.2.zip` from this checkout with `python scripts/build_release.py`. Extract it into WanGP's `plugins` directory, then enable the plugin and restart as above. The prepared package includes a SHA-256 checksum and per-file manifest.

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
      web/
```

GitHub's **Code > Download ZIP** source archive uses a branch suffix such as `wan2gp-h3-camera-main`. Rename that extracted folder to `wan2gp-h3-camera` before placing it under `plugins/`. The prepared release ZIP already uses the correct name.

### Update an existing installation

For an existing installation, keep the folder name `wan2gp-h3-camera`. Back up the old folder outside `plugins/` or name the backup with a `.bak` suffix. Replace it with the new package. Keep your existing enabled-plugin setting and restart when convenient. Installing a second copy under another name creates conflicting editor IDs.

For a Git clone, fetch updates from the WanGP root with:

```sh
git -C plugins/wan2gp-h3-camera pull --ff-only
```

Review and preserve local edits before updating. Restart WanGP to load changed Python and browser assets. A ZIP installation has no Git remote; update it by replacing the plugin folder.

The repository root contains the plugin files directly. While it is private, use the authenticated clone or ZIP methods above; a browser login alone does not authenticate WanGP's GitHub-URL installer or its update checks.

## Compatibility

Release **0.2.2** targets **WanGP 13.141**, upstream revision `b8b18f8114e432eea8f3d7e853a51dd91fa99571`, with **Gradio 5.29.0**. Older releases are not certified by this package. Future host changes require repeating the compatibility checks.

The plugin uses `WAN2GPPlugin`, component/global requests, `insert_after`, and `add_custom_js`. Its remaining host imports are the installed `shared.utils.frame_scheduler.normalize_frame_count` and `shared.utils.prompt_parser.split_prompt_units`. These are WanGP dependencies, not additional files to ship. It does not patch the pipeline, launch another server, submit its own generation jobs, or import `h3cam_ref` or MiniMaxH3Mod.

Missing required form controls or injected host functions produce a compatibility message when WanGP can render the insertion point. The plugin also requests WanGP's native prompt/duration label helpers and prompt-tool attachment script to keep labels synchronized after Apply. The static version field is not a promise that every older or future host layout works.

## Use

1. Enable **H3 Camera** in the Plugin Manager and restart WanGP if it is not already enabled.
2. Select an H3 **FL2VA** or **Ref2VA** video model. Compatible H3 Multishot variants also work for one shot.
3. Write one scene in the main prompt, including any desired dialogue and sound. Set the native frame count and FPS.
4. Open **H3 Camera - single-shot planner** below the prompt. Choose a preset, edit keyframes and preview the camera move. Time is normalized across the selected clip, with the final pose at its last frame.
5. Choose whether the scene can animate (default) or is frozen, and smooth or linear camera movement. **Preview camera prompt** changes no generation settings.
6. Click **Apply camera path to generation form**, then use the normal **Generate** or **Add to Queue** button. Apply again after editing the path, scene, or duration.

Select a keyframe in the strip, then drag the diagram horizontally to orbit or vertically to change elevation. The first drag direction locks that axis until release. Focus the diagram and use the mouse wheel to change distance, or use its arrow keys (hold Shift for larger steps). Numeric controls provide precise edits. **Add** inserts a keyframe between the selected pose and the next pose; the first pose stays fixed. Scrub or play to preview the planned move.

Applying updates the prompt, makes all lines part of one prompt, aligns the frame count to H3's native `17k+5` grid, selects one sliding window, and sets both trim controls to zero. Existing reference media, audio, LoRAs, model selection, inference settings and memory settings remain in the normal form. Applying again replaces the prior generated camera instructions rather than adding duplicates.

For a **closed loop**, choose a path ending at its original camera pose (for example Full orbit), activate the native **Start Image** and provide exactly one image. Enable the loop checkbox before applying. The plugin copies that image into **End Image**, replacing its previous contents, and enables endpoint conditioning. With no loop requested, the plugin leaves the existing End Image alone. Matching endpoints does not guarantee that H3 completes an orbit.

If reference images are active, adding an End Image changes their H3 picture numbers. The plugin stops this operation with an explanation: first enable and fill the native End Image, then update the scene's picture labels (Start Image is Picture 1, End Image is Picture 2, other images follow). An already active, populated End Image keeps its existing position when replaced.

Use **Camera path JSON and saved plans** to save or load portable `.json` plans. Saved plans include keyframes, scene-motion choice, interpolation and loop choice. They do not overwrite the main scene prompt or duration. Bare keyframe arrays from the original editor are supported when they satisfy the normalized schema. The ordinary WanGP queue stores the compiled prompt and generation settings, so queued jobs do not need the planner to run.

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
| Repository is not found or cloning fails | Confirm that GitHub CLI is signed in to an account with access to this private repository. |
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
