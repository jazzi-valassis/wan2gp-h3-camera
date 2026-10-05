# H3 Camera validation

## 0.2.4 orbit-direction fix

Reported path: orbit 90 degrees, then crane to a near-overhead view. Keyframe 3 kept its absolute orbit angle of 90 degrees. The editor's **Rotation** label was read as a per-segment amount. Setting it to 0 sent the camera back to the start side, and the prompt called that move "continuing 90 degrees toward camera left".

Checked on 2026-10-05 on Windows against WanGP 13.141 (revision `b8b18f8114e432eea8f3d7e853a51dd91fa99571`) with Gradio 5.29.0:

- **63 automated tests passed** with `scripts/run_tests.py --suite all`, including nine orbit-direction compiler tests. The 34 plan-suite tests also pass under standalone Python 3.13.
- A differential check compiled 20,000 random 2- to 5-keyframe paths with both 0.2.3 and 0.2.4. Output changed only for segments that stop, reverse, keep the orbit angle during an elevation change, or orbit into a held angle. The render-validated three-keyframe path below is byte-identical.
- Headless Chrome drove the editor in the live WanGP page through the reported workflow: add a middle keyframe, set it to 90 degrees, then raise the last keyframe. The last keyframe showed turn 0 with "the orbit stops", and a value of 0 showed a reversal. Editing a turn shifted later keyframes, and an out-of-range turn was rejected with no published change. Preview and Apply wrote the stop wording into the native prompt. No page errors came from the plugin.

### Rendered orbit-then-crane comparison

Each render applied the reported path (orbit 90 degrees by 4.6875 s, then crane to 89 degrees by 9.375 s) through the live Apply button. It used Multishot Ref2VA Singularity v1.3 Pruned, 608 x 832, 226 frames at 24 fps, 12 steps, and no reference media. The scene was a hiker standing on a salt flat. Within a seed, only the camera-plan lines differ:

- **A:** the first 0.2.4 stop sentence, "stops orbiting sideways and cranes upward through a rising arc around the main subject".
- **B:** A's second segment replaced with the current no-circling sentence.
- **C:** B plus the current rest point at the end of the first orbit. C is the shipped wording.

| Seed | A | B | C |
| --- | --- | --- | --- |
| 764485034 | Side view by 3.5 s, then a full back view by 6 s before rising | Side view held until about 5.3 s; ends about three-quarters behind | Side view reached near 4.7 s and held until about 5.9 s; ends about three-quarters behind |
| 708144286 | Orbit continues to about 180 degrees by 5.3 s, then that angle holds while rising | Same as A | Same as A |

Every clip rose to a near-overhead endpoint. The stronger wording reduced drift during the crane for one seed. No wording prevented the second seed from overshooting the first 90-degree orbit. Orientation was judged visually from subject-centred crops at fixed times. With a plain background, camera orbit and subject turning cannot be fully separated. This is two seeds and one scene, not a reliability estimate.

## 0.2.3 record

Version 0.2.3 was checked on 2026-10-05 against official WanGP 13.141, revision `b8b18f8114e432eea8f3d7e853a51dd91fa99571`, using Python 3.11.11 and Gradio 5.29.0. Compiler and interface checks ran on macOS arm64; the rendered comparisons used the connected Windows WanGP installation. This record distinguishes prompt correctness from observed model behavior.

The elevation report showed that a numerical camera plan could become a frontal approach instead of a climb. Version 0.2.3 describes physical camera movement, separate lens tilt, endpoint framing, and proportional distance adjustments. It preserves the keyframes and scene text. The 75-degree and 5% thresholds select prose; they are not model controls.

## Automated checks

```sh
python scripts/run_tests.py --host /path/to/Wan2GP --suite all
python scripts/run_tests.py --host /path/to/Wan2GP --suite integration
python scripts/run_tests.py --suite plan
```

All **54 tests passed**: 25 compiler tests, 20 form/lifecycle tests, two native-label helper tests, and seven native model/prompt/gallery tests. The 25 compiler tests also passed independently using standard Python. The nine new elevation tests pass on the candidate; the final negative control against 0.2.2 produces five intended failures and four passing preservation checks, with no errors.

New coverage checks physical travel versus lens rotation, the reported 81-degree endpoint, small and larger distance adjustments, spherical height when radius changes reverse angular travel, downward views, unchanged path values and times, scene/audio/reference preservation, legacy processed prompts, and comment-stripped new-prompt replacement. Non-elevation orbit, push, pull and hold prompts match the 0.2.2 baseline byte-for-byte.

These tests use the actual pinned plugin loader, scheduler, prompt parser, model handler, text-presentation method, and Gradio gallery processing. Form unit tests supply model/FPS metadata to isolate their behavior. Native-label tests execute the selected unmodified host helper functions without starting a second application. No model weights are initialized by the test suite.

The package checker passes for all 12 Python files, with no errors or warnings. The native form integration is unchanged apart from the version string. Browser assets, dependency requirements, and the saved-plan schema are unchanged. The plugin adds no Python dependencies or host patches.

## Current browser checks

All **six focused browser checks passed** in a separate WanGP 13.141 installation. The actual Preview and Apply controls used the reported three-keyframe path at 124 frames and 24 fps. The checks verified the new elevation wording, exact rows and coordinates, scene/gaze/audio text preservation, repeated Apply, comment-free plan replacement, save/load, and failure atomicity. The invalid-input error was confirmed in the backend log; a toast was not captured.

The test app had exactly one H3 Camera panel and no browser page errors. Its checkpoint and output directories remained empty. The browser and owned test app were closed afterward. Camera compiler and plugin code hashes matched the candidate used for the rendering comparisons.

## Rendered camera comparisons

The supplied cat-on-stool comparison used Original Ref2VA 33B, seed 764485034, eight steps, and enhancement disabled. The original numeric camera wording stayed largely frontal. A manually rewritten second segment reached a steep downward view but lost eye contact. That pair motivated this compiler change; it did not establish which phrase caused the improvement.

The current pilot rendered the exact output of the modified compiler using the same settings and scene text. It reaches a clear top-down endpoint. The cat's gaze and posture still depart from the scene request. No cat- or stool-specific wording is hardcoded into the compiler.

The subsequent matched comparisons also recover the top-down endpoint:

| Model selection | Seed | 0.2.2 wording | 0.2.3 compiler output |
| --- | --- | --- | --- |
| Original Ref2VA 33B | 764485034 | Earlier verified A clip stays mostly frontal | Clear top-down endpoint |
| Original Ref2VA 33B | 708144286 | Fresh baseline stays frontal and grows closer | Clear top-down endpoint |
| Pruned Ref2VA 20B | 764485034 | Fresh baseline stays mostly frontal | Clear top-down endpoint |

Five new videos completed with no renderer errors: one compiler-output pilot and two fresh old/new pairs. All three compiler-output clips reach the top-down endpoint in these observations. The manually rewritten B clip is supporting background, not one of those three compiler-output results. This is a small check of two related model variants and two seeds, not a reliability estimate for all users.

Every new clip uses eight steps, 704 x 1280, 124 frames at 24 fps, one guidance phase, Euler, flow shift 12, and `int8,int8_convrot`. There are no reference media, separate LoRAs, selected step-skipping cache, or prompt enhancement. Within each fresh pair, only the prompt and output filename differ. All five videos decode correctly, and saved metadata matches the requested settings and camera instructions. The new prompts lose only their two plugin-owned comment markers in metadata. Model metadata identifies the original and pruned-rank8 INT8 ConvRot checkpoints respectively. The two variants' compiler-output videos have different decoded-frame hashes.

Visual inspection used frames 0, 31, 62, 93 and 123. The last frame is at 5.125 seconds; each clip lasts 5.166667 seconds. No path coordinates were adjusted between the old and new wording. The comparison uses coordinates reconstructed from the supplied prompt's printed values, not an original full-precision editor export.

## Earlier interface baseline

The 0.2.2 release passed 22 full-application browser checks, eight editor presets, and four fresh-process lifecycle modes: enabled, disabled, safe mode, and removed. Those results remain historical coverage of the unchanged integration and browser assets; they are not additional 0.2.3 reruns. The earlier guideline audit covered package layout, metadata, host requests, form ownership, session isolation, input preservation, browser sandboxing, exports, installation/update/removal, and attribution. Its separate test installation had all 2,764 upstream source files unchanged.

## Scope of the result

These checks verify the tested plugin and host combination. They do not guarantee every future host version, plugin combination, checkpoint, seed, or camera move. The DaSiWa Hybrid V3 checkpoint from the supplied export is not available on the connected renderer and is not certified by these comparisons.

Visual review checks changes in viewpoint and the top-down endpoint. It does not measure an exact 81-degree angle, orbit distance, fixed focal length, or timing accuracy. Subject gaze, pose, and framing can still vary. The enhancer-history interaction is documented in README; no global enhancer setting is changed. The normal application is not restarted by this update. Reloading the plugin and applying a saved plan again is necessary to use the new wording; existing queued prompts retain their previous text.
