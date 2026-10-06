# H3 Camera validation

## Local 0.4.0 generated view anchors - 2026-10-06

After 0.3.0, a sixth render of the reported path (seed 144618832, live UI) still failed: it reached the subject's back by 5 s and ended on the far side profile (about 270 degrees). The failure reproduced frame-for-frame on the MCP worker, so later variants were compared on that seed. All renders used the user's Singularity preset as in 0.3.0 (Ref2VA Singularity v1.3 Pruned 20B, 12 steps, one phase, Euler, shift 6, INT8 ConvRot, First Block Cache 0.08 from 25%), 243 frames at 24 fps, 704 x 1280, with the same start/reference image and scene text.

Two text-only variants were rejected:

| Variant | Seed 144618832 result |
| --- | --- |
| Views named by side ("seen from the right side of the opening frame"), a visibility bound and halfway milestones | Stopped at about 90 degrees and held, but orbited left: H3 read "right side" as the subject's right side |
| Same bound and milestones without side names | Still passed behind the subject and ended on the far side |

Words do not identify which side an orbit should end on. 0.4.0 therefore generates a still of each target view from the Start Image and injects it as an anchor. Views come from Qwen Image Edit Plus (2511) with fal's Multiple-Angles LoRA (`<sks> [azimuth] [elevation] [distance]`; its "right" matches the plugin's positive orbit) and the 8-step Lightning LoRA, CFG 1. Without Lightning the angle LoRA produced grainy, under-denoised views at 30 steps and at strength 0.6.

| Anchored render (hold view at frames 122/134, side view at 243) | Seed | Result |
| --- | --- | --- |
| Reported path, scripted | 144618832 | Arcs right and up to the high three-quarter hold, holds, ends on the correct side profile |
| Reported path, scripted | 764485034, 708144286, 315046792, 20261006 | Same path; 708144286 stays at the hold view until about 7.5 s, then descends late |
| Reported path, live UI button then Generate | 144618832 | Views in 42 s, apply, 345 s render; same correct path as the scripted run |
| Keyframe every second, anchored only at the hold and end | 708144286 | Correct path with steadier pacing: gradual rise to the hold and gradual descent to the end |
| Reported path with a close-up end view (0.5x distance) | 144618832 | Same correct path; the end is pushed in like the planned 0.5x, so 0.5x now maps to close-up |

Frame-difference checks found no visible pop at the injected frames: changes around frames 120-123 and 239-242 are below the clip median, and the hold frames are near-static (mean difference 0.04). Anchored stills also fix the subject's pose and background at those frames. Exact angles are approximate: the LoRA has 45-degree orbit steps and four heights, so the 48-degree hold uses its 60-degree "high-angle" view.

The live test found a browser bug in the first implementation: progress messages sent gr.update() placeholders and the final message raw values, and Gradio's generator diffs then reached the Dataframe as a patch the browser could not apply (`Cannot read properties of undefined (reading 'length')`). The form fields updated but the status and views gallery did not. All streamed outputs are now update dicts; the next live run updated every output with no page errors. WanGP also suspends plugin-submitted jobs until a Media Generator tab has browser focus, so the views render only while that tab is focused.

All 126 automated tests passed from source and from the extracted 0.4.0 ZIP, including ten view-anchor tests: pose-to-view mapping and tolerances, wrapping, start-view and shared-view rules, the keyframe-every-second path, resolutions, task settings, the queue-backed apply with Picture renumbering, failure handling and streamed output types.

Evidence is in `local_runtime/deepy_projects/h3_camera_spin_20261006/` (`anchors/`, `eval/`, `anchor_settings.py`, the live-UI scripts under `live/`) and the renders `D:/outputs/h3cam_spin_{ANC,ANCdense,ANCclose,V2,V3}_s*_1006.mp4`. This is one scene and five seeds.

## Local 0.3.0 spin fix - 2026-10-06

Report: a "simple" path rendered as a spinning shot. The path was a 45-degree orbit right while rising to elevation 48 and dollying to 0.75x by 5.041667 s, a 0.5 s hold, then another 45 degrees right while descending to eye level at 0.5x by 10.083333 s. The supplied clip rose to a top-down view, spun about the vertical axis and finished on the front view. Adding a keyframe every second did not help.

The 0.2.9 plan for this path named many moves that were not requested. These included full turns, overhead, twisting, rotation around the lens axis, banking, "no orbit, crane, dolly, pan, tilt or roll", "without orbiting past" and seven orbit/around-the-subject phrases. It described the 48-degree view as "above the subject", and its smooth easing asked the camera to stop at every keyframe. Version 0.3.0 names only the planned motion. Each segment gets a turn fraction with its angle, an endpoint view and camera height, and a speed word, and one line bounds the whole take's orbit and highest viewpoint.

Five renders used the user's saved Singularity preset: MiniMax H3 Ref2VA Singularity v1.3 Pruned 20B, 12 steps, one phase, Euler, shift 6, INT8 ConvRot, First Block Cache 0.08 from 25%, prompt enhancer off. Each is 243 frames at 24 fps, 704 x 1280. Frame zero of the supplied clip served as Start Image and identity reference, with the same rooftop scene text. Only the compiled camera plan and seed differ.

| Trial | Seed | Observed result |
| --- | --- | --- |
| 0.2.9 wording | 764485034 | Side view by 2.5 s, behind the subject by 5 s, top-down spin 5.5–8 s, back to the front view at the end (about 360 degrees) |
| 0.3.0 wording | 764485034 | Three-quarter high angle at 5 s, visibly held to 5.5 s, side profile at the end; descent stops short of eye level |
| 0.2.9 wording | 708144286 | Steep near-overhead by 5 s, spins while overhead, returns to the front view (about 360 degrees) |
| 0.3.0 wording | 708144286 | High angle three-quarter view held through the hold, side profile near eye level and closer at the end |
| 0.3.0, keyframe every second | 764485034 | One continuous arc to a high three-quarter view, then a side profile at eye level; no stops between keyframes |

No 0.3.0 render reached overhead or spun. Viewpoints were judged visually from contact sheets and frames 0, 60, 121, 133, 180 and 242, not measured as camera coordinates. This is one scene and two seeds, so it does not establish reliability on other scenes. Exact angles, distances and timing are still not enforced.

All 116 automated tests passed. Six new regressions in `test_h3_camera_spin.py` cover the reported path, one-keyframe-per-second paths, overhead wording thresholds, reversal and multi-turn totals, view names, and replacement of a comment-free 0.2.9 plan. Wording assertions and byte baselines in the orbit, elevation, motion, plan, plugin and round-trip tests were updated to the 0.3.0 wording; their structural checks are unchanged.

Evidence is in `local_runtime/deepy_projects/h3_camera_spin_20261006/` under the WanGP root: compiled prompts and settings, contact sheets, key-frame strips, `make_settings.py` and `dense_path.json`. Renders are `D:/outputs/h3cam_spin_{A029,B030}_s{764485034,708144286}_1006.mp4` and `h3cam_spin_B030dense_s764485034_1006.mp4`; the labelled comparison is `D:/outputs/h3_camera_030_spin_fix_comparison_20261006.mp4`.

## Local 0.2.9 automatic timing acceptance - 2026-10-06

Added a bounded workflow for one to three already rendered candidates. It checks measured source motion, retimes into a private temporary directory, remeasures the encoded output and publishes only an output passing the timing checks. Failed attempts are recorded and the next supplied candidate is tried; it never launches a new GPU generation. All-failed and invalid-input UI responses clear the previous video so an earlier success cannot be mistaken for the current result.

Acceptance requires at least 90% tracking coverage in each segment, movement on at least 60% of tracked pre/post-hold pairs, a stationary source hold of at least three frame intervals, section speed changes between 0.5x and 2x, and exact measured output hold boundaries. Tracking and stationarity use the 0.2.8 method below. These thresholds are conservative heuristics, not confidence probabilities or calibrated camera-pose estimates. A passing report explicitly marks camera geometry and roll as unverified. Visual and audio review remain necessary.

110 automated tests passed. Seven new regressions exercise static/untracked/moving-hold rejection, incomplete/out-of-order evidence, extreme correction rejection, post-export rejection followed by successful fallback, all-candidate failure with no video, invalid-input clearing and the real Gradio API signature. Previous media/audio/source-preservation tests remain passing.

Real-media validation uses existing renders without changing the user's generation preset. The 0.2.5 text-only half-second-hold clip is rejected for lacking a reliable stationary interval. The following 0.2.7 selected candidate passes after correction. The two 0.2.8 approach-96 renders (seeds 764485034 and 708144286) are checked separately. Evidence and per-candidate reports are in `local_runtime/deepy_projects/h3_camera_verified_20261006/`. This is validation on the same robot scene, not a broad reliability benchmark or a near-guarantee across scenes.

## Local 0.2.8 approach checkpoints and timing correction - 2026-10-06

The remaining 0.2.7 problem was early arrival at the held side view. This update adds extraction of a checkpoint from an earlier render and optional correction of the timing in an already generated clip. Timing correction maps reviewed source arrival/departure frames to the selected hold boundaries, preserving the first and last frames, duration, FPS and dimensions. It exports a separate MP4 and a JSON frame-mapping report. It does not change the source video, generation form or camera path.

Three new GPU renders used the same saved Singularity preset, scene, first image, hold/rise images and four camera poses as 0.2.7. They retain 12 steps, one guidance phase, Euler, shift 6, INT8 ConvRot and First Block Cache 0.08 from 25%. The native output is 243 frames at 24 fps, 1280 x 704, lasting 10.125 seconds. The planned hold occupies zero-based frames 121 through 133 (5.041667 to 5.541667 seconds; 1-based frames 122 through 134).

Two reviewed views extracted from zero-based frames 96 and 108 of the earlier anchored render were tried as approach checkpoints at frame 109 (4.541667 seconds) of the new clip. Both are still approaching the held composition. The frame-96 variant was also rendered with a second seed. Native gallery positions are `122 134 158 110`; H3 Picture labels follow chronological native packing, and the original identity reference remains present. No geometric keyframe was added or moved.

| Source clip | Seed | Detected arrival frame | Detected departure frame |
| --- | --- | --- | --- |
| Existing 0.2.7 selected clip | 764485034 | 113 | 135 |
| Frame-96 approach image | 764485034 | 121 | 137 |
| Frame-108 approach image | 764485034 | 119 | 137 |
| Frame-96 approach image | 708144286 | 120 | 138 |

All table frames are zero-based. Detection measures the contiguous stationary background interval overlapping the planned hold, with median tracked displacement no greater than 0.5 original-image pixels per frame pair. Images are resized to at most 640 pixels wide for tracking, using the outer 32% on each side, excluding a scaled 30-pixel border. Forward/backward consistency and at least 12 tracked features are required. This is an editable timing suggestion, not recovered camera geometry. A moving background or large subject can mislead it; the UI asks the user to review the views.

The approach checkpoint improves arrival in these trials, but still does not guarantee generation timing. The existing 0.2.7 clip and both frame-96 clips were therefore exported through the new correction callback, using nearest-frame sampling and audio retimed with pitch preservation. All three corrected clips independently measured the intended stationary interval at frames 121 through 133. The existing clip required 10 repeated source selections; the two approach clips required 4 and 6. Their JSON reports preserve every inverse source-frame mapping and source hashes. The selected deliverable corrects the existing 0.2.7 clip, retaining its scene appearance without requiring another image checkpoint or render.

Exactness applies to the selected frame boundaries. Retiming affects subject motion and, by default, the soundtrack too. Nearest sampling can repeat/drop frames; optional blending can ghost moving details. Preserving the original audio timing or muting it are also available. The clip must already contain the wanted views and a hold. These experiments do not establish exact 90-degree poses, roll accuracy, constant camera speed, text-only reliability or general reliability across scenes.

Validation used the installed WanGP v17.01 at host HEAD `0e58385fbde7ff102d276e4a9e490845de76b4ea`, with pre-existing host edits, Python 3.11.9 and Gradio 5.29.0:

- **103 automated tests passed**, with Python exit code zero. The 11 additional tests cover exact rational maps, invalid inputs, real CFR/irregular timestamp detection, frame extraction, actual encoded boundary-frame identity, all three audio modes, pitch/duration preservation, source preservation and registered Gradio API shapes.
- All three new GPU clips passed full video/audio decoding and submitted-versus-recovered settings/prompt checks. All three corrected clips passed full decoding, output geometry/frame-count checks and independent background-motion measurement. Generation continues to use the user's saved preset.
- The actual candidate Gradio UI loaded through the host plugin manager extracted frame 97 (1-based) at 4 seconds into a 1280 x 704 checkpoint image. The inspection button suggested frames 122 and 138 for the approach-96 clip; the export button returned a playable MP4 and downloadable report placing the boundaries at 122 and 134. The main prompt stayed unchanged, no alerts appeared and there was no horizontal overflow at 1280 x 800. T3 snapshot capture was unavailable; button interaction and DOM inspection were used. Separate real Gradio-client calls verified serialization for all three new APIs.

Evidence is under `local_runtime/deepy_projects/h3_camera_arrival_trials_20261006/`, including settings, image hashes, job records, contacts, full per-pair motion data, test results and API/browser verification. The selected corrected clip is `D:/outputs/h3cam_framed_s764485034_1006_timed_3205e23fb8.mp4`. The comparison `D:/outputs/h3_camera_028_timing_comparison_20261006.mp4` shows its original and corrected versions side by side, with audio from the corrected version. The configured approach example is `D:/outputs/h3_camera_028_example.json`; its editable path is `D:/outputs/h3_camera_028_camera_plan.json`. Media evidence and configured examples are local validation artifacts, not included in this repository.

## Local 0.2.7 movement checkpoints - 2026-10-06

The remaining 0.2.6 problem was late departure from the held side view. The new control adds a native image during the moving segment, and the compiler now describes interior images as timed views to pass through without stopping. Previously it described only images at camera keyframes. The four camera poses and their times are unchanged.

Five new GPU renders used MiniMax H3 Ref2VA Singularity v1.3 Pruned 20B, the user's saved preset, the same workshop robot scene and the source first frame. Generation remains 12 steps, one phase, Euler, shift 6, INT8 ConvRot and First Block Cache 0.08 from 25%. Native alignment produces 243 frames at 24 fps, 1280 x 704, lasting 10.125 seconds.

For the 0.5-second hold, the two existing image anchors remain at 1-based frames 122 and 134. A third image at frame 158 (6.541667 seconds) shows the camera already rising. This is one second after the nominal hold end. The original identity image remains in the gallery and becomes Picture 5. The helper updates native Picture numbering; it does not change geometric keyframes or add another pause.

The first three new renders used frame 190 from the earlier anchored baseline. Their departure timing improved at both seeds and with the original 0.100833-second hold, but that source image had too little headroom. The selected example instead uses baseline frame 184 (7.666667 seconds in its source video), which keeps the head inside the frame. Both frame-184 trials visibly retain the side-view hold and then rise toward an overhead view. These selected images are reviewed compositions, not calibrated camera poses.

| Trial | Seed | Rounded hold end (s) | Detected sustained motion (s) | Hold displacement (px/frame) |
| --- | --- | --- | --- | --- |
| 0.2.6: hold anchors only | 764485034 | 5.541667 | 6.625000 | 0.0106 |
| 0.2.7: frame 190 checkpoint | 764485034 | 5.541667 | 5.541667 | 0.0139 |
| 0.2.7: frame 190 checkpoint | 708144286 | 5.541667 | 5.625000 | 0.0068 |
| 0.2.7: frame 190, original tiny hold | 764485034 | 5.125000 | 5.125000 | 0.0280 |
| 0.2.7: frame 184 checkpoint, selected example | 764485034 | 5.541667 | 5.625000 | 0.0096 |
| 0.2.7: frame 184 checkpoint | 708144286 | 5.541667 | 5.625000 | 0.0141 |

Motion onset is the first frame pair at or after the rounded hold end followed by two more pairs whose median tracked background displacement exceeds 0.5 pixels. Tracking uses the outer 32% of the frame on both sides, excluding a 30-pixel border, with forward/backward consistency checking. This is a background-motion proxy, not a recovered 3D camera path. The original tiny hold ends at 5.142500 seconds in the prompt; its image anchor rounds to 5.125000 seconds. No path time is rewritten.

The delayed rise is corrected in these tested image-conditioned workflows. Global timing remains approximate: the orbit can reach its side view before the requested hold starts. The sampled output views show no obvious bank before the overhead transition, but roll stabilization is enabled throughout and its separate causal effect is not established. One scene, two seeds and selected images do not establish reliability on other scenes. Text-only Apply still cannot guarantee the same result; installing the update does not automatically add a checkpoint to old jobs.

Validation used the installed WanGP v17.01 at host HEAD `0e58385fbde7ff102d276e4a9e490845de76b4ea`, with pre-existing local host edits, Python 3.11.9 and Gradio 5.29.0:

- **92 automated tests passed** with a verified Python exit code of zero. The eight additional regressions cover interior moving/held views, unchanged paths, native reference ordering, duplicate replacement, hold preservation, incomplete timelines, invalid times and invalid FPS.
- The actual candidate Gradio UI produced the exact prompt used for the render, positions `122 134 158`, native mode `FI` and four reference-gallery images. Repeating the checkpoint action preserved that state. No horizontal overflow occurred at 480 x 850. Browser snapshot capture was unavailable; interaction and DOM inspection were used.
- All five new clips passed complete video/audio decoding. Recovered prompts match after native comment removal, and exported generation settings match submitted values. Input-image paths and SHA-256 hashes are stored separately because video settings do not export every attachment.

Evidence, submitted/recovered settings, camera plans, optical-flow measurements, contacts, job results and verification scripts are saved under `local_runtime/deepy_projects/h3_camera_departure_trials_20261006/`. The reviewed comparison is `D:/outputs/h3_camera_027_final_comparison_20261006.mp4` (1920 x 582, 243 frames at 24 fps); it uses the old clip's soundtrack, while both source clips retain their own audio. `D:/outputs/h3_camera_027_example.json` contains the selected generation settings and local images; `h3_camera_027_camera_plan.json` contains its editable camera path. Media evidence and configured examples are local validation artifacts, not included in this repository.

## Local 0.2.6 native image anchors - 2026-10-06

The text-only 0.2.5 renders below still overshoot and glide through holds. Two additional wording trials (explicit visual endpoints and a shorter three-phase description) also failed to stop the camera at the intended side view. Those prompt variants are not shipped.

The new button uses native Ref2VA image conditioning. The held-view image is frame 90 (3.75 seconds) from the earlier 0.5-second baseline, reviewed as a side profile. It is not a measured or reconstructed 90-degree camera pose. The same 90-degree orbit, fixed-azimuth rise, source first frame, scene, and saved Singularity preset remain in use. With a 0.5-second hold, the image is injected at 1-based frames 122 and 134 (5.041667 and 5.541667 seconds); the original identity reference follows it as Picture 4. The shorter reported hold keeps its original normalized times and injects at the nearest displayed frames, 122 and 124. Every render uses 243 frames at 24 fps, 1280 x 704, lasting 10.125 seconds.

| Trial | Seed | Observed result | Median background displacement during hold |
| --- | --- | --- | --- |
| 0.2.5 text only, 0.5-second hold | 764485034 | Continues toward the back view | 32.597 px/frame |
| 0.2.6 two hold images, 0.5-second hold | 764485034 | Reaches side view, visibly holds, then rises overhead | 0.011 px/frame |
| 0.2.6 two hold images, 0.5-second hold | 708144286 | Also reaches side view and holds before rising | 0.007 px/frame |
| 0.2.6 two hold images, 0.100833-second hold | 764485034 | Reaches side view and pauses; the visible dwell exceeds the requested interval | 0.072 px/frame |

The motion metric uses forward/backward-checked feature tracking in the outer 32% of each frame, excluding the center subject and frame border. It is evidence of reduced background movement, not a recovered camera trajectory or proof of an exact 3D lock. The half-second trials use frame pairs 121 through 133; the short trial uses 121 through 123 after rounding. These tests use one scene, one selected anchor image and two seeds; they do not establish general reliability.

The remaining limitation is departure timing: the camera can stay at the side view beyond the nominal hold and begin most of its rise late in the clip. No automatic retiming or smaller orbit angle is applied. The inspected outputs show no obvious bank before the overhead transition, but the roll checkbox is enabled in every run; its causal effect is not isolated.

A sixth trial added the first anchored clip's overhead endpoint as a native End Image (seed 764485034, same half-second hold). It still held the side view until late in the clip, so adding an end image is not presented as a timing fix. The supplied end image is mapped correctly by the compiler, but it is optional and is not part of the recommended two-anchor example.

Validation used the installed WanGP v17.01, host HEAD `0e58385fbde7ff102d276e4a9e490845de76b4ea` with existing local host edits, Python 3.11.9 and Gradio 5.29.0:

- **84 automated tests passed**, including 11 new compiler/form tests. Coverage includes real native image ordering, frame rounding without path edits, sorted/duplicate injections, endpoint labels, retained references/audio, repeated Apply, missing host support and invalid-input rejection.
- Browser interaction with the actual candidate plugin verified that the new button produces the exact prompt used in the successful render, sets frames `122 134` and mode `FI`, retains the original reference, and remains unchanged on repeated Apply. At 480 x 850 the page had no horizontal overflow. Browser snapshots were unavailable; interaction and DOM inspection were used.
- Submitted and recovered prompt/settings values match after native removal of comment markers. Every completed test clip passed full video and audio decoding, with the expected dimensions, frame count and FPS. Media paths and input hashes are retained separately because embedded video metadata does not export all image inputs.

Evidence, settings, prompts, source images, job results, contact sheets, optical-flow measurements and verification scripts are in `local_runtime/deepy_projects/h3_camera_endpoint_trials_20261006/` under the WanGP root. The reviewed before/after comparison is `D:/outputs/h3_camera_026_before_after_20261006.mp4` (1920 x 582, 243 frames at 24 fps). It uses the baseline soundtrack; both original renders retain their own generated audio. Media evidence remains local and is not included in this repository.

## Local 0.2.5 stop and roll changes - 2026-10-05

The supplied `TEST 1.mp4` and `TEST 1-1.mp4` both contain 243 video frames at 24 fps, 1280 x 704, about 10.125 seconds. Inspection sampled frames 0, 48, 96, 120, 144, 168, 192, 216 and 242. The broad orbit-then-rise progression is visible, with a stronger late orientation change in the first clip. These are visual observations, not recovered camera coordinates. The scenes/compositions differ, and no common seed or full generation settings were supplied, so the absence of an obvious roll in the second clip cannot be attributed to the hold alone.

The second screenshot specifies a hold from 5.041667 to 5.142500 seconds: about 0.100833 seconds, or 2.42 frame intervals. Its ascent wording also changes. The compiler had forgotten the stopped orbit immediately after an equal-pose segment, reintroducing "around the main subject". The local changes correct that discontinuity, strengthen hold and boundary instructions, and offer roll stabilization separately.

Checks performed locally on Windows with Python 3.11.9 and Gradio 5.29.0 in the installed WanGP environment (host HEAD `0e58385fbde7ff102d276e4a9e490845de76b4ea`, with existing local host edits):

- **73 automated tests passed** across the plan (41), integration (25) and native (7) suites. The reported-hold regression was rerun after correcting the screenshot timestamp. Coverage includes legacy/comment-free prompt replacement, exact path/timing preservation, roll enable/disable, old/new saved plans, native prompt processing, image labels and gallery round trips.
- A separate Gradio fixture loaded the actual plugin via WanGP's plugin manager. Browser interaction verified Preview leaves the scene prompt unchanged, the short-hold advice is displayed, and Apply respects a disabled roll setting while retaining a single plan and stopped-orbit ascent.
- The actual editor HTML/CSS/JavaScript was also served standalone for focused browser checks. Inserting a hold duplicated the selected pose for 0.5 seconds, preserving existing times. A gap shorter than 0.5 seconds produced an error and no edit; the last keyframe disabled insertion. A 0.101-second hold displayed the short-pause advice. At a 480-pixel viewport the new controls fit without page overflow. Snapshot capture was unavailable, so these checks used DOM inspection and browser interaction.

The follow-up GPU comparison below tests the three hold durations with stabilization enabled. The 0.5-second advice remains an experimental starting point, not a measured temporal limit or a demonstrated fix. A matched roll-off/on comparison and a second seed are still needed before attributing a roll improvement to the checkbox or generalizing the motion result.

### 0.2.5 controlled hold-duration renders

The local WanGP UI was restarted after confirming its queue was empty. Its live form exposed the default-enabled roll checkbox and the new hold button. Three renders then completed through the connected MCP worker using MiniMax H3 Ref2VA Singularity v1.3 Pruned 20B, seed `764485034`, and the user's saved Singularity preset: 12 steps, one guidance phase, Euler, shift 6, INT8 ConvRot, and First Block Cache 0.08 from 25%. The preset requested 239 frames at `704x1280`; native camera Apply alignment produced 243 frames, and image aspect handling produced 1280 x 704 at 24 fps. Each video lasts 10.125 seconds, with its last frame at 10.083333 seconds.

All three use frame zero of the supplied `TEST 1-1.mp4` as both starting image and identity/scene reference, plus the same bronze-robot workshop scene text. No original generation settings could be recovered from the supplied clips, so this is a new matched comparison, not an exact reproduction of the original runs. The camera path requests a 90-degree orbit by 5.041667 seconds, a hold, then a fixed-azimuth rise to elevation 89 degrees. Stabilization is on in every run. Within the comparison, only the hold duration, corresponding ascent start, and output filename differ. The final time stays fixed, so longer holds leave less time for the rise.

| Requested hold | Requested ascent start | Observed motion |
| --- | --- | --- |
| 0.100833 seconds | 5.142500 seconds | Continues moving through the brief hold; progresses beyond a side view toward the back before rising. |
| 0.5 seconds | 5.541667 seconds | Changes from a rear-quarter view to a nearly full back view across the intended hold, without a stationary interval. |
| 1 second | 6.041667 seconds | Also continues toward a back view across the intended hold; extending the hold does not produce a reliable stop. |

All three reach a near-overhead endpoint. The inspected frames show no obvious bank or lens-axis twist, but all three include the same stabilization wording, so this does not isolate its effect. Viewpoints were judged visually, not measured as exact 3D camera coordinates. These renders still overshoot the requested quarter turn; the longer holds do not establish an improvement in stop fidelity for this seed.

All three outputs passed complete video/audio decoding. Recovered generation settings match the submitted values that the host exports, and the prompts match exactly after removal of plugin comment markers. The host updates its version label from the saved preset to WanGP v17.01; shared effective generation settings match across all three clips. Visual inspection used the full-path contact sheets plus matching transition frames at 4.5, 5.041667, 5.125, 5.541667, 6.041667 and 10.083333 seconds.

Local evidence is saved in `local_runtime/deepy_projects/h3_camera_hold_comparison_20261005/` under the WanGP root: the source preset, submitted and recovered settings, camera plans, prompts, job result, hashes, contact sheets, and `verification.json`. The three full-resolution outputs are `D:/outputs/h3_camera_025_hold_{0101s,0500s,1000s}_seed764485034_20261005.mp4`. The labeled comparison is `D:/outputs/h3_camera_025_hold_comparison_20261005.mp4` (1920 x 400, 243 frames, 24 fps). Its audio comes from the 0.100833-second clip. The MCP comparison exporter stalled and timed out; its owned process was stopped and a bounded local FFmpeg export completed and passed full decoding. Original renders were preserved.

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

Visual review checks changes in viewpoint and the top-down endpoint. It does not measure an exact 81-degree angle, orbit distance, fixed focal length, or timing accuracy. Subject gaze, pose, and framing can still vary. The enhancer-history interaction is documented in README; no global enhancer setting is changed. Installing the package does not automatically restart the application; the local 0.2.5 follow-up above explicitly restarted it. Reloading the plugin and applying a saved plan again is necessary to use the new wording; existing queued prompts retain their previous text.
