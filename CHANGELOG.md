# Changelog

## 0.3.0 - 2026-10-06

- Fix spinning renders: a 90-degree orbit with a 48-degree rise produced a full orbit, a top-down view and a spin. The compiled plan no longer lists motions to avoid (orbit, roll, twisting, rotation, banking), mentions full turns, or names an overhead view unless the path reaches 75 degrees. A 48-degree view is no longer described as "above the subject".
- Describe each move as a turn fraction with its angle, a named endpoint view and camera height, and a speed word derived from the segment's rate. A new whole-take line states the total orbit and highest viewpoint.
- Smooth easing no longer asks the camera to stop at every keyframe. Extra intermediate keyframes now continue one arc instead of adding stops. Only timed holds pause.
- The first orbit segment no longer says "continuing". Stopped orbits rise or lower "straight up/down" on the current view, and dolly-only segments state their distances once.
- Six new regressions cover the reported path, one-keyframe-per-second paths, overhead wording thresholds, reversal/multi-turn totals, view names and replacement of a comment-free 0.2.9 plan. Wording baselines were updated.
- Five renders on the reported scene: 0.2.9 wording spun overhead at both seeds; 0.3.0 did not at either seed or with a keyframe every second. See VALIDATION.md. Saved plans are unchanged; press Apply again to get the new wording.

## 0.2.9 - 2026-10-06

- Added bounded automatic checking/correction of up to three existing candidate renders, stopping at the first passing output.
- Added independent post-export hold measurement, background tracking/movement gates and automatic speed limits. Failed candidates return reasons; all-failed runs return no video. Invalid input clears the previous result.
- Reports explicitly leave camera geometry and roll unverified. This feature does not launch GPU retries.
- Seven new regression tests cover failed-source/output rejection, candidate fallback, unavailable tracking, static shots, correction limits and native API registration.

## 0.2.8 - 2026-10-06

- Add extraction of a selected video frame into the movement-checkpoint image control. Approach checkpoints can guide a camera that reaches its held view early.
- Add optional post-generation timing correction for one interior hold. Editable arrival/departure marks map to exact output frames while retaining clip duration, FPS, dimensions and endpoints.
- Suggest source marks using background motion, with explicit review guidance. Validate actual frame timestamps and reject unsupported variable-rate clips.
- Provide pitch-preserving audio retiming, unchanged audio timing or muted output, and nearest-frame or blended sampling. Source clips remain intact; unique exports include a detailed timing report.
- Add 11 tests covering exact frame mapping, real encoded boundary images, source preservation, native UI registration, extraction, irregular timestamps and audio modes.

## 0.2.7 - 2026-10-06

- Compile timed images inside camera segments as movement checkpoints. Their instructions now specify when to pass through the view without introducing a stop. Images inside stationary holds keep stationary wording.
- Add **Movement image checkpoints** with a view image, time in seconds, and an Apply button. Saved camera geometry, keyframe times, hold duration and clip duration are preserved.
- Merge native injected images while retaining existing holds, end images and references. Renumber Picture roles using the native host's actual image order, including earlier insertions and duplicate positions. Repeated application replaces the same checkpoint.
- Preserve later checkpoints when reapplying the hold helper, and reject inactive references or unsupported injection timelines before form changes.
- Validate native FPS before mapping image times, including rejecting zero FPS without a division error.
- Add eight regression tests for timed pass-through instructions, unchanged paths, native ordering, replacement, hold preservation and invalid inputs.
- Validate five new renders across two seeds and the original tiny hold. The selected checkpoint improves late departure while preserving the side-view hold; exact arrival timing remains approximate. See VALIDATION.md.

## 0.2.6 - 2026-10-06

- Add **Anchor hold and apply camera path** for H3 Ref2VA. A supplied held-view image conditions both hold boundaries through native Inject Frames; other references and their Picture labels are preserved.
- Link camera segments to actual native image labels, including sorted/deduplicated injections and an active end image. Repeat application does not accumulate images or prompt blocks.
- Preserve saved path times and round only injection positions to displayed frames. Reject conflicting injection timelines, missing media, unsupported models and ambiguous endpoint galleries without modifying the form.
- Keep ordinary planning available when only the optional timed-image helper is missing. Record the installed v17.01 host used for validation.
- Add compiler and native-form regression coverage. Render trials show that image anchors correct the tested orbit overshoot and hold drift at two seeds, while departure timing remains approximate. Additional text-only wording trials did not fix the motion and are not included.

## 0.2.5 - 2026-10-05

- Keep the stopped-orbit instruction through intervening holds and dollies. Fixed-azimuth elevation no longer reintroduces "around the main subject" after a pause.
- Describe stationary holds with their exact duration and pose, and ask preceding moves to stop before a hold. Smooth easing now separates speed easing from corner rounding, with no implied extra dwell.
- Add optional **Stabilize camera roll**, enabled by default, and preserve the choice in saved plans. Older plans load with stabilization on.
- Add **Add 0.5s hold** without shifting existing keyframe times, mark hold keyframes, and show advice for pauses shorter than 0.5 seconds. This advice is not a validated model threshold and does not rewrite timing.
- Use the same native frame alignment in the editor timing and Apply, so an inserted half-second hold retains its duration.
- Preserve scene actions while constraining the camera's azimuth; remove wording that could force an animated subject to keep the same side toward the camera.
- Add regressions for the reported 50%–51% hold, timing preservation, repeated application, roll settings, saved plans and native timing alignment.
- Record a three-render comparison of 0.101-, 0.5- and 1-second holds at one fixed seed. All reach overhead, but longer holds do not reliably stop the camera or prevent orbit overshoot. The roll option's effect is not isolated; see VALIDATION.md.

## 0.2.4 - 2026-10-05

- Fix misleading orbit wording: a segment turning against the previous orbit now says it reverses direction instead of "continuing", and a segment that keeps the previous orbit angle says the orbit stops.
- When the next segment holds an orbit angle, end the orbit with an explicit rest point ("come to rest a quarter turn from the starting view, without orbiting past azimuth 90 degrees"). The following elevation segment drops "around the main subject" and states that the camera does not circle the subject and that the subject keeps the same side toward the camera.
- Rename the editor's **Rotation** field to **Orbit angle (from start)** and add an editable **Segment turn** field, per-keyframe turns in the keyframe strip, and a live description of the selected segment. Editing a turn keeps later keyframes' turns.
- Elevation segments that keep their orbit angle now add that the camera does not orbit sideways.
- Label the preview table's orbit column as an angle from the start view. Saved plans and keyframe JSON are unchanged. Paths without stops, reversals, held-angle elevation segments or orbits followed by a held angle compile byte-for-byte as in 0.2.3, including the render-validated three-keyframe path.
- Add orbit-direction compiler regressions.

## 0.2.3 - 2026-10-05

- Describe elevation changes as physical camera movement, separate lens tilt, and an explicit endpoint view, including near-overhead framing for high endpoints.
- Describe small distance adjustments during elevation moves proportionally, preserving their values without an emphatic dolly instruction.
- Keep non-elevation camera wording, scene and audio text, keyframe values, native form integration, and legacy plan replacement intact.
- Add compiler regressions for elevation semantics, spherical height, distance thresholds, preservation, and repeat application. Document the correct order of enhancement and Apply.

## 0.2.2 - 2026-10-04

- Exclude dedicated H3 ControlNet models from the single-shot planner while retaining FL2VA, Ref2VA, and their supported variants.
- Refresh native prompt, endpoint, and duration labels after a successful Apply using WanGP's existing helpers.
- Add regressions for model support and native label synchronization.
- Document public installation through WanGP's Plugin Manager, Git, and the release ZIP.

## 0.2.1 - 2026-10-04

- Use Jazzi as the sole author and maintainer name in documentation, plugin metadata, and copyright notices.
- Repackage the download with the corrected credit. Planner behavior is unchanged.

## 0.2.0 - 2026-10-04

- Package H3 Camera as a standalone WanGP extension with local browser assets, portable tests, and a reproducible release ZIP.
- Register the browser bridge through WanGP's supported `add_custom_js` API.
- Retain the main generation-form binding across the host's main and edit setup passes.
- Show compatibility diagnostics for missing host controls and functions.
- Clean up the current browser session's temporary plan exports when replaced or when Gradio releases the session state.
- Document installation, updates, author details, supported host APIs, and validation limits.

Validation: 34 CPU tests and 11 Chrome checks against the WanGP 13.141 loader and Gradio 5.29.0. Four additional native-model tests are included but were not rerun. No inference or camera-motion quality evaluation was performed. See [VALIDATION.md](VALIDATION.md).

## 0.1.0 - 2026-09-11

Initial local planner: visual keyframes, camera presets, prompt preview and application, H3 frame alignment, optional closed-loop endpoints, and JSON plan import/export.
