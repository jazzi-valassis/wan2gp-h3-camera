# H3 Camera validation

Version 0.2.2 was checked on 2026-10-04 against official WanGP 13.141, revision `b8b18f8114e432eea8f3d7e853a51dd91fa99571`, using Python 3.11.11 and Gradio 5.29.0 on macOS arm64.

The final regression found two defects in 0.2.1: dedicated H3 ControlNet models passed the support check, and Apply could leave native prompt and duration labels stale. Both are corrected in 0.2.2. The new tests fail on the original implementation and pass with the fixes.

## Automated checks

```sh
python scripts/run_tests.py --host /path/to/Wan2GP --suite all
python scripts/run_tests.py --host /path/to/Wan2GP --suite integration
python scripts/run_tests.py --suite plan
```

All **45 tests passed**: 16 compiler tests, 20 form/lifecycle tests, two native-label helper tests, and seven native model/prompt/gallery tests. The focused integration suite also passed independently, with 22 tests. All 45 tests also passed from a freshly extracted installation ZIP.

Coverage includes strict path validation, H3 frame alignment, FPS timing, scene/audio/reference preservation, repeated Apply, unsupported modes, closed-loop endpoints, reference numbering, saved-plan round trips, independent session inputs, export cleanup, native label refresh, and success-only callback wiring.

These tests use the actual pinned plugin loader, scheduler, prompt parser, model handler, text-presentation method, and Gradio gallery processing. Form unit tests supply model/FPS metadata to isolate their behavior. Native-label tests execute the selected unmodified host helper functions without starting a second application. No model weights are initialized by the test suite.

Static package checks passed for 11 Python files with no errors or warnings. Both unchanged browser JavaScript files passed `node --check`. The isolated environment's 192 installed packages passed dependency consistency checking. The plugin itself adds no Python dependencies.

## Installation and lifecycle

Four fresh-process checks passed using the real plugin manager: enabled, disabled, safe mode, and removed. Metadata discovery did not import the plugin. Enabled loading registered the extension and one browser bridge; the other modes imported neither the plugin nor its JavaScript. The package keeps the required `wan2gp-h3-camera` root and its own local assets.

The guideline review covered root layout, extension metadata, relative imports, host requests, main/edit form ownership, session state, input preservation, browser sandbox/message checks, exports, installation, update, disable, removal, and attribution. No host source, model handler, or pipeline patch is shipped.

## Full application and browser checks

The normal WanGP application started successfully in a separate test installation with H3 Camera and Motion Designer enabled. All 2,764 upstream source files matched the downloaded official revision. Configuration, settings, output paths, and checkpoint paths were separate from the user's normal application, with no preload policy and no startup queue.

The test configuration initially omitted the host's standard `clear_file_list` setting, causing an unrelated native gallery refresh error. Restoring the upstream default of 5 resolved that test-setup issue without changing host or plugin source.

All **22 browser checks passed** against the corrected 0.2.2 package. They covered PG/W-to-FG prompt and endpoint labels, duration-label refresh, failure atomicity, preservation of unrelated settings and audio, Preview, repeated Apply, saved-plan download/upload, numeric edits, keyframe add/remove, pointer dragging, playback, invalid-JSON recovery, a real-image closed loop, two separate sessions, Motion Designer coexistence, layout, and empty page-error logs. Each session had exactly one H3 Camera panel. The unchanged editor's eight presets also passed against the baseline application.

The isolated browser's native model selectors offered only MiniMax H3 / Ref2VA 33B. Dedicated ControlNet selection was therefore verified through the actual host-backed model tests rather than browser selection. The browser retained the host's pre-existing `Too many arguments provided for the endpoint.` console warnings, also present before the candidate changes; no candidate page error occurred.

The test application uses WanGP's stock Apple/MPS support. Nonfatal MPS autocast and PyAV/OpenCV AVFoundation warnings occurred during startup; no plugin-load failure occurred. Optional Deepy was disabled for these checks.

## Scope of the result

This verifies the tested plugin/host combination; it is not a guarantee for every future WanGP version or third-party plugin combination. The repository remains private, so installation from GitHub requires an authenticated clone or an authorized ZIP download.

No camera-guidance video was generated or evaluated. Windows/CUDA rendering, exact camera trajectories, complete orbits, and output-quality improvements are not established by these checks. Camera movement remains prompt guidance. The user's normal WanGP application was not restarted, and its generation queue was not used.
