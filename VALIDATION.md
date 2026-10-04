# H3 Camera validation

Release 0.2.1 changes author credits, release documentation, and the version identifier only. Its planner code and browser assets match 0.2.0 apart from that identifier. The behavioral results below belong to 0.2.0 and are retained as the baseline; they are not a claim of a new full application or GPU run.

Release 0.2.0 was checked on 2026-10-04 against WanGP 13.141, upstream revision `b8b18f8114e432eea8f3d7e853a51dd91fa99571`, with Python 3.11.11 and Gradio 5.29.0.

## Automated checks

```sh
python scripts/run_tests.py --suite plan
python scripts/run_tests.py --host /path/to/Wan2GP --suite integration
```

The sixteen compiler tests passed independently with standard Python. Together with the eighteen form and export tests, all **34 CPU checks passed** in the isolated verification environment. Coverage includes strict plan validation, frame alignment, FPS timing, prompt/audio/reference preservation, repeat application, unsupported input rejection, loop endpoint safeguards, saved-plan round trips, supported JavaScript registration, compatibility diagnostics, independent session inputs, and export cleanup. The startup and diagnostic changes first failed their new assertions on the baseline. Export lifecycle assertions also failed before the corresponding implementation existed.

The verification environment used the unmodified pinned plugin loader, frame scheduler, and prompt parser. Test-supplied model metadata and FPS functions isolate form behavior from model loading. The loader's unused configuration-migration import was replaced by a boundary that raises if called. No migration, model initialization, or GPU code was executed. This is a scoped compatibility check, not a complete WanGP startup.

The package also retains four additional native-model tests under `test_h3_camera_roundtrip.py`. They require the full WanGP environment and **were not rerun for this release**. Run `--suite native` there; `--suite all` includes them. Historical checks from the previous local plugin are not counted as current results.

Static package checks passed with ten Python files and no errors or warnings. Both browser JavaScript files passed `node --check`.

## Loader and browser checks

An isolated Gradio test interface loaded a fresh package copy through the real WanGP plugin manager. Discovery found its extension metadata without executing the plugin. Disabled loading registered no plugin or JavaScript. Enabled loading constructed exactly one main-form panel across main and edit setup passes, and Apply remained bound to the main form. The checks reported no loader errors.

Eleven checks passed in Chrome against that interface: Preview preserved the scene; Apply preserved the scene and audio; the selected full orbit appeared in the prompt; repeat Apply kept one identical plan; the edit form remained untouched; invalid JSON disabled playback; a valid preset restored the path; controls recovered; Save produced a downloadable plan; repeated Save remained usable; and the browser reported no page errors. These checks exercised real DOM events, the sandboxed editor, Gradio callbacks, and file outputs. The interface used test model metadata rather than loading H3.

The release builder includes only the plugin's own sources, assets, documentation, and portable tests. It writes an archive checksum and per-file manifest. Installation uses the existing `wan2gp-h3-camera` folder name. No host core edits or additional dependencies are required.

The user's full WanGP application was not restarted for this release. Its saved model settings, enabled-plugin configuration, and generation queue were not modified by these checks. An installed-file update becomes active after a normal restart.

## Scope of the result

These checks establish editor and generation-form integration. No camera-guidance video was generated or evaluated, and no same-seed visual comparison was performed. Camera movement remains text guidance; exact trajectories, complete orbits, and improved output quality are not established by these checks.
