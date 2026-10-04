# Changelog

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
