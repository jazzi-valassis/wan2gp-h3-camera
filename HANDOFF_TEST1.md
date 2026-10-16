# Handoff: why test 1's 45° orbit rendered as ~80–90°

Paste this into a Claude Code session **on the WanGP PC**, where the `wangp` MCP server is available (load it with `/mcp__wangp__wangp_agent` if that is how your tools start). The session needs the plugin's git checkout, WanGP, ffmpeg/ffprobe on PATH, and the Python environment WanGP uses (it has OpenCV and NumPy).

Goal: find the exact cause by experiment, fix it if the cause is in the plugin, and report with evidence. Don't guess from wording alone; every claim needs a render or an image to back it.

## 1. Context

**Repository.** `jazzi-valassis/wan2gp-h3-camera`, plugin version 0.5.0. Work on branch `claude/dreamy-mendel-kbc9q0` (currently the same as `main` at `5f85a7d`). Read `README.md`, `VALIDATION.md` (newest section first), `camera_plan.py` and `view_anchors.py` before starting.

**What the user reported for test 1** (clip `CAMERA TEST 1.mp4`, 243 frames, 24 fps, 704 x 1280):
> Much, much closer. Not a 45 degree orbit, but more of 75-90, but no spinning, no wild pathways, pretty close to predicted.

**What the clip shows** (measured in a cloud session; the user's copy was re-encoded, so it has no WanGP metadata):
- Start (frame 0): high-angle wide shot of the girl on a crate on a night rooftop. She is in side profile, facing screen right (her right side toward the camera).
- About 30° turned by frame 96 (4.0 s), about 60° by frame 160 (6.7 s), and face-on (front view, near eye level, medium shot) at frame 242. The crate turns as much as she does, so the turn is the camera's, not only hers. She turns her head to the camera early, so judge by her torso, knees and the crate, not her head.
- In the plugin's convention (positive orbit = camera moves to its right), a 45° right orbit from this start swings toward her face and should end on a **front three-quarter view**. The render went on to **face-on**, about 80–90°.
- The camera never stops. Background motion (from `timing.background_motion`) in px/frame: 7.7 at 0 s, 11 at 2 s, 11–16 between 3 and 5.7 s, about 5 from 5.8 to 7.5 s, then easing to 1.7 at 10 s. There is no stationary run (the plugin treats a still camera as 0.5 px/frame or less). If the path had a hold, H3 skipped it.

**What the plugin sends for a 45° path** (confirmed by compiling paths; the Start Image pose is always azimuth 0, elevation 0, distance 1):
- The plan text says coordinates are "relative to the starting view, looking toward the main subject at eye level" (`camera_plan.py:431`). This start is a high-angle wide shot, so coming down to eye level has to be entered as a negative elevation. The text then says the camera "ends at a low angle, looking up at about N degrees". That contradicts a high start coming down to eye level, which is a confirmed plugin problem with heights. Whether it affects the orbit is untested.
- View anchors use absolute LoRA labels (`view_anchors.py:21-44`). Anchors requested for a 45° path, by end elevation:

  | End elevation | Anchor at the last frame (243) |
  | --- | --- |
  | 0 to −10° | `front-right quarter view eye-level shot` + shot size |
  | −15 to −45° | `front-right quarter view low-angle shot` + shot size |
  | −48° or lower | none: **text guidance only** (elevation more than 15° from every LoRA height) |

  Shot size: `close-up` at 0.6x distance or closer, `wide shot` at 1.6x or farther, `medium shot` otherwise. These are picked as if the start were a medium shot; this start is wide.
- The azimuth names (`front view`, `front-right quarter view`, ...) were verified only with front-facing Start Images. There, "the view seen relative to the input image" and "the view seen relative to the subject's own front" give the same image, so earlier tests could not tell the two apart. This start is side-on, where they differ by 90°.
- The fal LoRA documentation does not say which of the two it uses.

**Precedent.** In 0.5.0 testing on the robot scene, a *text-only* 45° orbit (seed 144618832) "reached nearly the side view ... instead of 45 degrees", while anchored renders held the 45° view (`VALIDATION.md`, 0.5.0 section).

## 2. Hypotheses to separate

| ID | Cause | Predicts |
| --- | --- | --- |
| H1 | Apply fell back to text only (no keyframe matched a LoRA view), and H3 overshoots text-only orbits | Test 1's prompt has no "camera composition shown in <Picture N>" line; a text-only render overshoots; a correct 45° anchor fixes it |
| H2 | The anchor named the wrong view: the LoRA reads view names relative to the subject's front, so `front-right quarter view` from this side-on start is about 135° away | Step 2 grid shows `front view` face-on and `left side view` matching the start; test 1's anchor image is far past 45° |
| H3 | The names are relative to the input image, but the LoRA swings toward her face (frontal bias), so its "45°" view is really 80–90° | Step 2 grid shows `front view` matching the start, but `front-right quarter view` already face-on |
| H4 | The anchor was a correct 45° view and H3 overshot anyway | Variant C (a known-good 45° anchor) still ends at 80–90° |

More than one can be true (for example, H1 for test 1 plus H2 for anchored paths from side-on starts). Report each that the evidence supports.

## 3. Setup

Make an evidence folder under the WanGP root: `local_runtime/deepy_projects/h3_camera_test1_20261016/`. Name renders `h3cam_test1_<variant>_s<seed>_1016.mp4` and views `h3cam_test1_view_<label>_s<seed>.png`. Earlier runs are templates for submitting jobs through the `wangp` tools: `local_runtime/deepy_projects/h3_camera_spin_20261006/` (`anchor_settings.py`, `anchors/`, `eval/`, `live/`) and `local_runtime/deepy_projects/h3_camera_scene2_20261006/` (`build.py`). Do not change the user's saved presets.

Save this probe as `h3cam_probe.py` in the evidence folder:

```python
"""Measure an H3 Camera render: WanGP settings, camera motion, key-frame sheet.

python h3cam_probe.py PLUGIN_DIR meta   VIDEO
python h3cam_probe.py PLUGIN_DIR motion VIDEO
python h3cam_probe.py PLUGIN_DIR sheet  VIDEO OUT.png [FRAME ...]
"""
import json
import math
import re
import subprocess
import sys


def meta(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=comment", "-of", "json", video],
                         check=True, capture_output=True, text=True).stdout
    comment = ((json.loads(out).get("format") or {}).get("tags") or {}).get("comment") or ""
    try:
        settings = json.loads(comment)
    except ValueError:
        print("No WanGP settings in this file (re-encoded or renamed copy). Use the original render from the outputs folder.")
        return
    for key in ("model_type", "seed", "resolution", "video_length", "num_inference_steps", "video_prompt_type",
                "image_prompt_type", "frames_positions", "image_refs", "activated_loras"):
        if key in settings:
            print(f"{key}: {settings[key]}")
    prompt = settings.get("prompt") or ""
    plan = [line for line in prompt.splitlines()
            if line.startswith(("Camera plan:", "Camera coordinates", "At 0.000000s")) or re.match(r"\[\d+(\.\d+)?s–", line)]
    print("\n--- camera plan lines")
    print("\n".join(plan) or "(none)")
    anchored = re.findall(r"<Picture (\d+)>", "\n".join(plan))
    print("\nVIEW ANCHORS USED" if "camera composition shown in" in prompt else "\nTEXT ONLY (no view anchors in the plan)",
          f"- pictures referenced in plan: {sorted(set(anchored), key=int)}" if anchored else "")


def motion(video):
    import timing
    pairs = timing.background_motion(video)
    fps = timing.probe_video(video)["fps"]
    print("second  median_px/frame  (background motion; < 0.5 means the camera is still)")
    step = max(1, round(fps / 2))
    for start in range(0, len(pairs), step):
        values = sorted(p["pixels"] for p in pairs[start:start + step] if p["pixels"] is not None)
        if values:
            print(f"{start / fps:5.2f}s  {values[len(values) // 2]:6.2f}")
    runs, first = [], None
    for index, pair in enumerate(pairs + [dict(pixels=None)]):
        still = pair["pixels"] is not None and math.isfinite(pair["pixels"]) and pair["pixels"] <= 0.5
        if still and first is None:
            first = index
        if not still and first is not None:
            if index - first >= 3:
                runs.append((pairs[first]["frame"], pairs[index - 1]["frame"] + 1))
            first = None
    print("stationary runs (0-based frames):", runs or "none")


def sheet(video, output, frames):
    frames = frames or [0, 60, 96, 121, 160, 200, 242]
    select = "+".join(f"eq(n\\,{frame})" for frame in frames)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vf",
                    f"select='{select}',scale=-2:480,tile={len(frames)}x1", "-frames:v", "1", "-vsync", "vfr", output],
                   check=True)
    print("wrote", output, "- tiles left to right are 0-based frames", frames)


if __name__ == "__main__":
    plugin, command, video, *rest = sys.argv[1:]
    sys.path.insert(0, plugin)
    if command == "meta":
        meta(video)
    elif command == "motion":
        motion(video)
    else:
        sheet(video, rest[0], [int(frame) for frame in rest[1:]])
```

It was tested on the test-1 clip and on a synthetic render carrying WanGP-style metadata. On the test-1 copy, `meta` prints the "No WanGP settings" line and `motion` reports `stationary runs (0-based frames): none`.

## 4. Steps and expected output

### Step 1: Recover test 1's real settings

Find the original render in the WanGP outputs folder: 243 frames, 704 x 1280, a prompt containing `Camera plan: one continuous take`, made shortly before the user uploaded `CAMERA TEST 1.mp4`. Its frame 0 should match the uploaded clip's frame 0. Run `meta` on it, and save the path JSON (from the editor or the plan lines), the seed, the Start Image, and any `h3cam_view_*` images it used.

Expected output: the plan lines, the seed, and one of:
- `TEXT ONLY`: H1 is in play. The plan's end elevation probably falls outside the LoRA heights (check it against the table in section 1).
- `VIEW ANCHORS USED`: note the anchor prompt (from the `h3cam_view_<label>_s<seed>` file name) and look at the image. If it already shows her face-on or turned away to screen left, the anchor is wrong (H2 or H3).

If the original render can't be found, ask the user for the path JSON and the Apply status line, then continue with that.

### Step 2: How the LoRA reads view names from this start (about 3 minutes)

Generate views from the test-1 Start Image with exactly the plugin's view settings (`view_anchors.view_tasks`): `model_type="qwen_image_edit_plus2_20B"`, `image_mode=1`, `resolution="704x1280"`, `num_inference_steps=8`, `guidance_scale=1`, `video_prompt_type="KI"`, `image_refs=[start image]`, `activated_loras=[ANGLES_LORA, LIGHTNING_LORA]` (URLs in `view_anchors.py`), `loras_multipliers="0.9 1"`, `prompt_enhancer=""`, `seed=` the test-1 view seed (or 0). Prompts:

1. `<sks> front view elevated shot wide shot`
2. `<sks> front-right quarter view elevated shot wide shot`
3. `<sks> right side view elevated shot wide shot`
4. `<sks> front-left quarter view elevated shot wide shot`
5. `<sks> left side view elevated shot wide shot`
6. `<sks> front view eye-level shot medium shot`
7. The exact anchor prompt from step 1, if any.

Tile them with the Start Image into one sheet.

Expected output, read as a table (the start shows her right profile, facing screen right):

| Prompt | If names are relative to the input image | If names are relative to her own front |
| --- | --- | --- |
| front view | ≈ the start (right profile) | face-on |
| front-right quarter view | front three-quarter, facing screen right: **the correct 45° target** | three-quarter of her *left* side, facing screen left (≈135° from the start) |
| right side view | face-on | left profile, facing screen left |
| front-left quarter view | rear three-quarter (face hidden) | front three-quarter, facing screen right: **the correct 45° target** |
| left side view | her back | ≈ the start |

Prompt 6 against prompt 1 shows whether height and shot size are absolute (prompt 6 comes down to eye level and closer) or relative to the input (it stays high and wide). If `front-right quarter view` comes out face-on even though `front view` matches the start, that's H3.

### Step 3: Renders that separate the causes (about 40 minutes)

Use test 1's path, model and settings from step 1 (the user's preset was MiniMax H3 Ref2VA Singularity v1.3 Pruned 20B, 12 steps, one phase, Euler, shift 6, INT8 ConvRot, First Block Cache 0.08 from 25%, prompt enhancer off, 243 frames at 24 fps, 704 x 1280; the recovered settings win). Use two seeds: test 1's seed and 764485034. Change only the anchors:

- **A, text only:** Apply with view anchors off.
- **B, plugin anchor:** Apply with views on (B with test 1's seed should reproduce test 1). If step 1 showed text only, B is the same as A; then force an anchor by rounding the end elevation into the nearest LoRA height and record that you did so.
- **C, correct 45° anchor:** like B, but replace the injected image with a verified front three-quarter view at the planned end height and shot size. Generate it with the label step 2 identified as the 45° target (for example `front-left quarter view eye-level shot medium shot` if names are relative to her front). If no label gives a clean 45°, use frame 96–121 of test 1 and say so.

For each render, run `motion` and `sheet` (frames 0, 60, 96, 121, 160, 200, 242), then estimate the turn at the end from torso, knees and crate.

Expected decision:

| Result | Cause |
| --- | --- |
| Test 1 was text only; A ≈ 80–90°; C ≈ 45° | H1: no anchor was generated, and H3 overshoots text-only orbits |
| B's anchor image is ≥ 80°; B ≈ 80–90°; C ≈ 45° | H2 or H3, depending on step 2: the anchor asked for the wrong view |
| C still ≈ 80–90° with a verified 45° anchor | H4: H3 overrides a correct end anchor |

### Step 4: Fix (only for a cause in the plugin), test, commit

- **H2:** add Start Image view settings (which side of the subject it shows, its height, its shot size), defaulting to front, eye level and medium so existing paths and baselines are unchanged. Compute each LoRA label from the start view plus the move. Have `camera_plan.py` describe heights and view names from the declared start, so a descent from a high start reads as coming down toward eye level, not "looking up at N degrees".
- **H3:** pick the label by measured behaviour (step 2) instead of the nominal angle, or verify each generated view and refuse to anchor one that isn't within about 15° of the target.
- **H1:** anchor ends whose elevation is outside the LoRA heights by matching azimuth and the nearest height, or say clearly in the Apply status that the orbit is unanchored and likely to overshoot. Also fix the height wording above if the cause is height-related.
- **H4:** don't guess. Report the evidence and stop.

Then:
- Add regressions for the test-1 path and start.
- Run `python scripts/run_tests.py --host <WanGP root>`; all tests must pass.
- Add a `VALIDATION.md` section in the existing style (settings, seeds, a results table, evidence paths), and a `CHANGELOG.md` entry with a version bump in `plugin_info.json`.
- Commit to `claude/dreamy-mendel-kbc9q0` and push. Don't open a PR. Delete this handoff file in that commit.

## 5. What to report back

1. Test 1's recovered settings: path, seed, and anchored or text only, with the anchor image.
2. The step 2 sheet, with one sentence on how the LoRA reads view names, heights and shot sizes from a side-on start.
3. A results table for A, B and C × 2 seeds: final turn, highest and lowest height, whether it stopped, and sheet paths.
4. The cause in one sentence, naming H1–H4.
5. The fix commit and test count, or why no plugin fix applies.
