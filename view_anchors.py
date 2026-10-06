"""Plan generated view anchors: one still per keyframe view, rendered from the Start Image.

Prompt wording alone leaves H3 free to choose which "side view" to end on, and
some seeds orbit past the subject's back to reach the far side. A picture of the
target view removes that ambiguity. Views come from Qwen Image Edit Plus (2511)
with fal's Multiple-Angles LoRA, whose camera vocabulary covers 45-degree orbit
steps and four heights, plus the 8-step Lightning LoRA (without it the angle
LoRA renders noisy views). Keyframes between those bins get no anchor, so an
anchor never shows a view the plan did not ask for.
"""

import math


QWEN_MODEL = "qwen_image_edit_plus2_20B"
ANGLES_LORA = ("https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/resolve/main/"
               "qwen-image-edit-2511-multiple-angles-lora.safetensors")
LIGHTNING_LORA = ("https://huggingface.co/DeepBeepMeep/Qwen_image/resolve/main/loras_accelerators/"
                  "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors")
# Positive azimuth moves the camera to its right; the LoRA's "right" views match (checked on renders).
AZIMUTHS = {0: "front view", 45: "front-right quarter view", 90: "right side view", 135: "back-right quarter view",
            180: "back view", -135: "back-left quarter view", -90: "left side view", -45: "front-left quarter view"}
ELEVATIONS = {-30: "low-angle shot", 0: "eye-level shot", 30: "elevated shot", 60: "high-angle shot"}
AZIMUTH_TOLERANCE = 5
ELEVATION_TOLERANCE = 15
PIXEL_BUDGET = 704 * 1280


def _match(pose):
    """Nearest LoRA view as (azimuth, elevation, azimuth error, elevation error)."""
    wrapped = (pose["azimuth"] + 180) % 360 - 180
    nearest = round(wrapped / 45) * 45
    elevation = min(ELEVATIONS, key=lambda angle: abs(pose["elevation"] - angle))
    return 180 if nearest == -180 else nearest, elevation, abs(wrapped - nearest), abs(pose["elevation"] - elevation)


def view_prompt(pose):
    """Multiple-angles prompt for a pose, or None when the LoRA cannot show it."""
    azimuth, elevation, azimuth_error, elevation_error = _match(pose)
    if azimuth_error > AZIMUTH_TOLERANCE or elevation_error > ELEVATION_TOLERANCE:
        return None
    # Shot sizes are absolute; from a full-body start, a 0.5x push-in matches the LoRA's close-up best.
    distance = "close-up" if pose["distance"] <= 0.6 else "wide shot" if pose["distance"] >= 1.6 else "medium shot"
    return f"<sks> {AZIMUTHS[azimuth]} {ELEVATIONS[elevation]} {distance}"


def anchor_targets(path, frames, keyframe_frame):
    """Return (targets, skipped): targets map a 0-based frame to its view prompt; skipped lists 1-based keyframes.

    The Start Image already shows the start view, so that view is never anchored again. When several
    keyframes round to one view, only the closest pose keeps it (with its exact repeats, i.e. a hold):
    the same still at two different poses would read as a stop between them."""
    start = view_prompt(path[0])
    candidates, skipped = {}, []
    for index, pose in enumerate(path[1:], 2):
        prompt = view_prompt(pose)
        if keyframe_frame(pose, frames) == 0:
            continue
        if prompt is None or prompt == start:
            skipped.append(index)
        else:
            candidates.setdefault(prompt, []).append((index, pose))
    targets = {}
    for prompt, members in candidates.items():
        best = min((pose for _, pose in members), key=lambda pose: _match(pose)[2:])
        for index, pose in members:
            if all(pose[field] == best[field] for field in ("azimuth", "elevation", "distance")):
                targets[keyframe_frame(pose, frames)] = prompt
            else:
                skipped.append(index)
    return targets, sorted(skipped)


def view_resolution(width, height):
    """Start Image aspect ratio at H3's usual pixel budget, in multiples of 16."""
    scale = math.sqrt(PIXEL_BUDGET / (width * height))
    return f"{max(16, round(width * scale / 16) * 16)}x{max(16, round(height * scale / 16) * 16)}"


def view_tasks(prompts, image_path, resolution, seed):
    """One Qwen Edit task per distinct prompt, in a stable order."""
    return [dict(model_type=QWEN_MODEL, prompt=prompt, image_mode=1, resolution=resolution,
                 num_inference_steps=8, guidance_scale=1, seed=int(seed), video_prompt_type="KI",
                 image_refs=[image_path], activated_loras=[ANGLES_LORA, LIGHTNING_LORA],
                 loras_multipliers="0.9 1", prompt_enhancer="", repeat_generation=1)
            for prompt in prompts]
