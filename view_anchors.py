"""Plan exact-pose view anchors: one still per keyframe view plus pacing checkpoints inside long moves.

Prompt wording alone leaves H3 free to choose which side view to end on, and some seeds orbit past the
subject's back. A picture of the target view removes that ambiguity; a picture in the middle of a long
move also tells H3 how far along the move should be by then, so it cannot finish early and linger.

Each view is rendered at the exact planned pose: ``view_geometry`` re-renders the nearest available view
(the Start Image, or a view made earlier in the same Apply) from the new camera using Depth Anything 3,
and Qwen Image 2.1 with the AnyAngle LoRA turns that coarse render into a clean picture in the Start
Image's style. The Pruna 8-step LoRA keeps a view to about ten seconds once the model is loaded.
"""

import math

QWEN_MODEL = "qwen_image_21_7B"
ANYANGLE_LORA = ("https://huggingface.co/lilylilith/QI_2.1_AnyAngle/resolve/5f6e234c1705c9b4eb653cb1d3452b0cfc987996/"
                 "QI2.1_AnyAngle.safetensors")
TURBO_LORA = ("https://huggingface.co/DeepBeepMeep/Qwen_image_2/resolve/8a136b04af66b7b20f689032f8d5a80f28add0cc/"
              "loras/p_qwen_image_2.1_8step_v0.1.safetensors")
VIEW_PROMPT = "Change the camera angle from <image2> to <image1>."
PIXEL_BUDGET = 704 * 1280
# Longest stretch of a move left without a picture; H3 holds its pace between closer anchors.
PACE_SECONDS = 2.6
# Moves smaller than this (in pose_gap units, roughly degrees) need no pacing checkpoint.
MIN_PACED_MOVE = 12.0
# A view is re-rendered from an existing one; farther than this, a helper view is made in between.
MAX_STEP = 50.0
# Measured views farther than this from the plan are retried once with the aim corrected, then left out:
# H3 follows an injected view within a few degrees, so a view this far off steers the render off the plan.
ACCEPT_GAP = 15.0


def pose_of(keyframe):
    return (float(keyframe["azimuth"]), float(keyframe["elevation"]), float(keyframe["distance"]))


def pose_gap(a, b):
    """Rough distance between two (azimuth, elevation, distance) poses, in degrees."""
    turn = (a[0] - b[0] + 180) % 360 - 180
    return math.hypot(turn, a[1] - b[1]) + 60 * abs(math.log(a[2] / b[2]))


def _ease(u, interpolation):
    return u if interpolation == "linear" else u * u * (3 - 2 * u)


def _blend(a, b, u):
    return tuple(x + (y - x) * u for x, y in zip(a, b))


def anchor_targets(path, frames, keyframe_frame, fps, interpolation="smooth"):
    """Views to render, in time order. Each target is a dict with ``pose`` (azimuth, elevation, distance),
    ``frames`` (zero-based frames that show it), ``kind`` ("keyframe" or "checkpoint") and ``keyframes``
    (1-based keyframe numbers). Returns (targets, skipped keyframe numbers).

    The Start Image already shows the start view, so a keyframe back at that pose is not anchored again.
    Keyframes sharing a pose (a hold) share one picture: the same still at both ends reads as a stop."""
    start = pose_of(path[0])
    targets, skipped, by_pose = [], [], {}
    used = {0}

    def add(pose, frame, kind, keyframe=None):
        if frame in used:
            return
        target = by_pose.get(pose) if kind == "keyframe" else None
        if target is None:
            target = dict(pose=pose, frames=[], kind=kind, keyframes=[])
            targets.append(target)
            if kind == "keyframe":
                by_pose[pose] = target
        target["frames"].append(frame)
        if keyframe is not None:
            target["keyframes"].append(keyframe)
        used.add(frame)

    for index, (left, right) in enumerate(zip(path, path[1:]), 2):
        a, b = pose_of(left), pose_of(right)
        first, last = keyframe_frame(left, frames), keyframe_frame(right, frames)
        seconds = (last - first) / fps
        if a != b and pose_gap(a, b) >= MIN_PACED_MOVE and seconds > PACE_SECONDS:
            count = math.ceil(seconds / PACE_SECONDS) - 1
            for step in range(1, count + 1):
                u = step / (count + 1)
                frame = int(round(first + (last - first) * u))
                # Keep checkpoints clear of keyframe pictures.
                if first + 2 <= frame <= last - 2:
                    add(_blend(a, b, _ease(u, interpolation)), frame, "checkpoint")
        if last == 0:
            continue
        if b == start:
            skipped.append(index)
            continue
        add(b, last, "keyframe", index)
    targets.sort(key=lambda target: min(target["frames"]))
    return targets, skipped


def helper_pose(previous, target):
    """Pose MAX_STEP along the path from the ``previous`` planned pose toward ``target``, or None when the target
    is within reach. Azimuths are the plan's unwrapped values, so helpers turn the way the path turns."""
    gap = math.hypot(target[0] - previous[0], target[1] - previous[1]) + 60 * abs(math.log(target[2] / previous[2]))
    if gap <= MAX_STEP:
        return None
    u = MAX_STEP / gap
    return (previous[0] + (target[0] - previous[0]) * u, previous[1] + (target[1] - previous[1]) * u,
            previous[2] * (target[2] / previous[2]) ** u)


def aimed_pose(target, measured):
    """Target corrected by the last miss: aim past the plan by as much as the view fell short."""
    turn = (target[0] - measured[0] + 180) % 360 - 180
    return (target[0] + turn, max(-85.0, min(85.0, 2 * target[1] - measured[1])),
            max(0.1, target[2] * target[2] / measured[2]))


def view_resolution(width, height):
    """Start Image aspect ratio at H3's usual pixel budget, in multiples of 16."""
    scale = math.sqrt(PIXEL_BUDGET / (width * height))
    return f"{max(16, round(width * scale / 16) * 16)}x{max(16, round(height * scale / 16) * 16)}"


VIEW_PREFIX = "h3cam_view_"


def _signed(value, positive, negative):
    return f"{abs(value):g}{positive if value >= 0 else negative}"


def pose_label(pose):
    """Short pose text such as ``45r 48u 0.75x``."""
    azimuth, elevation, distance = pose
    return f"{_signed(round(azimuth, 1), 'r', 'l')} {_signed(round(elevation, 1), 'u', 'd')} {round(distance, 3):g}x"


def view_name(pose, seed, attempt=0):
    """Output file name for a generated view; the prefix marks it as replaceable on the next Apply."""
    name = VIEW_PREFIX + pose_label(pose).replace(" ", "_").replace(".", "p") + f"_s{int(seed)}"
    return name + (f"_{attempt}" if attempt else "")


def is_generated_view(path):
    """Whether a reference image path is a view generated by this plugin."""
    return isinstance(path, str) and path.replace("\\", "/").rsplit("/", 1)[-1].startswith(VIEW_PREFIX)


def _target_text(target, fps):
    when = " and ".join(f"{frame / fps:.2f}s" for frame in sorted(target["frames"]))
    what = (f"keyframe{'s' if len(target['keyframes']) > 1 else ''} {', '.join(map(str, target['keyframes']))}"
            if target["kind"] == "keyframe" else "pacing checkpoint")
    return f"{what} at {when}: {pose_label(target['pose'])}"


def describe(targets, skipped, measured, fps):
    """Plain summary of the anchored views for the Apply status; ``measured`` holds one pose per target."""
    parts = [f"{_target_text(target, fps)} (measured {pose_label(got)})" for target, got in zip(targets, measured)]
    text = " Anchored views: " + "; ".join(parts) + "."
    if skipped:
        text += f" Keyframes {', '.join(map(str, skipped))} return to the start view, which the Start Image already shows."
    return text


def describe_missed(missed, fps):
    """Apply status text for views left out after both attempts missed; ``missed`` holds (target, measured) pairs."""
    if not missed:
        return ""
    parts = [f"{_target_text(target, fps)} (closest attempt measured {pose_label(got)})" for target, got in missed]
    return (f" Left out because neither attempt came within {ACCEPT_GAP:g} degrees of the plan, so text alone guides "
            "these poses: " + "; ".join(parts) + ".")


PLATE_PROMPT = ("Remove the main subject and everything it rests on. Fill the masked area with the empty background "
                "of the same scene, continuing the floor, walls and distant details seamlessly, with matching light and texture.")


def plate_task(image_path, mask_path, resolution, seed, name):
    """Qwen Image 2.1 inpainting task that paints the subject out of the Start Image (the background plate)."""
    return dict(model_type=QWEN_MODEL, prompt=PLATE_PROMPT, image_mode=2, resolution=resolution,
                num_inference_steps=8, guidance_scale=1, sample_solver="pruna", seed=int(seed),
                video_prompt_type="VAG", image_guide=image_path, image_mask=mask_path, denoising_strength=1.0,
                masking_strength=1.0, model_mode=0, activated_loras=[TURBO_LORA], loras_multipliers="1",
                negative_prompt="", prompt_enhancer="", repeat_generation=1, output_filename=name)


def view_task(image_path, coarse_path, resolution, seed, name):
    """Qwen Image 2.1 task that turns a coarse re-render into a clean view in the Start Image's style."""
    return dict(model_type=QWEN_MODEL, prompt=VIEW_PROMPT, image_mode=1, resolution=resolution,
                num_inference_steps=8, guidance_scale=1, sample_solver="pruna", seed=int(seed),
                video_prompt_type="KI", image_refs=[image_path, coarse_path],
                activated_loras=[ANYANGLE_LORA, TURBO_LORA], loras_multipliers="1 1",
                negative_prompt="", prompt_enhancer="", repeat_generation=1, output_filename=name)
