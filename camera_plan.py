"""Compile a camera sketch into native H3 prompt sections without model dependencies.

This is an original WanGP implementation inspired by NyckM's prompt-based
3d-Camera-control-H3-Minimax editor (https://github.com/NyckM/3d-Camera-control-H3-Minimax).
Coordinates are prose requests, not camera tensors: exact trajectories, timing,
and a completed orbit cannot be guaranteed. A matching end image only conditions
the endpoint. Scene/audio instructions remain the user's; freezing is opt-in.
"""

import json
import math
import re


MAX_PATH_BYTES = 32 * 1024
MAX_PROMPT_BYTES = 1024 * 1024
MAX_AZIMUTH_TRAVEL = 11520.0
FIELDS = ("time", "azimuth", "elevation", "distance")
ORIGIN = dict(time=0.0, azimuth=0.0, elevation=0.0, distance=1.0)
FL2VA_SECTIONS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")
REF2VA_SECTIONS = ("subject_definitions", "summary", "retention_analysis", "detailed_description",
                   "overall_soundscape", "non_diegetic_music")
SECTION_RE = re.compile(r"(?m)^\s*(" + "|".join(dict.fromkeys(FL2VA_SECTIONS + REF2VA_SECTIONS)) + r")\s*:")
BLOCK_RE = re.compile(r"(?ms)^# WanGP H3 camera (plan|alignment) begin\n.*?^# WanGP H3 camera \1 end(?:\n|$)")
# WanGP omits comment lines before inference. Recognize the compiler's exact
# prose structure too, so reusing a processed prompt does not duplicate plans.
PLAIN_PLAN_RE = re.compile(
    r"(?m)^Camera plan: one continuous take, [^\n]+; last frame at [^\n]+\. No cuts\.\n"
    r"Camera coordinates are relative to the starting view,[^\n]+\n"
    r"(?:Frozen scene:|Allow the subject and environment to move)[^\n]+\n"
    r"(?:Ease smoothly into and out of each segment;|Use a constant rate within each segment,)[^\n]+\n"
    r"(?:\[\d+(?:\.\d+)?s–\d+(?:\.\d+)?s\] [^\n]+(?:\n|$))+"
    r"(?:Complete the full [^\n]+-turn camera journey and return to the starting viewpoint at the last frame\.(?:\n|$))?"
)
PLAIN_ALIGNMENT_RE = re.compile(
    r"(?m)^At 0\.000000s, <Picture 1> anchors the first frame of \[Shot 1\]; preserve its starting composition\.(?:\n|$)"
    r"(?:At \d+(?:\.\d+)?s, <Picture 2> is the same image and anchors the last frame of \[Shot 1\]\.(?:\n|$))?"
)


def _preset(azimuth=0, elevation=0, distance=1):
    return json.dumps([ORIGIN, dict(time=1, azimuth=azimuth, elevation=elevation, distance=distance)])


PRESETS = {
    "Static": _preset(),
    "Orbit right 90°": _preset(azimuth=90),
    "Orbit left 90°": _preset(azimuth=-90),
    "Half orbit": _preset(azimuth=180),
    "Full orbit": _preset(azimuth=360),
    "Push in": _preset(distance=0.65),
    "Pull back": _preset(distance=1.5),
    "Rise": _preset(elevation=20),
}
DEFAULT_PATH = PRESETS["Orbit right 90°"]


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    try:
        number = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite number.") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    return number


def validate_path(path_json):
    """Return independent normalized poses; times must cover the whole take.

    Azimuth is unwrapped: 360 means one full turn, not a zero-degree move.
    A hold is represented by two equal poses with different times. First pose
    is exactly time=0, azimuth=0, elevation=0, distance=1; the last time is 1.
    """
    if not isinstance(path_json, str) or len(path_json.encode("utf-8")) > MAX_PATH_BYTES:
        raise ValueError("Camera path must be a JSON array smaller than 32 KB.")
    try:
        path = json.loads(path_json)
    except (ValueError, RecursionError) as error:
        raise ValueError("Camera path must be a valid JSON array.") from error
    if not isinstance(path, list) or not 2 <= len(path) <= 24:
        raise ValueError("Use between 2 and 24 camera keyframes.")
    result = []
    for index, item in enumerate(path, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Keyframe {index} must be an object.")
        if set(item) != set(FIELDS):
            raise ValueError(f"Keyframe {index} must contain only time, azimuth, elevation and distance.")
        pose = {field: _finite_number(item[field], f"Keyframe {index} {field}") for field in FIELDS}
        if not 0 <= pose["time"] <= 1:
            raise ValueError(f"Keyframe {index}: time must be between 0 and 1.")
        if not -89 <= pose["elevation"] <= 89:
            raise ValueError(f"Keyframe {index}: elevation must be between -89 and 89 degrees.")
        if not 0.1 <= pose["distance"] <= 4:
            raise ValueError(f"Keyframe {index}: distance must be between 0.1 and 4.")
        if abs(pose["azimuth"]) > MAX_AZIMUTH_TRAVEL:
            raise ValueError(f"Keyframe {index}: azimuth exceeds 32 full turns.")
        result.append(pose)
    if result[0] != ORIGIN:
        raise ValueError("The first keyframe must be time=0, azimuth=0, elevation=0, distance=1.")
    if result[-1]["time"] != 1:
        raise ValueError("The last keyframe must be at time=1; add an identical final pose for a hold.")
    if any(right["time"] <= left["time"] for left, right in zip(result, result[1:])):
        raise ValueError("Keyframe times must be strictly increasing.")
    if sum(abs(right["azimuth"] - left["azimuth"]) for left, right in zip(result, result[1:])) > MAX_AZIMUTH_TRAVEL:
        raise ValueError("Total azimuth travel exceeds 32 full turns.")
    return result


def normalize_frames(frame_count):
    """Match H3 pipeline's upward rounding to its native 17k+5 frame grid.

    The plugin also applies the selected model's UI minimum (normally 107).
    """
    count = _finite_number(frame_count, "Frame count")
    if count < 5 or not count.is_integer():
        raise ValueError("Frame count must be a whole number of at least 5.")
    return ((int(count) - 5 + 16) // 17) * 17 + 5


def is_closed_path(path):
    """Whether the endpoint returns after a nonzero integer number of turns."""
    end = path[-1]
    turns = end["azimuth"] / 360.0
    return (abs(turns) >= 1 and math.isclose(turns, round(turns), rel_tol=0, abs_tol=1e-9)
            and math.isclose(end["elevation"], 0, rel_tol=0, abs_tol=1e-9)
            and math.isclose(end["distance"], 1, rel_tol=0, abs_tol=1e-9))


def strip_camera_plan(prompt):
    """Remove only blocks owned by this plugin, leaving user text and labels."""
    text = prompt.replace("\r\n", "\n").replace("\r", "\n")
    text = BLOCK_RE.sub("", text)
    text = PLAIN_PLAN_RE.sub("", text)
    text = PLAIN_ALIGNMENT_RE.sub("", text)
    if "# WanGP H3 camera " in text:
        raise ValueError("A camera plan marker is incomplete; restore both begin/end markers or remove that block.")
    if re.search(r"(?m)^Camera plan: one continuous take,", text):
        raise ValueError("A processed camera plan was edited and cannot be replaced safely; remove its camera directions before applying again.")
    return text.strip()


def _clean_prompt(prompt):
    if not isinstance(prompt, str) or len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise ValueError("The scene prompt must be text smaller than 1 MB.")
    text = strip_camera_plan(prompt)
    if re.search(r"\[\s*/", text) or re.search(r"(?m)^\s*---\s*$", text):
        raise ValueError("Camera planning supports one shot; remove sliding-window commands or multiple-shot separators first.")
    shots = re.findall(r"\[\s*Shot\s+(\d+)\s*\]", text, re.IGNORECASE)
    if any(int(shot) != 1 for shot in shots):
        raise ValueError("Camera planning supports one continuous shot; remove Shot 2 and later cuts first.")
    # Plain-text paragraphs can describe one scene. Blank lines must not turn
    # sections into separate windows if the user later selects paragraph mode.
    return "\n".join(line.rstrip() for line in text.splitlines() if line.strip())


def _sections(prompt, reference_mode):
    matches = list(SECTION_RE.finditer(prompt))
    existing = {}
    prefix = prompt[:matches[0].start()].strip() if matches else ""
    for index, match in enumerate(matches):
        name = match.group(1)
        if name in existing:
            raise ValueError(f"The prompt has more than one {name} section; use one shot per camera plan.")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(prompt)
        existing[name] = prompt[match.end():end].strip()
    if reference_mode and "integrated_multimodal_description" in existing:
        raise ValueError("This is an FL2VA structured prompt; use the FL2VA model or change its visual field to detailed_description for Ref2VA.")
    if not reference_mode and any(name in existing for name in REF2VA_SECTIONS[:4]):
        raise ValueError("This is a Ref2VA structured prompt; use a Ref2VA model or an FL2VA three-section prompt.")
    visual = "detailed_description" if reference_mode else "integrated_multimodal_description"
    if visual not in existing:
        existing[visual] = prefix if matches else prompt
        prefix = ""
    if not existing[visual].strip():
        existing[visual] = "[Shot 1] Continue the requested scene." if matches else "[Shot 1] Show the main subject in one continuous take."
    defaults = {
        "subject_definitions": "Use the subjects and reference roles specified in the description.",
        "summary": "[reference generation] One continuous camera take of the requested scene.",
        "retention_analysis": "Preserve the requested identities, appearance, setting and reference roles.",
        "overall_soundscape": "Follow the scene's audio and dialogue instructions; do not add unrequested sound.",
        "non_diegetic_music": "Follow the scene's music instructions; do not add unrequested music.",
    }
    names = REF2VA_SECTIONS if reference_mode else FL2VA_SECTIONS
    return prefix, {name: existing.get(name, defaults.get(name, "")) for name in names}, visual


def _block(kind, lines):
    return f"# WanGP H3 camera {kind} begin\n" + "\n".join(lines) + f"\n# WanGP H3 camera {kind} end"


def _segment(left, right):
    da = right["azimuth"] - left["azimuth"]
    de = right["elevation"] - left["elevation"]
    dd = right["distance"] - left["distance"]
    changes = []
    if da:
        changes.append(f"orbit {abs(da):g} degrees toward camera {'right' if da > 0 else 'left'} "
                       f"(azimuth {left['azimuth']:g} to {right['azimuth']:g} degrees)")
    if de:
        changes.append(f"{'raise' if de > 0 else 'lower'} camera elevation from {left['elevation']:g} to {right['elevation']:g} degrees")
    if dd:
        changes.append(f"{'dolly back' if dd > 0 else 'dolly in'} from {left['distance']:g}x to {right['distance']:g}x the starting distance")
    if not changes:
        return "Hold the camera at this pose."
    unchanged = []
    if not de:
        unchanged.append(f"elevation {right['elevation']:g} degrees")
    if not dd:
        unchanged.append(f"distance {right['distance']:g}x")
    text = "; simultaneously ".join(changes)
    if unchanged:
        text += "; maintain " + " and ".join(unchanged)
    return text[0].upper() + text[1:] + "."


def compile_plan(path_json, *, prompt, frame_count, fps, frozen=False, close_loop=False,
                 reference_mode=False, has_start_image=False, interpolation="smooth"):
    """Return a single-shot prompt, timing, display rows, and safe form settings.

    ``rows`` contains [keyframe_number, seconds, azimuth, elevation, distance].
    ``duration_seconds`` is frames/fps; ``end_seconds`` is the timestamp of the
    last frame, (frames-1)/fps. The caller must apply ``multi_prompts_gen_type``
    (FG) alongside the prompt and arrange first/last image conditioning when
    ``close_loop`` is true. Compiling alone never manipulates image inputs.
    """
    path = validate_path(path_json)
    frames = normalize_frames(frame_count)
    rate = _finite_number(fps, "Frame rate")
    if rate <= 0:
        raise ValueError("Frame rate must be positive.")
    for name, value in (("frozen", frozen), ("close_loop", close_loop),
                        ("reference_mode", reference_mode), ("has_start_image", has_start_image)):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be true or false.")
    if interpolation not in ("smooth", "linear"):
        raise ValueError("Interpolation must be smooth or linear.")
    if close_loop and not has_start_image:
        raise ValueError("Add a Start Image before enabling a closed camera loop.")
    if close_loop and not is_closed_path(path):
        raise ValueError("A closed loop must end after nonzero whole 360-degree turns at elevation 0 and distance 1.")
    cleaned = _clean_prompt(prompt)
    prefix, sections, visual = _sections(cleaned, reference_mode)
    end_seconds = (frames - 1) / rate
    rows = [[index, pose["time"] * end_seconds, pose["azimuth"], pose["elevation"], pose["distance"]]
            for index, pose in enumerate(path, 1)]
    travel = sum(abs(right["azimuth"] - left["azimuth"]) for left, right in zip(path, path[1:]))
    camera = [
        f"Camera plan: one continuous take, {frames} frames at {rate:g} fps; last frame at {end_seconds:.6f}s. No cuts.",
        "Camera coordinates are relative to the starting view, looking toward the main subject: azimuth 0 degrees, elevation 0 degrees, distance 1x. "
        "Positive azimuth means the camera travels to its right around the subject; angles stay unwrapped across full turns. "
        "Move the camera through the scene with natural parallax while keeping the subject framed.",
        ("Frozen scene: keep subjects, expressions, objects, water, smoke and background motion still; only the camera moves. Preserve the requested soundtrack."
         if frozen else "Allow the subject and environment to move naturally according to the scene description, with requested speech and action synchronized to the audio."),
        ("Ease smoothly into and out of each segment; briefly settle at each keyframe."
         if interpolation == "smooth" else "Use a constant rate within each segment, changing direction at its keyframes."),
    ]
    for index, (left, right) in enumerate(zip(path, path[1:])):
        camera.append(f"[{rows[index][1]:.6f}s–{rows[index + 1][1]:.6f}s] {_segment(left, right)}")
    if close_loop:
        camera.append(f"Complete the full {abs(path[-1]['azimuth']) / 360:g}-turn camera journey and return to the starting viewpoint at the last frame.")
    sections[visual] = sections[visual] + "\n" + _block("plan", camera)
    parts = [prefix] if prefix else []
    if has_start_image:
        alignment = ["At 0.000000s, <Picture 1> anchors the first frame of [Shot 1]; preserve its starting composition."]
        if close_loop:
            alignment.append(f"At {end_seconds:.6f}s, <Picture 2> is the same image and anchors the last frame of [Shot 1].")
        parts.append(_block("alignment", alignment))
    parts.extend(f"{name}: {value}" for name, value in sections.items())
    summary = (f"{len(path)} keyframes · {frames} frames at {rate:g} fps · {frames / rate:.3f}s clip "
               f"· {travel:g}° total orbit travel · {'frozen scene' if frozen else 'subject motion allowed'}. "
               "Camera motion and timing are prompt guidance, not exact 3D constraints.")
    if close_loop:
        summary += " Matching first/last images encourages a loop but does not guarantee a complete orbit."
    return {"prompt": "\n".join(parts), "frame_count": frames, "fps": rate,
            "duration_seconds": frames / rate, "end_seconds": end_seconds,
            "close_loop": close_loop, "summary": summary, "rows": rows,
            "multi_prompts_gen_type": "FG", "path": path}
