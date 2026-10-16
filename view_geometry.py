"""Exact-pose view anchors from depth: reconstruct a view in 3D, re-render it from a new camera, measure poses.

Depth Anything 3 (bundled with WanGP) estimates depth and focal length for one or several views at once.
A view's pixels become a point cloud, which is splatted into the camera that orbits the subject by the
requested azimuth, elevation and distance. The resulting coarse render has gaps where the source never
saw the scene; Qwen Image 2.1 with the AnyAngle LoRA turns it into a clean picture that keeps the Start
Image's look. Measuring the result against the Start Image tells how close it landed to the plan.

Coordinates follow OpenCV: x right, y down, z forward. The Start Image's camera sits at the origin; the
orbit pivot is the depth of the frame's centre region (the subject in a centred composition).
"""

import math

import numpy as np

DEPTH_MODEL = "depth/depth_anything_v3_vitl_bf16.safetensors"
PROCESS_RES = 756  # Multiple of the model's 14-pixel patches.
FAR = 50.0
# Extra share of the target view the Start Image is credited with when choosing a source to re-render.
START_BONUS = 0.1
BACKGROUND = (60, 60, 60)


class DepthModel:
    """Depth Anything 3 loaded on demand and released with ``close``."""

    def __init__(self, device=None):
        self.device = device
        self.model = None

    def _load(self):
        import torch
        from shared.utils import files_locator as fl
        from shared.utils.download import process_files_def_if_needed
        from preprocessing.depth_anything_v3.assets import query_download_def
        from preprocessing.depth_anything_v3.depth import _load_da3
        process_files_def_if_needed(query_download_def())
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _load_da3(fl.locate_file(DEPTH_MODEL), self.device, model_name="da3-large")

    def reconstruct(self, images):
        """Depths (N,H,W), camera-to-world poses (N,4,4) relative to the first view, intrinsics (N,4) as fx fy cx cy.

        The model only occupies the GPU while it runs, so the image model keeps its memory between views."""
        import torch
        from preprocessing.depth_anything_v3.depth import _run_da3_prediction
        if self.model is None:
            self._load()
        size = images[0].size
        video = np.stack([np.asarray(image.convert("RGB").resize(size)) for image in images])
        self.model.to(self.device)
        try:
            with torch.inference_mode():
                depths, _, c2w, intrinsics = _run_da3_prediction(self.model, video, PROCESS_RES, chunk_size=0)
        finally:
            self.model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return depths.astype(np.float64), c2w.astype(np.float64), intrinsics.astype(np.float64)

    def close(self):
        import torch
        if self.model is not None:
            self.model.to("cpu")
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def subject_mask(image):
    """Main-subject mask from WanGP's background remover (rembg u2net), or None when unavailable."""
    try:
        from rembg import remove
        from shared.utils.utils import new_rembg_session
    except ImportError:
        return None
    mask = remove(image.convert("RGB"), session=new_rembg_session("u2net"), only_mask=True)
    return np.asarray(mask.convert("L")) >= 128


def plate_mask(mask, reach=0.02):
    """The subject mask grown by ``reach`` of the frame, so the painted-out plate also covers its edges and
    contact shadow."""
    import cv2
    size = max(3, int(round(max(mask.shape) * reach)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def pivot_point(depth, k, mask=None):
    """Orbit pivot: the 3D centre of the main subject. A thin subject such as a seated robot covers little of
    the frame centre, so the centre's median depth would land on the wall behind it and swing the orbit wide.
    Without a usable subject mask, the nearest substantial depth in the centre region stands in for it."""
    h, w = depth.shape
    fx, fy, cx, cy = k
    if mask is not None and mask.shape == depth.shape and mask.mean() >= .005:
        v, u = np.nonzero(mask)
        z = depth[v, u]
        # The median keeps mask bleed onto the background from pulling the pivot back.
        zc = float(np.median(z))
        near = np.abs(np.log(z / zc)) < .25
        uc, vc = float(np.median(u[near])) + .5, float(np.median(v[near])) + .5
        return np.array([(uc - cx) / fx * zc, (vc - cy) / fy * zc, zc])
    centre = depth[int(h * .35):int(h * .65), int(w * .35):int(w * .65)]
    return np.array([0.0, 0.0, float(np.percentile(centre, 25))])


def _axis_rotation(axis, degrees):
    axis = axis / np.linalg.norm(axis)
    a = math.radians(degrees)
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * cross + (1 - math.cos(a)) * cross @ cross


UP = np.array([0.0, -1.0, 0.0])


def orbit(pivot, azimuth, elevation, distance):
    """Camera-to-world (rotation, centre) after orbiting the start camera about ``pivot``.

    Positive azimuth moves the camera to its right around the vertical axis through the pivot; positive
    elevation raises it within the vertical plane through the pivot, so the two never mix even when the
    subject sits off the frame centre; distance scales the camera-to-pivot distance. The camera keeps its
    framing relative to the pivot."""
    start = -np.asarray(pivot, float)
    flat = start - UP * (start @ UP)
    tilt = _axis_rotation(np.cross(flat, UP), elevation)
    turn = np.array([[math.cos(math.radians(azimuth)), 0, -math.sin(math.radians(azimuth))], [0, 1, 0],
                     [math.sin(math.radians(azimuth)), 0, math.cos(math.radians(azimuth))]])
    rotation = turn @ tilt
    return rotation, pivot + distance * rotation @ start


def pose_of(c2w, pivot):
    """(azimuth, elevation, distance) of a camera around ``pivot``, in the plan's conventions."""
    up = UP
    v, start = c2w[:3, 3] - pivot, -pivot
    flat, start_flat = v - up * (v @ up), start - up * (start @ up)
    azimuth = math.degrees(math.atan2(np.cross(start_flat, flat) @ up, start_flat @ flat))
    # Relative to the start view, which looks slightly up or down at a subject off the frame centre.
    height = lambda vector: math.degrees(math.asin(float(np.clip((vector @ up) / np.linalg.norm(vector), -1, 1))))
    return azimuth, height(v) - height(start), float(np.linalg.norm(v) / np.linalg.norm(start))


def pose_gap(a, b):
    """Rough distance between two (azimuth, elevation, distance) poses, in degrees."""
    turn = (a[0] - b[0] + 180) % 360 - 180
    return math.hypot(turn, a[1] - b[1]) + 60 * abs(math.log(a[2] / b[2]))


def expand_far(depth, near, far=FAR):
    """Monocular depth is affine in disparity and squeezes distant scenery toward the subject. Shift the
    disparity so the farthest structure sits ``far`` times the pivot distance while the subject keeps its
    scale; otherwise a skyline slides like a nearby wall when the camera orbits."""
    d = near / depth
    dmin = float(np.percentile(d, 0.5))
    if dmin >= 1 - 1e-6:
        return depth  # Nothing lies beyond the pivot.
    d0 = (dmin - 1 / far) / (1 - 1 / far)
    return near / np.maximum((d - d0) / (1 - d0), 1 / (2 * far))


def point_cloud(image, depth, k, c2w=None, edge=0.08, only=None):
    """World points, colours and source depths of one view. Points on depth edges are dropped: seen from
    the side they would smear into streaks between foreground and background. ``only`` limits the cloud to
    a pixel mask (a background plate contributes only where the subject stood)."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    h, w = depth.shape
    fx, fy, cx, cy = k
    z = torch.from_numpy(np.ascontiguousarray(depth)).to(device, torch.float64)
    v, u = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float64),
                          torch.arange(w, device=device, dtype=torch.float64), indexing="ij")
    points = torch.stack([(u + .5 - cx) / fx * z, (v + .5 - cy) / fy * z, z], -1).reshape(-1, 3)
    colors = torch.from_numpy(np.array(image.convert("RGB").resize((w, h)))).to(device).reshape(-1, 3)
    lz = torch.log(z)
    jump = torch.zeros_like(lz, dtype=torch.bool)
    jump[1:-1, 1:-1] = torch.maximum((lz[1:-1, 2:] - lz[1:-1, :-2]).abs(), (lz[2:, 1:-1] - lz[:-2, 1:-1]).abs()) > edge
    # At a depth edge keep the nearest pixels (thin legs, rails and hair stay) and drop the in-between ones.
    nearest = -torch.nn.functional.max_pool2d(-lz[None, None], 3, stride=1, padding=1)[0, 0]
    flying = jump & (lz > nearest + edge / 2)
    keep = ~flying.reshape(-1)
    if only is not None:
        keep &= torch.from_numpy(np.ascontiguousarray(only)).to(device).reshape(-1)
    # The frame's outline, clockwise from the top-left corner, marks where the source picture ends.
    grid = points.reshape(h, w, 3)
    step = max(1, min(h, w) // 64)
    outline = torch.cat([grid[0, ::step], grid[::step, -1], grid[-1].flip(0)[::step], grid[:, 0].flip(0)[::step]])
    points, colors, source = points[keep], colors[keep], z.reshape(-1)[keep]
    if c2w is not None:
        m = torch.from_numpy(np.ascontiguousarray(c2w)).to(device)
        points = points @ m[:3, :3].T + m[:3, 3]
        outline = outline @ m[:3, :3].T + m[:3, 3]
    return points, colors, source, (w, h), outline


def render(cloud, k, rotation, centre, size, fill=True, layers=()):
    """Splat a point cloud, plus optional hidden ``layers`` behind it, into a camera with a z-buffer.
    Returns (RGB array, covered mask); the first cloud's frame outline bounds the gaps that get filled."""
    import torch
    points, colors, source, source_size, outline = cloud
    if layers:
        points = torch.cat([points, *[layer[0] for layer in layers]])
        colors = torch.cat([colors, *[layer[1] for layer in layers]])
        source = torch.cat([source, *[layer[2] for layer in layers]])
    device = points.device
    fx, fy, cx, cy = k
    r = torch.from_numpy(np.ascontiguousarray(rotation)).to(device)
    c = torch.from_numpy(np.ascontiguousarray(centre)).to(device)
    cam = (points - c) @ r
    front = cam[:, 2] > 1e-6
    cam, colors, source = cam[front], colors[front], source[front]
    W, H = size
    scale = W / source_size[0]
    px = (cam[:, 0] / cam[:, 2] * fx + cx) * scale
    py = (cam[:, 1] / cam[:, 2] * fy + cy) * scale
    # A point seen closer than in its source view covers more pixels.
    # At least 3x3, so slightly stretched surfaces (oblique floors, a background plate) stay closed.
    radius = torch.clamp(torch.ceil(source / cam[:, 2] * scale * 0.75), 2, 4).to(torch.int64)
    zbuf = torch.full((H * W,), float("inf"), device=device, dtype=torch.float64)
    best = torch.full((H * W,), -1, device=device, dtype=torch.int64)
    order = torch.arange(len(cam), device=device)
    offsets = [(dx, dy) for dy in range(-3, 4) for dx in range(-3, 4)]
    for write in (False, True):
        for dx, dy in offsets:
            sel = radius > max(abs(dx), abs(dy))
            x = torch.floor(px + dx).to(torch.int64)
            y = torch.floor(py + dy).to(torch.int64)
            sel &= (x >= 0) & (x < W) & (y >= 0) & (y < H)
            idx = y[sel] * W + x[sel]
            # A splat's own pixel beats its neighbours' spill at equal depth, keeping detail sharp.
            depth = cam[sel, 2] * (1 + 1e-6 * (abs(dx) + abs(dy)))
            if write:
                hit = depth <= zbuf[idx]
                best[idx[hit]] = order[sel][hit]
            else:
                zbuf.scatter_reduce_(0, idx, depth, reduce="amin")
    image = torch.tensor(BACKGROUND, device=device, dtype=torch.uint8).repeat(H * W, 1)
    covered = best >= 0
    image[covered] = colors[best[covered]]
    image = image.reshape(H, W, 3).cpu().numpy()
    covered = covered.reshape(H, W).cpu().numpy()
    if not fill:
        return image, covered
    rim = (outline - c) @ r
    rim = rim[rim[:, 2] > 1e-6]
    footprint = np.zeros((H, W), np.uint8)
    if len(rim) >= 3:
        import cv2
        polygon = torch.stack([(rim[:, 0] / rim[:, 2] * fx + cx) * scale, (rim[:, 1] / rim[:, 2] * fy + cy) * scale], -1)
        polygon = polygon.clamp(-4 * max(W, H), 5 * max(W, H)).round().to(torch.int32).cpu().numpy()
        cv2.fillPoly(footprint, [polygon], 1)
    return fill_interior_holes(image, covered, footprint.astype(bool)), covered


def push_pull(image, known):
    """Fill unknown pixels from successively coarser averages of the known ones. Large gaps get soft,
    out-of-focus colour instead of the streaks that edge-propagating inpainting drags across them."""
    import cv2
    color = image.astype(np.float32) * known[..., None]
    weight = known.astype(np.float32)
    levels = [(color, weight)]
    while min(color.shape[:2]) > 2 and weight.min() <= 0:
        size = (max(1, color.shape[1] // 2), max(1, color.shape[0] // 2))
        color = cv2.resize(color, size, interpolation=cv2.INTER_AREA)
        weight = cv2.resize(weight, size, interpolation=cv2.INTER_AREA)
        levels.append((color, weight))
    filled = levels[-1][0] / np.maximum(levels[-1][1], 1e-6)[..., None]
    for color, weight in reversed(levels[:-1]):
        coarse = cv2.resize(filled, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_LINEAR)
        alpha = np.clip(weight, 0, 1)[..., None]
        filled = np.where(alpha > 0, color / np.maximum(weight, 1e-6)[..., None], 0) * alpha + coarse * (1 - alpha)
    return np.clip(filled + .5, 0, 255).astype(np.uint8)


# Enclosed gaps up to this fraction of the frame are cracks to fill; larger ones are areas the subject hid.
CRACK_AREA = 0.002


def fill_interior_holes(image, covered, footprint):
    """Fill splat cracks inside the source picture's projected outline. The editing model reads a grey crack
    as a new object (a panel on a lap). A larger gap is background the subject used to hide: filled with
    blurred colour it keeps the subject's outline and comes back as a ghost of the subject, while left grey
    it is completed as plausible new scenery, so it stays grey like everything outside the outline."""
    import cv2
    enclosed = (footprint & ~covered).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(enclosed, connectivity=4)
    small = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] <= CRACK_AREA * enclosed.size) + 1
    cracks = np.isin(labels, small)
    if not cracks.any():
        return image
    image = image.copy()
    image[cracks] = push_pull(image, covered)[cracks]
    return image


class Scene:
    """The Start Image plus the views generated so far, registered together by Depth Anything 3."""

    def __init__(self, model, start, mask=None, plate=None, plate_mask=None):
        """``plate`` is the Start Image with the subject painted out; inside ``plate_mask`` it becomes a
        background layer, so a moving camera sees scenery where the subject stood instead of a
        subject-shaped hole that the editing model would turn into a second subject."""
        self.model = model
        self.images = [start.convert("RGB")]
        self.targets = [(0.0, 0.0, 1.0)]
        self.mask = mask
        self.plates = {}
        self._history = []
        self._solve()
        self.set_plate(0, plate, plate_mask)

    def set_plate(self, index, plate, plate_mask):
        """Give a registered view its background plate (the view with its subject painted out)."""
        if plate is None or plate_mask is None or not plate_mask.any() or plate_mask.all():
            return
        plate = plate.convert("RGB").resize(self.images[0].size)
        depths, _, _ = self.model.reconstruct([self.images[index], plate])
        # Relative to the view's visible scenery, so later solves can rescale it.
        self.plates[index] = (plate, plate_mask, depths[1] / float(np.median(depths[1][~plate_mask])))

    def _solve(self):
        self.depths, self.c2w, self.k = self.model.reconstruct(self.images)
        self.pivot = pivot_point(self.depths[0], self.k[0], self.mask)
        self.measured = [pose_of(pose, self.pivot) for pose in self.c2w]
        self.measured[0] = (0.0, 0.0, 1.0)

    def add(self, image, target):
        """Register a generated view with all earlier ones; returns its measured pose."""
        self._history.append((self.depths, self.c2w, self.k, self.pivot, self.measured))
        self.images.append(image.convert("RGB").resize(self.images[0].size))
        self.targets.append(tuple(target))
        self._solve()
        return self.measured[-1]

    def drop_last(self):
        """Forget the last added view and restore the registration from before it."""
        self.images.pop()
        self.targets.pop()
        self.depths, self.c2w, self.k, self.pivot, self.measured = self._history.pop()

    def _cloud(self, source):
        pose = self.c2w[source]
        near = float((pose[:3, :3].T @ (self.pivot - pose[:3, 3]))[2])
        return point_cloud(self.images[source], expand_far(self.depths[source], near), self.k[source], pose), near

    def coarse(self, target, size):
        """Coarse render of ``target`` (azimuth, elevation, distance) from the best registered view."""
        from PIL import Image
        source = self.source_for(target, size)
        pose = self.c2w[source]
        cloud, near = self._cloud(source)
        layers = ()
        if source in self.plates:
            plate, mask, relative = self.plates[source]
            depth = relative * float(np.median(self.depths[source][~mask]))
            layers = (point_cloud(plate, expand_far(depth, near), self.k[source], pose, only=mask),)
        rotation, centre = orbit(self.pivot, *target)
        image, covered = render(cloud, self.k[0], rotation, centre, size, layers=layers)
        return Image.fromarray(image), source, float(covered.mean())

    def source_for(self, target, size=(256, 256)):
        """The registered view that sees most of ``target``. The Start Image gets a START_BONUS head start:
        it is the cleanest source, and re-rendering a generated view compounds that view's flaws."""
        rotation, centre = orbit(self.pivot, *target)
        width = min(size[0], 256)
        small = (width, max(1, round(size[1] * width / size[0])))

        def score(index):
            if pose_gap(self.measured[index], target) > 120:
                return -1.0
            _, covered = render(self._cloud(index)[0], self.k[0], rotation, centre, small, fill=False)
            return float(covered.mean()) + (START_BONUS if index == 0 else 0.0)

        return max(range(len(self.images)), key=score)
