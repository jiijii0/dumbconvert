"""Conversion engine: a thin wrapper around NegPy's darkroom engine.

All the colour science (film base detection, density normalization, paper
curves, dye models, cast removal...) is NegPy's, in vendor/negpy (GPLv3).
This file only does four things:

  1. `Settings`   - the simple controls SimpleNeg shows.
  2. `to_config`  - translate them into NegPy's WorkspaceConfig.
  3. `preview` / `export` - ask NegPy to render a file.
  4. `auto_wb`    - a small grey-world white balance on top of NegPy's result.

To expose another NegPy control, find its name in
vendor/negpy/negpy/features/*/models.py (or docs/USER_GUIDE.md), add a field
to Settings, and map it in `to_config`.
"""

import sys
import functools
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from negpy.domain.models import ExportConfig, ExportFormat, WorkspaceConfig
from negpy.features.process.models import DEFAULT_CROSSTALK_MATRIX
from negpy.services.assets.crosstalk import CrosstalkProfiles
from negpy.infrastructure.loaders.constants import SUPPORTED_RAW_EXTENSIONS
from negpy.services.rendering.image_processor import ImageProcessor

IMAGE_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"} | {
    e.lower() for e in SUPPORTED_RAW_EXTENSIONS
}

MODES = {"color": "Color Negative", "bw": "B&W Negative"}

# One step of the C/M/Y buttons (0.01) moves NegPy's filtration by this much.
CMY_SCALE = 2.0
# Per-colour strength, so a step of C, M or Y moves the picture equally (measured on four
# white-light scans): the yellow layer has less density range, so the same filter moves it less.
CMY_BALANCE = (1.0, 0.87, 1.40)


# What a white light and a camera's colour filters do to a colour negative, measured: the same
# frames scanned under a narrow-band RGB light and under a high-CRI white light (CineStill's
# sample pairs), fitted in density space. Red is read almost cleanly; green and blue are not.
# NegPy's film matrices describe the film's dyes only, so SimpleNeg uses half of each: the
# film's dyes and this rig correction. Half is deliberate: on the test pairs it came closer to
# the narrow-band scan than the dye matrix alone on every frame (also on frames left out of
# the fit), while the full measured matrix overshot on some.
WHITE_LIGHT_MATRIX = (1.005, -0.010, 0.005,
                      -0.290, 1.074, 0.215,
                      -0.069, -0.452, 1.521)
WHITE_LIGHT_SHARE = 0.5


def white_light_matrix(dyes: tuple) -> tuple:
    """The film's dye-unmix matrix combined with the white-light correction (rows sum to 1)."""
    d = np.array(dyes, dtype=np.float64).reshape(3, 3)
    d = d / d.sum(axis=1, keepdims=True)
    m = (1.0 - WHITE_LIGHT_SHARE) * d + WHITE_LIGHT_SHARE * np.array(WHITE_LIGHT_MATRIX).reshape(3, 3)
    return tuple(float(v) for v in (m / m.sum(axis=1, keepdims=True)).ravel())


def films() -> list[str]:
    """Crosstalk profiles NegPy ships for colour negative film ("Generic C41" first)."""
    return [n for n in CrosstalkProfiles.list_profiles()
            if CrosstalkProfiles.get_process(n) in ("", "Color Negative")]


# ---------------------------------------------------------------------------
# 1. The settings of one photo. Defaults = NegPy's automatic result.
# ---------------------------------------------------------------------------
@dataclass
class Settings:
    # Main panel (1.00 = automatic)
    contrast: float = 1.0
    density: float = 1.0         # higher = darker print
    saturation: float = 1.0
    cyan: float = 1.0            # higher = more cyan (less red)
    magenta: float = 1.0         # higher = more magenta (less green)
    yellow: float = 1.0          # higher = more yellow (less blue)
    grayscale: bool = False
    # Retouch
    # Geometry
    rotation: int = 0            # quarter turns counter-clockwise
    fine_rotation: float = 0.0   # straighten, degrees (-45..45)
    mirror: bool = False
    auto_crop: bool = False
    crop: tuple | None = None    # (x0, y0, x1, y1), 0..1, in the rotated image
    analysis_rect: tuple | None = None   # area NegPy measures the film in (Recalc.)
    # "Advanced" panel (collapsed). Defaults chosen for scanning with a bright WHITE light:
    # a straight conversion (no print-paper look), full dye unmixing, fewer "neutralizing"
    # helpers (README, "Why these defaults").
    dye_separation: float = 1.1   # colour strength between the dye layers: fixed, restores what white light loses
    # Tones, like a darkroom's split-grade printing: -1 soft .. +1 hard (contrast of that end only)
    highlight_tone: float = 0.0
    shadow_tone: float = 0.0
    # Colour casts that only sit in one end of the tonal range (fog in shadows, red highlights)
    shadow_warmth: float = 0.0     # -1 cooler (blue) .. +1 warmer (amber)
    shadow_tint: float = 0.0       # -1 greener .. +1 more magenta
    highlight_warmth: float = 0.0
    highlight_tint: float = 0.0
    # Finish
    glow: float = 0.0              # 0 .. 1: soft glow around bright areas, like a minilab scan
    sharpen: float = 0.25          # 0 .. 1 (NegPy's default)
    color_noise: float = 0.33      # 0 .. 1 = NegPy Chroma Denoise 0 .. 3; 1.0 removes most speckle, NegPy's 2.5 barely more
    # Dust healed by hand: (x, y, size) per spot. x, y are 0..1 in the raw scan (before rotation
    # and crop), size is a diameter in px at NegPy's 1600 px reference
    heal_spots: tuple = ()
    # Areas to clean automatically: each a closed shape, ((x, y), ...) in the raw scan like heal_spots.
    # Inside them NegPy's dust and scratch detector runs; the rest of the picture is left alone.
    heal_areas: tuple = ()
    # Auto crop, once found: the rectangle (0..1, like `crop`) and NegPy's key for the framing it was
    # found on. Kept so every render (draft, full, export) uses the same crop instead of looking again.
    auto_rect: tuple | None = None
    auto_key: str = ""
    # Set automatically, not in the panel
    film: str = "Generic C41"    # crosstalk profile (dye unmix matrix); set from the roll's film stock
    crosstalk: float = 1.0       # 0 = off .. 1 = full matrix. White light mixes the layers: unmix fully
    cast_removal: float = 0.2    # NegPy default 0.5
    skin_protection: float = 0.25   # NegPy default 0.5
    border: float = 0.05         # fraction of each edge ignored when measuring
    # Vignetting fix (same for the whole roll: it belongs to the scanning setup)
    flat_field: str | None = None   # photo of the bare light panel: evens out light falloff and vignetting
    # Roll baseline (set by analyze_roll for frames dropped as a roll; None = per frame)
    roll_floors: tuple | None = None
    roll_ceils: tuple | None = None
    roll_axis: tuple | None = None   # the roll's grey axis: one colour cast correction for the whole roll

    def copy(self) -> "Settings":
        return Settings(**asdict(self))

    @property
    def mode(self) -> str:
        return "bw" if self.grayscale else "color"


# ---------------------------------------------------------------------------
# 2. Settings -> NegPy config
# ---------------------------------------------------------------------------
TONE_GRADE = 35.0   # ISO-R points at the end of a Tone slider
CAST_CC = 0.75      # zone filtration at the end of a cast slider (0.75 = 15 CC)


def to_config(st: Settings, uncropped: bool = False, path: Path | None = None) -> WorkspaceConfig:
    """`uncropped=True` renders the whole frame (used while editing the crop).
    `path` is the scan being rendered: Contrast needs to measure it (see _with_contrast)."""
    flat = {
        "process_mode": MODES[st.mode],
        # NegPy's "Digital" output: an ideal print with no paper dyes, i.e. no print look
        "paper_profile": "neutral",
        # Print Density (exposure time under the enlarger). Auto stays on.
        "density": st.density,
        # ISO-R grade: lower number = more contrast, 115 is normal.
        # With a scan to measure, Contrast is applied afterwards on the frame's own automatic contrast.
        "grade": 115.0 if path is not None else float(np.clip(115.0 - 70.0 * (st.contrast - 1.0), 50.0, 180.0)),
        "saturation": st.saturation,
        "cast_removal_strength": st.cast_removal,
        "skin_protection": st.skin_protection,
        "dye_separation": st.dye_separation,
        "wb_cyan": CMY_SCALE * CMY_BALANCE[0] * (st.cyan - 1.0),
        "wb_magenta": CMY_SCALE * CMY_BALANCE[1] * (st.magenta - 1.0),
        "wb_yellow": CMY_SCALE * CMY_BALANCE[2] * (st.yellow - 1.0),
        # Split grade, ISO-R points: negative = harder, like the main grade
        "highlight_grade": -TONE_GRADE * st.highlight_tone,
        "shadow_grade": -TONE_GRADE * st.shadow_tone,
        # Zone filtration (1.0 = 20 CC). Warmth = yellow plus a little red, tint = magenta
        "shadow_yellow": CAST_CC * st.shadow_warmth,
        "shadow_cyan": -0.35 * CAST_CC * st.shadow_warmth,
        "shadow_magenta": CAST_CC * st.shadow_tint,
        "highlight_yellow": CAST_CC * st.highlight_warmth,
        "highlight_cyan": -0.35 * CAST_CC * st.highlight_warmth,
        "highlight_magenta": CAST_CC * st.highlight_tint,
        "glow_amount": st.glow,
        "sharpen": st.sharpen,
        "chroma_denoise": 3.0 * st.color_noise,
        # A spot is a search area: NegPy repairs only what stands out from the film around it
        "manual_heal_strokes": [([[x, y]], float(size), 0.0, 0.0) for x, y, size in st.heal_spots],
        # NegPy's automatic dust removal looks at the whole picture; everything outside the areas
        # you drew is handed back to it as "leave this alone" (see _outside_strokes)
        "dust_remove": bool(st.heal_areas),
        "dust_exclusion_strokes": _outside_strokes(_as_tuples(st.heal_areas)) if st.heal_areas else [],
        "analysis_buffer": st.border,
        "analysis_rect": st.analysis_rect,
        "rotation": st.rotation % 4,
        "flip_horizontal": st.mirror,
        "fine_rotation": float(st.fine_rotation),
        # straightening leaves empty wedges in the corners: crop them away (unless you cropped)
        "crop_to_valid": abs(st.fine_rotation) > 0.01,
    }
    matrix = None
    if st.mode == "color":
        # The built-in "Generic C41" is a constant in NegPy's code, not a file.
        matrix = CrosstalkProfiles.get_matrix(st.film) or (
            DEFAULT_CROSSTALK_MATRIX if st.film == CrosstalkProfiles.DEFAULT_NAME else None)
    if matrix and st.crosstalk > 0:
        # Unmix the dye layers in density space, before NegPy measures the film.
        flat["crosstalk_strength"] = st.crosstalk
        flat["crosstalk_matrix"] = white_light_matrix(tuple(matrix))
        flat["crosstalk_profile"] = st.film
        flat["crosstalk_process"] = "Color Negative"
    if st.roll_floors and st.roll_ceils:
        # Colour from the whole roll (film base + cast are the same on every frame),
        # brightness range per frame, like a lab scanner reading the whole order first.
        flat["locked_floors"] = tuple(st.roll_floors)
        flat["locked_ceils"] = tuple(st.roll_ceils)
        flat["use_color_average"] = True
        flat["use_luma_average"] = False
        if st.roll_axis and st.mode == "color":
            # Cast removal against the roll's greys, not just this frame's: a frame with no grey
            # in it (a field, a sunset) is balanced like its neighbours, as a lab scanner does.
            flat["use_cast_average"] = True
            flat["locked_neutral_axis"] = st.roll_axis
    profile = flat_profile_id(st)
    if profile:
        flat["apply"] = True          # FlatFieldConfig.apply
        flat["profile_id"] = profile
    if not uncropped:
        if st.crop:
            flat["crop_rect"] = st.crop
        elif st.auto_crop:
            flat["crop_from_auto"] = True
            if st.auto_rect and st.auto_key:
                # NegPy's frozen auto crop: used while the framing still matches its key
                # (a turn, a flip or straightening changes the key, and NegPy looks again)
                flat["crop_rect"] = tuple(st.auto_rect)
                flat["crop_detect_key"] = st.auto_key
    if path is not None and flat["crop_to_valid"] and not (st.crop or st.auto_crop or st.analysis_rect):
        # straightened, no crop: measured inside the film, as before straightening (see _film_area)
        area = _film_area(path, st)
        if area:
            flat["analysis_rect"] = area
    cfg = WorkspaceConfig.from_flat_dict(flat)
    if path is not None and abs(st.contrast - 1.0) > 1e-6:
        cfg = _with_contrast(cfg, path, st)
    return cfg


# ---------------------------------------------------------------------------
# Straightened frames
# ---------------------------------------------------------------------------
# On a frame with no crop, NegPy finds the film in the scan and measures exposure and colour
# inside it, not in the light around it. Straightening leaves empty corners, which NegPy trims
# away (crop_to_valid); to NegPy that is a crop, so it stops looking for the film and measures
# all that is left, the light included: the picture went dark at the first 0.1°. The film is
# still where it was. Its box is found once, on the frame before straightening, and a
# straightened frame is measured in the upright rectangle inside that box as it now lies.
_film_boxes: dict[tuple, tuple | None] = {}
_film_boxes_lock = threading.Lock()


def _film_box(path: Path, st: Settings) -> tuple[tuple, float] | None:
    """NegPy's film box on this frame before straightening: (rect in 0..1 of the turned picture,
    that picture's width / height). None when the film fills the scan. Cached per turn and flip."""
    key = (source_hash(path), flat_profile_id(st), st.rotation % 4, st.mirror)
    with _film_boxes_lock:
        if key in _film_boxes:
            return _film_boxes[key]
    plain = st.copy()
    plain.fine_rotation, plain.crop, plain.auto_crop, plain.analysis_rect, plain.contrast = 0.0, None, False, None, 1.0
    cfg = to_config(plain)
    ip = _processor()
    src, _, _ = ip._load_source_f32(str(path), cfg, fast_decode=True)
    step = max(1, int(np.ceil(max(src.shape[:2]) / 1200)))   # finding the film needs no full resolution
    small = np.ascontiguousarray(src[::step, ::step])
    _, metrics = ip.run_pipeline(small, cfg, source_hash(path) + "|film", max(small.shape[:2]),
                                 prefer_gpu=False, readback_metrics=True)
    found = metrics.get("meter_resolved")
    h, w = small.shape[:2]
    box = None if found is None or found.rect is None else (
        tuple(float(v) for v in found.rect), (h / w) if st.rotation % 2 else (w / h))
    with _film_boxes_lock:
        if len(_film_boxes) > 2000:      # a long session: forget the oldest
            for k in list(_film_boxes)[:500]:
                del _film_boxes[k]
        _film_boxes[key] = box
    return box


def _film_area(path: Path, st: Settings) -> tuple | None:
    """Where a straightened frame with no crop is measured: inside the film, less the border NegPy
    leaves out at every edge (its analysis buffer, which a given area replaces). None: no film box."""
    box = _film_box(path, st)
    if box is None:
        return None
    x0, y0, x1, y1 = _inside_turned(box[0], box[1], st.fine_rotation)
    b = min(max(st.border, 0.0), 0.3)
    dx, dy = b * (x1 - x0), b * (y1 - y0)
    return (x0 + dx, y0 + dy, x1 - dx, y1 - dy)


def _inside_turned(rect: tuple, aspect: float, angle: float) -> tuple:
    """The largest upright rect inside `rect` once the picture (width / height `aspect`) is
    straightened by `angle` degrees: `rect` turns with the picture, about the picture's centre."""
    import negpy_core

    w, h = 1000.0 * aspect, 1000.0
    x0, y0, x1, y1 = rect
    fw, fh = (x1 - x0) * w, (y1 - y0) * h
    turn = negpy_core.rotation_matrix((w / 2.0, h / 2.0), angle, 1.0)    # as NegPy's fine rotation turns it
    cx, cy = turn @ np.array([(x0 + x1) / 2 * w, (y0 + y1) / 2 * h, 1.0])
    a = min(abs(np.radians(angle)), np.pi / 4)
    s, c = np.sin(a), np.cos(a)
    long_side, short_side = max(fw, fh), min(fw, fh)
    if short_side <= 2 * s * c * long_side:      # turned so far that the short side alone sets the size
        half = short_side / 2
        iw, ih = (half / s, half / c) if fw >= fh else (half / c, half / s)
    else:
        iw, ih = (fw * c - fh * s) / (c * c - s * s), (fh * c - fw * s) / (c * c - s * s)
    return (float(max(0.0, (cx - iw / 2) / w)), float(max(0.0, (cy - ih / 2) / h)),
            float(min(1.0, (cx + iw / 2) / w)), float(min(1.0, (cy + ih / 2) / h)))


# ---------------------------------------------------------------------------
# Cleaning areas
# ---------------------------------------------------------------------------
# NegPy can find dust and scratches by itself, but only on the whole picture, with a brush to
# protect film it mistook for dust. An area to clean is that turned inside out: the detector
# runs everywhere and everything outside your shapes is protected. The protection is painted
# as thin horizontal bands, row by row, each ending where a shape begins.
HEAL_AREA_ROWS = 240


def _as_tuples(v):
    return tuple(_as_tuples(x) for x in v) if isinstance(v, (list, tuple)) else v


@functools.lru_cache(maxsize=64)
def _outside_strokes(areas: tuple) -> list:
    """Protection bands (NegPy's dust_exclusion_strokes) covering the scan outside `areas`."""
    size = 3200.0 / HEAL_AREA_ROWS * 0.9          # band width: neighbouring rows overlap on any frame shape
    cap = size / 3200.0 * 1.5                     # a band's round end reaches this far: stop short of the shape
    strokes = []
    for row in range(HEAL_AREA_ROWS):
        y = (row + 0.5) / HEAL_AREA_ROWS
        inside = []                               # the stretches of this row that lie in a shape
        for poly in areas:
            xs = []
            for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
                if (y0 <= y) != (y1 <= y):
                    xs.append(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
            xs.sort()
            inside += list(zip(xs[0::2], xs[1::2]))
        inside.sort()
        x, gaps = 0.0, []
        for a, b in inside:
            if a > x:
                gaps.append((x, a))
            x = max(x, b)
        gaps.append((x, 1.0))
        for a, b in gaps:
            a = a + cap if a > 0.0 else -cap      # the frame's own edge needs no margin
            b = b - cap if b < 1.0 else 1.0 + cap
            if b > a:
                strokes.append(([[round(a, 5), round(y, 5)], [round(b, 5), round(y, 5)]], size))
    return strokes


# ---------------------------------------------------------------------------
# Contrast
# ---------------------------------------------------------------------------
# NegPy picks each frame's contrast itself (Auto Grade) and never prints softer than what keeps
# the shadows black, so asking it for a softer grade does nothing on many frames. The Contrast
# fader instead scales the contrast NegPy chose: we read what the automatic render used (its
# slope and its shadow/highlight holds), switch the automatic off, and print the same curve
# with the slope multiplied. Every step is the same change on every frame, up and down.
CONTRAST_GAIN = 0.6      # one fader step (0.05) changes the slope by about 3 %
# Settings that change the picture's look but not what the film measures (the meter ignores them)
_LOOK_FIELDS = ("contrast", "density", "saturation", "cyan", "magenta", "yellow", "highlight_tone", "shadow_tone",
                "shadow_warmth", "shadow_tint", "highlight_warmth", "highlight_tint", "glow", "sharpen",
                "color_noise", "heal_spots", "heal_areas")
_meters: dict[tuple, dict] = {}
_meters_lock = threading.Lock()


def _meter(path: Path, st: Settings) -> dict:
    """What NegPy's light meter reads on this frame (cached: it only changes with the framing,
    the film and the roll, not with the look)."""
    neutral, defaults = st.copy(), Settings()
    for f in _LOOK_FIELDS:
        setattr(neutral, f, getattr(defaults, f))
    key = (source_hash(path), flat_profile_id(neutral), repr(neutral))
    with _meters_lock:
        if key in _meters:
            return _meters[key]
    cfg = to_config(neutral, path=path)
    ip = _processor()
    src, _, _ = ip._load_source_f32(str(path), cfg, fast_decode=True)
    step = max(1, int(np.ceil(max(src.shape[:2]) / 1200)))   # measuring needs no full resolution
    small = np.ascontiguousarray(src[::step, ::step])
    _, metrics = ip.run_pipeline(small, cfg, source_hash(path) + "|meter", max(small.shape[:2]),
                                 prefer_gpu=False, readback_metrics=True)
    keep = {k: metrics.get(k) for k in ("norm_density_range", "textural_range", "shadow_point",
                                        "highlight_point", "metered_anchor")}
    with _meters_lock:
        if len(_meters) > 2000:          # a long session: forget the oldest
            for k in list(_meters)[:500]:
                del _meters[k]
        _meters[key] = keep
    return keep


def _with_contrast(cfg: WorkspaceConfig, path: Path, st: Settings) -> WorkspaceConfig:
    from dataclasses import replace

    from negpy.features.exposure import auto_sliders
    from negpy.features.exposure.logic import slope_to_grade
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS
    from negpy.features.exposure.papers import slope_gain

    m = _meter(path, st)
    ex, mode = cfg.exposure, cfg.process.process_mode
    slope = auto_sliders._auto_slope(ex, mode, m)
    if slope is None:                    # nothing measured: plain paper grades
        return replace(cfg, exposure=replace(ex, grade=float(np.clip(115.0 - 70.0 * (st.contrast - 1.0), 50.0, 180.0))))
    held = auto_sliders.print_shown_values(ex, mode, m)       # the holds the automatic applied
    target = float(np.clip(slope * np.exp(CONTRAST_GAIN * (st.contrast - 1.0)),
                           EXPOSURE_CONSTANTS["slope_min"], EXPOSURE_CONSTANTS["slope_max"]))
    paper = auto_sliders._paper(ex, mode)[0]
    grade = slope_to_grade(target, m["norm_density_range"], slope_gain(paper))
    return replace(cfg, exposure=replace(
        ex, auto_normalize_contrast=False, grade=float(grade),
        highlight_density=float(held.get("highlight_density", ex.highlight_density)),
        shadow_density=float(held.get("shadow_density", ex.shadow_density))))


# ---------------------------------------------------------------------------
# Scan check
# ---------------------------------------------------------------------------
_scan_notes: dict[str, str] = {}


def scan_note(path: Path) -> str:
    """A short warning when the scan itself limits the result, else "". Two cases only:
    the film base is blown out (nothing can bring that back), or the scan is so dark that most
    of the camera's range is unused (noisier colour, above all in the blue channel)."""
    key = source_hash(path)
    if key in _scan_notes:
        return _scan_notes[key]
    ip = _processor()
    src, _, _ = ip._load_source_f32(str(path), to_config(Settings()), fast_decode=True)
    px = src[::8, ::8].reshape(-1, 3)
    ip.release_source_cache()
    clipped = float((px >= 0.99).mean(axis=0).max())      # share of pixels at the sensor's limit
    brightest = float(np.percentile(px, 99.5, axis=0).max())
    note = ("scan overexposed: use a shorter shutter" if clipped > 0.01 else
            "scan very dark: a longer shutter gives cleaner colour" if brightest < 0.2 else "")
    _scan_notes[key] = note
    return note


# ---------------------------------------------------------------------------
# Vignetting / flat field
# ---------------------------------------------------------------------------
# NegPy's flat field multiplies the *linear scan* by a gain map before anything
# else runs (before film-base detection too, so the corners no longer fool it).
# NegPy asks a "provider" for the gain map of a profile id; we register our own
# provider for ids "file:<path>:<mtime>": a gain map measured from a flat-field photo.
from negpy.features.flatfield import logic as flatfield

_flat_gains: dict[str, np.ndarray] = {}
_flat_lock = threading.Lock()


def flat_profile_id(st: Settings) -> str:
    if st.flat_field:
        p = Path(st.flat_field)
        if p.exists():
            return f"file:{p}:{p.stat().st_mtime_ns}"
    return ""



def load_linear(path: Path) -> np.ndarray:
    """Decode a flat-field photo to linear RGB float32 (0..1), oriented like the scans."""
    ext = path.suffix.lower()
    if ext in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        import tifffile
        from PIL import Image

        img = tifffile.imread(str(path)) if ext in {".tif", ".tiff"} else np.asarray(Image.open(path).convert("RGB"))
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = img[..., :3]
        if img.dtype == np.uint8:   # 8-bit files are sRGB encoded, like NegPy assumes
            v = img.astype(np.float32) / 255.0
            return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)
        return (img.astype(np.float32) / np.iinfo(img.dtype).max).astype(np.float32)

    import rawpy

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16,
                              use_camera_wb=False, user_wb=[1, 1, 1, 1], half_size=True)
    return rgb.astype(np.float32) / 65535.0


def _gain_provider(profile_id: str):
    with _flat_lock:
        if profile_id not in _flat_gains:
            kind, rest = profile_id.split(":", 1)
            if kind == "file":
                path = Path(rest.rsplit(":", 1)[0])
                _flat_gains[profile_id] = flatfield.compute_gain(load_linear(path))
            else:
                return None
        gain = _flat_gains[profile_id]
    # sensor_layout=False: the map is already oriented like the decoded scan
    return flatfield.GainEntry(gain, flatfield.gain_token(gain), False)


flatfield.set_gain_provider(_gain_provider)


# ---------------------------------------------------------------------------
# 3. Rendering
# ---------------------------------------------------------------------------
# An ImageProcessor keeps caches (the decoded scan above all) and is not thread-safe:
# one per thread. Keyed by thread id, not threading.local: Qt's pool threads are not
# Python threads, so Python drops their threading.local data after every job.
_processors: dict[int, ImageProcessor] = {}
_processors_lock = threading.Lock()


def _processor() -> ImageProcessor:
    tid = threading.get_ident()
    with _processors_lock:
        ip = _processors.get(tid)
        if ip is None:
            ip = _processors[tid] = ImageProcessor(use_gpu=False)
    return ip


def source_hash(path: Path) -> str:
    """NegPy caches the decoded scan under this key; changes when the file does."""
    s = path.stat()
    return f"{path}:{s.st_mtime_ns}:{s.st_size}"


def preview(path: Path, st: Settings, long_edge: int, uncropped: bool = False,
            keep_source: bool = False) -> np.ndarray:
    """Render `path` to an sRGB uint8 array, long edge about `long_edge` px.

    `keep_source=True` keeps the decoded scan in this thread's memory, so the next render
    of the same file skips decoding (the live preview); others release it.
    """
    arr = _processor().render_display_array(
        str(path), to_config(st, uncropped, path), source_hash(path), long_edge,
        prefer_gpu=False, fast_decode=True, keep_source=keep_source,
    )
    if arr is None:
        raise RuntimeError(f"NegPy could not render {path.name}")
    return arr



def preview_full(path: Path, st: Settings) -> np.ndarray:
    """The picture at full resolution (a full raw decode, no half-size shortcut): for zooming in.
    Keeps the decode in this thread's memory for the next zoomed render; release_cache() frees it."""
    arr = _processor().render_display_array(
        str(path), to_config(st, path=path), source_hash(path), 1_000_000,
        prefer_gpu=False, fast_decode=False, keep_source=True,
    )
    if arr is None:
        raise RuntimeError(f"NegPy could not render {path.name}")
    return arr


def source_long_edge(path: Path) -> int:
    """The scan's long edge in pixels, read from the file header (fast): where 100% zoom is."""
    try:
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            from PIL import Image
            with Image.open(path) as im:
                return max(im.size)
        import rawpy
        with rawpy.imread(str(path)) as raw:
            return max(raw.sizes.width, raw.sizes.height)
    except Exception:
        return 0


def release_cache():
    """Free this thread's decoded scan (a full-resolution one is large)."""
    _processor().release_source_cache()


# Export formats: key -> (menu label, file extension)
EXPORT_FORMATS = {
    "jpg":  ("JPEG", "jpg"),
    "tif":  ("TIFF (16-bit)", "tif"),
    "png":  ("PNG (16-bit)", "png"),
    "heic": ("HEIC (10-bit)", "heic"),
    "avif": ("AVIF", "avif"),
    "jxl":  ("JPEG XL", "jxl"),
    "webp": ("WebP", "webp"),
    "dng":  ("DNG (compact, 14-bit, recommended)", "dng"),
    "dng16": ("DNG (lossless, 16-bit, large files)", "dng"),
}


@dataclass
class ExportOptions:
    fmt: str = "jpg"                 # key of EXPORT_FORMATS
    size: str = "original"           # "original" | "long_edge" | "print"
    long_edge: int = 2048            # pixels, for size == "long_edge"
    print_cm: float = 30.0           # long side in cm, for size == "print"
    dpi: int = 300
    color_space: str = "sRGB"        # "sRGB" | "P3 D65" | "Adobe RGB"
    pattern: str = "{name}"          # file name pattern, see file_name()
    overwrite: bool = False          # False = keep both (add a number)


COLOR_SPACES = {"sRGB": "sRGB (sharing, web)", "P3 D65": "Display P3 (Apple screens)",
                "Adobe RGB": "Adobe RGB (printing, editing)"}
NAME_TOKENS = "{name} original file name · {roll} roll name · {film} film stock · {frame} frame number · {camera}"


def file_name(pattern: str, tokens: dict) -> str:
    """Fill a pattern like "{roll}_{film}_{frame}" and make it safe as a file name."""
    import re

    class Missing(dict):
        def __missing__(self, key):
            return ""
    try:
        name = pattern.format_map(Missing(tokens)) if pattern.strip() else tokens.get("name", "image")
    except (ValueError, IndexError, KeyError, AttributeError, TypeError):
        name = re.sub(r"[{}]", "", pattern)          # a lone "{" or a half-typed token: keep the words
    name = re.sub(r"[\\/:*?\"<>|]+", "-", name)          # characters not allowed in file names
    name = re.sub(r"\s+", " ", name).strip(" ._-")
    return name or tokens.get("name", "image")


# Files written during this run of the app. Exports run on several threads at once: two frames
# whose pattern gives the same name must not both see "free" and write over each other.
_taken: set[Path] = set()
_taken_lock = threading.Lock()


def unique_path(path: Path, overwrite: bool) -> Path:
    """Where to write: `path`, or "name (2).ext" when it is taken. With `overwrite`, a file from
    an earlier export is replaced, but never one written by the export in progress."""
    with _taken_lock:
        candidate, n = path, 2
        while candidate in _taken or (not overwrite and candidate.exists()):
            candidate = path.with_name(f"{path.stem} ({n}){path.suffix}")
            n += 1
        _taken.add(candidate)
        return candidate


def new_export_batch() -> None:
    """Call before each export: names used by earlier exports may be reused (or overwritten)."""
    with _taken_lock:
        _taken.clear()


def _export_config(fmt: ExportFormat, opts: ExportOptions, color_space: str | None = None) -> ExportConfig:
    from negpy.domain.models import ExportResolutionMode

    mode = {"original": ExportResolutionMode.ORIGINAL, "long_edge": ExportResolutionMode.TARGET_PX,
            "print": ExportResolutionMode.PRINT}[opts.size]
    return ExportConfig(export_fmt=fmt, export_bit_depth=16, jpeg_quality=95, webp_quality=92,
                        jxl_lossless=False, jxl_distance=1.0,
                        export_color_space=color_space or opts.color_space,
                        export_resolution_mode=mode, export_target_long_edge_px=int(opts.long_edge),
                        export_print_size=float(opts.print_cm), export_dpi=int(opts.dpi))


def _negpy_export(path: Path, st: Settings, fmt: ExportFormat, opts: ExportOptions,
                  color_space: str | None = None) -> bytes:
    """Full-resolution render through NegPy, encoded by NegPy (crop/rotation/size applied)."""
    ip = _processor()
    try:
        data, _ = ip.process_export(str(path), to_config(st, path=path), _export_config(fmt, opts, color_space),
                                    source_hash(path), prefer_gpu=False)
    finally:
        ip.release_source_cache()        # a full-resolution decode: never keep it
    if not data:
        raise RuntimeError(f"NegPy could not export {path.name}")
    return data


def _render16(path: Path, st: Settings, opts: ExportOptions, color_space: str | None = None) -> np.ndarray:
    """The finished picture as a 16-bit RGB array, in `color_space` (encoded, not linear)."""
    import io

    import tifffile

    img = tifffile.imread(io.BytesIO(_negpy_export(path, st, ExportFormat.TIFF, opts, color_space)))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return np.ascontiguousarray(img[..., :3]).astype(np.uint16)


def _icc(color_space: str) -> bytes | None:
    """ICC profile bytes for a colour space, from NegPy (so the file's colours are tagged correctly)."""
    try:
        return _processor()._get_target_icc_bytes(color_space, None)
    except Exception:
        return _srgb_icc() if color_space == "sRGB" else None


def _srgb_icc() -> bytes:
    from PIL import ImageCms

    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _print_resolution(opts: ExportOptions | None):
    """The dpi to write into the file: only "Print size" asks for one."""
    if opts is None or opts.size != "print":
        return None
    from negpy.services.export.encoders import Resolution

    return Resolution.from_dpi(int(opts.dpi))


def _exif_xmp(metadata, path: Path, opts: ExportOptions | None = None) -> tuple[bytes | None, bytes | None]:
    """EXIF + XMP bytes for formats NegPy cannot write metadata into itself."""
    if metadata is None:
        return None, None
    from negpy.features.metadata.writer import export_embed_plan

    plan = export_embed_plan(metadata, None, str(path), resolution=_print_resolution(opts))
    return (plan[0], plan[1]) if plan else (None, None)


def _png_with_metadata(png: bytes, exif: bytes | None, xmp: bytes | None) -> bytes:
    """Add EXIF and XMP to a finished PNG as chunks of their own; the pixels are not touched.
    (NegPy's own rewrite re-saves the picture through Pillow, which turns a 16-bit PNG into 8-bit.)"""
    from negpy.services.export.encoders import _png_chunk

    chunks = b""
    if exif:      # eXIf holds the bare TIFF structure, without JPEG's "Exif" header
        chunks += _png_chunk(b"eXIf", exif[6:] if exif.startswith(b"Exif\x00\x00") else exif)
    if xmp:
        chunks += _png_chunk(b"iTXt", b"XML:com.adobe.xmp\x00\x00\x00\x00\x00" + xmp)
    cut = png.index(b"IDAT") - 4                      # they must come before the picture data
    return png[:cut] + chunks + png[cut:]


# Memory while exporting. A full-resolution render needs about 75 bytes per pixel of the scan
# at its peak (measured: 1.9 GB for 26 MP, 4.4 GB for 61 MP). Exports run in parallel only while
# their estimated total stays inside a share of the Mac's memory; otherwise they take turns.
EXPORT_BYTES_PER_PIXEL = 75
EXPORT_MEMORY_SHARE = 0.4


def _ram_bytes() -> float:
    import os
    try:
        if sys.platform == "win32":               # Windows has no sysconf: ask the system directly
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = MemoryStatus(dwLength=ctypes.sizeof(MemoryStatus))
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return float(status.ullTotalPhys)
            return 8 * 2.0**30
        return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 8 * 2.0**30


class _MemoryBudget:
    def __init__(self, total: float):
        self.total, self.used, self.cond = total, 0.0, threading.Condition()

    def __call__(self, cost: float):
        budget = self

        class _Hold:
            def __enter__(self):
                with budget.cond:       # one export always runs, however large
                    budget.cond.wait_for(lambda: budget.used == 0 or budget.used + cost <= budget.total)
                    budget.used += cost

            def __exit__(self, *exc):
                with budget.cond:
                    budget.used -= cost
                    budget.cond.notify_all()
        return _Hold()


_export_memory = _MemoryBudget(EXPORT_MEMORY_SHARE * _ram_bytes())


def source_pixels(path: Path) -> int:
    """Pixels in the scan, from the file header (0 if unknown)."""
    try:
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            from PIL import Image
            with Image.open(path) as im:
                return im.size[0] * im.size[1]
        import rawpy
        with rawpy.imread(str(path)) as raw:
            return raw.sizes.width * raw.sizes.height
    except Exception:
        return 0


def export(path: Path, out_dir: Path, st: Settings, opts: ExportOptions | str = "jpg",
           metadata=None, tokens: dict | None = None) -> Path:
    """Convert one file and save it (see _export); waits its turn when memory is busy."""
    with _export_memory((source_pixels(path) or 30_000_000) * EXPORT_BYTES_PER_PIXEL):
        return _export(path, out_dir, st, opts, metadata, tokens)


def _export(path: Path, out_dir: Path, st: Settings, opts: ExportOptions | str = "jpg",
            metadata=None, tokens: dict | None = None) -> Path:
    """Convert one file at full quality and save it according to `opts`.

    `metadata` (from metadata_from_exif) is written into the file (EXIF/XMP).
    `tokens` fill the file name pattern (see file_name); {name} defaults to the scan's name.
    """
    if isinstance(opts, str):
        opts = ExportOptions(fmt=opts)
    fmt = opts.fmt
    tokens = {"name": path.stem, **(tokens or {})}
    stem = file_name(opts.pattern, tokens)
    if fmt == "dng16":
        stem += "_lossless"                     # the two DNG kinds must not overwrite each other
    out_dir.mkdir(parents=True, exist_ok=True)
    out = unique_path(out_dir / f"{stem}.{EXPORT_FORMATS[fmt][1]}", opts.overwrite)

    if fmt in ("dng", "dng16"):
        write_dng(out, path, st, metadata, mosaic=(fmt == "dng"), opts=opts)
        return out

    if fmt in ("heic", "avif"):
        img16 = _render16(path, st, opts)
        icc = _icc(opts.color_space)
        exif, xmp = _exif_xmp(metadata, path, opts)
        if fmt == "heic":
            import pillow_heif

            # 16-bit data is stored as 10-bit HEIC: smooth skies, still small files
            heif = pillow_heif.from_bytes(mode="RGB;16", size=(img16.shape[1], img16.shape[0]), data=img16.tobytes())
            for key, val in (("icc_profile", icc), ("exif", exif), ("xmp", xmp)):
                if val:
                    heif.info[key] = val
            heif.save(out, quality=90)
        else:
            from PIL import Image

            kw = {"quality": 85}
            for key, val in (("icc_profile", icc), ("exif", exif), ("xmp", xmp)):
                if val:
                    kw[key] = val
            Image.fromarray((img16 >> 8).astype(np.uint8)).save(out, **kw)
        return out

    native = {"jpg": ExportFormat.JPEG, "tif": ExportFormat.TIFF, "png": ExportFormat.PNG,
              "jxl": ExportFormat.JXL, "webp": ExportFormat.WEBP}[fmt]
    # JPEG XL has no Adobe RGB: Display P3, the widest space it can name, is used instead
    space = "P3 D65" if fmt == "jxl" and opts.color_space == "Adobe RGB" else None
    data = _negpy_export(path, st, native, opts, space)
    if metadata is not None and fmt == "png":
        data = _png_with_metadata(data, *_exif_xmp(metadata, path, opts))
    elif metadata is not None:
        from negpy.features.metadata.writer import embed_metadata

        # source_exif=None: the scan's EXIF names the *scanning* camera, not the one that shot the film
        # resolution: writing the EXIF block would otherwise drop the dpi of a "Print size" export
        data = embed_metadata(data, metadata, None, resolution=_print_resolution(opts))
    out.write_bytes(data)
    return out


# ---------------------------------------------------------------------------
# DNG writer (LinearRaw)
# ---------------------------------------------------------------------------
# A "LinearRaw" DNG holds full RGB pixels (no sensor mosaic) in *linear light*,
# plus a matrix telling the raw editor what colour space they are in. Lightroom
# then treats the converted picture like a raw file: white balance, exposure and
# highlight recovery work on 16-bit linear data instead of a finished JPEG.

# Adobe RGB (1998) primaries, D65 white: linear RGB -> CIE XYZ
_ADOBE_TO_XYZ = np.array([[0.5767309, 0.1855540, 0.1881852],
                          [0.2973769, 0.6273491, 0.0752741],
                          [0.0270343, 0.0706872, 0.9911085]])
_ADOBE_GAMMA = 563 / 256   # Adobe RGB transfer curve (a pure power)


def _srational(values, den: int = 10000) -> list[int]:
    out = []
    for v in values:
        out += [int(round(float(v) * den)), den]
    return out


def write_dng(out: Path, path: Path, st: Settings, metadata=None, mosaic: bool = True,
              opts: ExportOptions | None = None) -> None:
    """Write the converted picture as a DNG a raw editor (Lightroom, Capture One, darktable...) can edit like a raw file.

    mosaic=True  (raw-like): one 14-bit value per pixel in an RGGB colour-filter
                 pattern, like a camera raw; the raw editor demosaics it again. About the
                 size of the original raw, near-lossless (fine colour edges soften a little).
    mosaic=False (lossless RGB): all three colours per pixel, exact 16-bit linear
                 data in a LinearRaw DNG. Exact, but about 3x larger.
    Both: linear light in Adobe RGB primaries, lossless JPEG compression.
    """
    import tifffile

    # always Adobe RGB: the DNG's colour matrix below describes Adobe RGB primaries
    img = _render16(path, st, opts or ExportOptions(), "Adobe RGB").astype(np.float64) / 65535.0
    linear = np.clip(img, 0, 1) ** _ADOBE_GAMMA                       # undo the transfer curve

    xyz_to_cam = np.linalg.inv(_ADOBE_TO_XYZ)                          # DNG ColorMatrix1: XYZ -> "camera" (here: linear Adobe RGB)
    camera = (getattr(metadata, "camera_model", "") or "").strip() if metadata else ""
    film = (getattr(metadata, "film", "") or "") if metadata else ""
    _, xmp = _exif_xmp(metadata, path)
    white = 16383 if mosaic else 65535

    tags = [
        (50706, "B", 4, (1, 4, 0, 0), True),          # DNGVersion 1.4
        (50707, "B", 4, (1, 1, 0, 0), True),          # DNGBackwardVersion
        (50708, "s", 0, "SimpleNeg", True),            # UniqueCameraModel (what the profile belongs to)
        (50721, 10, 9, _srational(xyz_to_cam.ravel()), True),   # ColorMatrix1 (10 = SRATIONAL)
        (50778, "H", 1, 21, True),                     # CalibrationIlluminant1 = D65
        (50728, 5, 3, _srational((1, 1, 1)), True),    # AsShotNeutral (5 = RATIONAL): white = (1,1,1)
        (50730, 10, 1, _srational((0,)), True),        # BaselineExposure 0
        (50936, "s", 0, "SimpleNeg Linear", True),     # ProfileName
        (50940, "f", 4, (0.0, 0.0, 1.0, 1.0), True),   # ProfileToneCurve: straight line, no extra contrast
        (50941, "I", 1, 0, True),                      # ProfileEmbedPolicy: allow copying
    ]
    make = (getattr(metadata, "camera_make", "") or "").strip() if metadata else ""
    if make:
        tags.append((271, "s", 0, make, True))        # Make
    if camera:
        tags.append((272, "s", 0, camera, True))      # Model: the camera that shot the film
    if film:
        tags.append((270, "s", 0, f"Film: {film}", True))  # ImageDescription
    if xmp:
        tags.append((700, "B", len(xmp), xmp, True))   # XMP packet (film, roll, frame...)

    common = dict(subfiletype=0, metadata=None, tile=(256, 256), compression="jpeg",
                  software="SimpleNeg (NegPy engine)")
    if mosaic:
        h, w = linear.shape[0] // 2 * 2, linear.shape[1] // 2 * 2    # whole 2x2 pattern cells
        lin = linear[:h, :w]
        cfa = np.empty((h, w), np.float64)              # RGGB: keep one colour per pixel
        cfa[0::2, 0::2] = lin[0::2, 0::2, 0]
        cfa[0::2, 1::2] = lin[0::2, 1::2, 1]
        cfa[1::2, 0::2] = lin[1::2, 0::2, 1]
        cfa[1::2, 1::2] = lin[1::2, 1::2, 2]
        data = np.round(cfa * white).astype(np.uint16)
        tags += [
            (50717, "I", 1, white, True),              # WhiteLevel (14-bit)
            (33421, "H", 2, (2, 2), True),             # CFARepeatPatternDim
            (33422, "B", 4, (0, 1, 1, 2), True),       # CFAPattern: R G / G B
            (50710, "B", 3, (0, 1, 2), True),          # CFAPlaneColor: R, G, B
            (50711, "H", 1, 1, True),                  # CFALayout: rectangular
        ]
        tifffile.imwrite(out, data, photometric=32803, extratags=tags,          # 32803 = CFA
                         compressionargs={"lossless": True, "bitspersample": 14}, **common)
    else:
        data = (linear * white + 0.5).astype(np.uint16)
        tags.append((50717, "I", 3, (white,) * 3, True))   # WhiteLevel
        tifffile.imwrite(out, data, photometric=34892, planarconfig="contig", extratags=tags,  # 34892 = LinearRaw
                         compressionargs={"lossless": True, "bitspersample": 16}, **common)   # else tagged 12-bit
    exif, _ = _exif_xmp(metadata, path)
    if exif:
        # date, lens and the rest of the EXIF go in an Exif IFD of their own, where the standard
        # puts them (tifffile can only write tags into the main one)
        from negpy.features.metadata.writer import attach_exif_ifd

        raw = out.read_bytes()
        done = attach_exif_ifd(raw, exif)
        if done is not raw:
            out.write_bytes(done)


# ---------------------------------------------------------------------------
# 4. Auto white balance (grey world)
# ---------------------------------------------------------------------------
def auto_wb(path: Path, st: Settings, point: tuple[float, float] | None = None) -> tuple[float, float, float]:
    """Find C/M/Y values that make the picture's neutral tones grey.

    Idea: the least colourful pixels in a photo (walls, clouds, asphalt,
    shadows) are probably grey in reality. Measure their average cast and
    nudge the filtration against it. Strongly coloured pixels (sky, grass,
    skin) are ignored so they don't get "corrected" away. Repeat 3x.

    With `point` (x, y from 0 to 1 in the picture) it neutralizes the small area you clicked
    instead: the white-balance picker. Click something that should be grey or white.
    """
    trial = st.copy()
    for _ in range(8 if point is not None else 6):
        img = preview(path, trial, 400, keep_source=True).astype(np.float32) / 255.0
        if point is not None:            # a small patch around the click (7x7 px of 400)
            h, w = img.shape[:2]
            cx, cy = int(point[0] * (w - 1)), int(point[1] * (h - 1))
            img = img[max(cy - 3, 0):cy + 4, max(cx - 3, 0):cx + 4]
        px = img.reshape(-1, 3)
        lum = px.mean(axis=1)
        if point is None:
            px = px[(lum > 0.08) & (lum < 0.92)]        # skip clipped blacks/whites
            if len(px) < 200:
                break
        logs = np.log(px + 1e-3)
        cast = logs - logs.mean(axis=1, keepdims=True)   # per-pixel colour offset
        chroma = np.abs(cast).max(axis=1)
        neutral = cast if point is not None else cast[chroma <= np.percentile(chroma, 30)]
        error = neutral.mean(axis=0)                # >0 = channel too strong
        # More cyan removes red, more magenta removes green, more yellow removes blue.
        # Raising a filter by 1.0 lowers its channel by roughly 0.6 in log terms.
        step = np.clip(0.8 * error / 0.6 / CMY_SCALE / np.array(CMY_BALANCE), -0.08, 0.08)
        if point is not None:
            step = np.clip(0.8 * error / 0.6 / CMY_SCALE / np.array(CMY_BALANCE), -0.15, 0.15)
        trial.cyan += float(step[0])
        trial.magenta += float(step[1])
        trial.yellow += float(step[2])
    _processor().release_source_cache()
    lo, hi = (0.5, 1.5) if point is not None else (0.7, 1.3)   # a picked spot is trusted further
    clamp = lambda v: round(float(np.clip(v, lo, hi)), 2)
    return clamp(trial.cyan), clamp(trial.magenta), clamp(trial.yellow)


# ---------------------------------------------------------------------------
# 5. Rolls: shared analysis, film stock catalogue, metadata
# ---------------------------------------------------------------------------
import json

from negpy.features.exposure.normalization import pool_frame_bounds

# NegPy's film catalogue: next to the code when run from source, at the bundle root in the packaged app
_GEAR_DIR = (Path(sys._MEIPASS) / "gear" if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")
             else Path(__file__).resolve().parent.parent / "vendor" / "negpy" / "gear")


def frame_bounds(path: Path, st: Settings) -> tuple[tuple, tuple, tuple | None]:
    """Measure one frame exactly as NegPy's renderer does: its per-channel film floors and
    ceils, and its grey axis (where its neutral tones sit; None when it has none)."""
    per_frame = st.copy()
    per_frame.roll_floors = per_frame.roll_ceils = per_frame.roll_axis = None
    per_frame.contrast = 1.0             # the film measures the same at any contrast
    cfg = to_config(per_frame, uncropped=False, path=path)
    ip = _processor()
    src, _, _ = ip._load_source_f32(str(path), cfg, fast_decode=True)
    step = max(1, int(np.ceil(max(src.shape[:2]) / 1200)))   # measuring needs no full resolution
    src = np.ascontiguousarray(src[::step, ::step])
    ip.release_source_cache()
    _, metrics = ip.run_pipeline(src, cfg, source_hash(path) + "|roll-meter", max(src.shape[:2]),
                                 prefer_gpu=False, readback_metrics=True)
    b = metrics["log_bounds"]
    return tuple(b.floors), tuple(b.ceils), metrics.get("neutral_axis_refs")


def auto_crop_key(st: Settings) -> str:
    """NegPy's key for everything Auto crop's detection depends on (turn, flip, straightening...)."""
    from negpy.features.geometry.logic import autocrop_detection_key
    return autocrop_detection_key(to_config(st).geometry)


def find_auto_crop(path: Path, st: Settings, long_edge: int = 1600) -> tuple[tuple, str] | None:
    """Where Auto crop puts the frame on this scan: (rect in 0..1 of the turned, straightened
    picture, key of that framing), or None when it finds no border."""
    trial = st.copy()
    trial.crop, trial.auto_crop, trial.auto_rect, trial.auto_key = None, True, None, ""
    cfg = to_config(trial)
    ip = _processor()
    src, _, _ = ip._load_source_f32(str(path), cfg, fast_decode=True)
    ip.release_source_cache()
    step = max(1, int(np.ceil(max(src.shape[:2]) / long_edge)))
    src = np.ascontiguousarray(src[::step, ::step])
    _, metrics = ip.run_pipeline(src, cfg, source_hash(path) + "|autocrop", max(src.shape[:2]),
                                 prefer_gpu=False, readback_metrics=True)
    found = metrics.get("autocrop_resolved")
    if found is None or found.rect is None:
        return None
    return tuple(float(v) for v in found.rect), found.key


def uv_grid(path: Path, st: Settings, long_edge: int = 900) -> np.ndarray:
    """For every pixel of the picture as shown (rotated, straightened, cropped): the raw-scan
    position it comes from, as (u, v) in 0..1. Maps a click on the preview to a heal spot."""
    cfg = to_config(st, uncropped=False)
    ip = _processor()
    src, _, _ = ip._load_source_f32(str(path), cfg, fast_decode=True)
    ip.release_source_cache()
    step = max(1, int(np.ceil(max(src.shape[:2]) / long_edge)))
    src = np.ascontiguousarray(src[::step, ::step])
    _, metrics = ip.run_pipeline(src, cfg, source_hash(path) + "|uv", max(src.shape[:2]),
                                 prefer_gpu=False, readback_metrics=True, wants_uv_grid=True)
    return metrics["uv_grid"]


def raw_from_view(grid: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """A point on the picture as shown (0..1) -> the same point in the raw scan (0..1)."""
    from negpy.services.view.coordinate_mapping import CoordinateMapping
    return CoordinateMapping.map_click_to_raw(x, y, grid)


def view_from_raw(grid: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """The inverse of raw_from_view (to draw healed spots where they are)."""
    from negpy.services.view.coordinate_mapping import CoordinateMapping
    return CoordinateMapping.map_raw_to_viewport(x, y, grid)


def analyze_roll(frames: list[tuple[Path, Settings]]) -> tuple[tuple, tuple, list[bool], tuple | None]:
    """One colour baseline for a whole roll: measure every frame, then pool.

    Returns (floors, ceils, outliers, grey axis). Pooling (NegPy's pool_frame_bounds) takes the
    median over frames and marks the frames whose colour is far from the rest (a sunset, a red
    wall, a frame scanned differently) as outliers: they are left out of the baseline, and
    they keep their own measurement instead of the roll's.
    """
    from negpy.features.exposure.normalization import pool_neutral_axis
    from negpy.features.process.models import neutral_axis_tuple

    floors, ceils, axes = [], [], []
    for path, st in frames:
        f, c, a = frame_bounds(path, st)
        floors.append(f)
        ceils.append(c)
        axes.append(a if st.mode == "color" else None)
    pooled, outliers = pool_frame_bounds(np.array(floors), np.array(ceils))
    axis = pool_neutral_axis(axes, outliers)
    return (tuple(pooled.floors), tuple(pooled.ceils), [bool(o) for o in outliers],
            neutral_axis_tuple(axis) if axis is not None else None)


def film_stocks() -> list[dict]:
    """NegPy's film catalogue, negatives only (what SimpleNeg converts), sorted by name."""
    data = json.loads((_GEAR_DIR / "film_stocks.json").read_text())
    stocks = [d for d in data if d.get("colorType") in ("ColorNegative", "B&W Negative")]
    return sorted(stocks, key=lambda d: d["displayName"].casefold())


def stock_by_id(stock_id: str) -> dict | None:
    return next((d for d in film_stocks() if d["id"] == stock_id), None)


def _words(text: str) -> set[str]:
    import re
    return set(re.findall(r"[a-z0-9]+", text.lower().replace("(approx)", ""))) - {"kodak", "fujifilm", "fuji", "fujicolor"}


def crosstalk_for_stock(stock: dict) -> str:
    """The crosstalk profile NegPy ships for this stock, else Generic C41.

    Matched by name: every word of the profile (minus the maker) must appear in
    the stock's name, e.g. "Kodak Gold 200 (approx)" -> "Kodak Gold 200".
    """
    stock_words = _words(f"{stock.get('displayName', '')} {stock.get('stockName', '')}")
    best, best_len = CrosstalkProfiles.DEFAULT_NAME, 0
    for name in films():
        w = _words(name)
        if w and w <= stock_words and len(w) > best_len:
            best, best_len = name, len(w)
    return best


def settings_for_stock(st: Settings, stock: dict | None) -> None:
    """Adjustments that follow from the film stock, applied to a frame's settings.

    Only physical facts are applied: B&W film converts in B&W, and a colour film
    uses its own dye-unmix matrix when NegPy has one. Look choices stay yours.
    """
    if stock is None:                       # film not set (or cleared): the generic colour conversion
        st.grayscale, st.film = False, CrosstalkProfiles.DEFAULT_NAME
        return
    bw = stock.get("colorType") == "B&W Negative"
    st.grayscale = bw
    if not bw:
        st.film = crosstalk_for_stock(stock)




_MONTHS = {  # English and Italian month names (and their first three letters)
    **{m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                   "september", "october", "november", "december"], 1)},
    **{m: i for i, m in enumerate(["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
                                   "agosto", "settembre", "ottobre", "novembre", "dicembre"], 1)},
}
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())})


def normalize_date(text: str) -> str | None:
    """A date typed any common way -> "YYYY-MM-DD" (or "YYYY-MM", "YYYY", plus " HH:MM" if given).

    "" stays "" (no date); None means it was not understood. NegPy writes the result as
    EXIF DateTimeOriginal, the date Photos and Lightroom show as "taken". Numbers with the
    year last are read day first (21/06/2024), unless the first number cannot be a day's month
    partner (06/21/2024 is then read as June 21).
    """
    import re
    from datetime import date
    t = (text or "").strip().lower()
    if not t:
        return ""
    hm = ""
    m = re.search(r"(?:\s|t|,)\s*(\d{1,2})[:.h](\d{2})(?::\d{2})?\s*$", t)      # a time at the end
    if m and int(m.group(1)) < 24 and int(m.group(2)) < 60:
        hm, t = f" {int(m.group(1)):02d}:{m.group(2)}", t[:m.start()].strip(" ,")

    def year4(y: str) -> int:
        y = int(y)
        if y < 100:                                     # 24 -> 2024, 95 -> 1995
            y += 2000 if y <= date.today().year % 100 else 1900
        return y

    def build(y, mo=None, d=None):
        y = year4(y)
        try:
            date(y, mo or 1, d or 1)
        except ValueError:
            return None
        if not 1826 <= y <= date.today().year:
            return None
        out = f"{y:04d}" + (f"-{mo:02d}" if mo else "") + (f"-{d:02d}" if d and mo else "")
        return out + (hm if d and mo else "")

    words = re.findall(r"[a-zà-ù]+", t)
    month = next((_MONTHS[w] for w in words if w in _MONTHS), None)
    nums = re.findall(r"\d+", t)
    if month:                                           # 21 June 2024 / June 21, 2024 / giugno 2024
        year = next((n for n in nums if len(n) == 4), nums[-1] if len(nums) >= 1 and len(nums[-1]) == 2 and len(nums) > 1 else None)
        day = next((int(n) for n in nums if n != year and len(n) <= 2), None)
        return build(year, month, day) if year else None
    if re.fullmatch(r"\d{8}", t):                      # 20240621
        return build(t[:4], int(t[4:6]), int(t[6:]))
    if len(nums) == 1 and len(nums[0]) == 4:            # 2024
        return build(nums[0])
    if len(nums) == 2:                                  # 2024-06 or 06/2024
        a, b = nums
        return build(a, int(b)) if len(a) == 4 else build(b, int(a)) if len(b) == 4 else None
    if len(nums) == 3:
        a, b, c = nums
        if len(a) == 4:                                 # 2024-06-21
            return build(a, int(b), int(c))
        day, mo = int(a), int(b)                        # 21/06/2024 (day first)
        if mo > 12 and day <= 12:                       # 06/21/2024 can only be month first
            day, mo = mo, day
        return build(c, mo, day)
    return None


def parse_gps(text: str) -> tuple[float, float] | None:
    """Coordinates as copied from Apple Maps or Google Maps -> (latitude, longitude).

    Takes "35.0116, 135.7681", "35.0116 135.7681", degree signs, N/S/E/W letters, and
    Google Maps links ("…/@35.0116,135.7681,15z").
    """
    import re
    t = (text or "").strip()
    if not t:
        return None
    m = re.search(r"@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)", t)      # a Google Maps link
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
    else:
        parts = re.findall(r"(-?\d+(?:\.\d+)?)\s*°?\s*([NSEWnsew]?)", t)
        if len(parts) < 2:
            return None
        (a, ha), (b, hb) = parts[0], parts[1]
        lat = -abs(float(a)) if ha.upper() == "S" else float(a)
        lon = -abs(float(b)) if hb.upper() == "W" else float(b)
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def metadata_from_exif(f: dict, roll_name: str = "", frame: int | None = None):
    """NegPy MetadataConfig from a frame's EXIF fields (the keys of the Exif tab, plus the
    roll's push/pull). Empty fields are left out."""
    from negpy.features.metadata.models import MetadataConfig

    text = lambda k: str(f.get(k) or "").strip()
    kw = dict(camera_make=text("camera_make"), camera_model=text("camera_model"),
              lens_make=text("lens_make"), lens_model=text("lens_model"),
              capture_date=text("capture_date"), location_city=text("location_city"),
              gps_latitude=(parse_gps(text("gps")) or (None, None))[0],
              gps_longitude=(parse_gps(text("gps")) or (None, None))[1],
              push_pull=int(f.get("push_pull") or 0), capture_roll=roll_name, capture_frame=frame)
    stock = stock_by_id(text("film_stock_id")) if text("film_stock_id") else None
    if stock:
        kw.update(film=stock["displayName"], film_stock_id=stock["id"],
                  film_manufacturer=stock.get("manufacturer", ""), film_iso=stock.get("iso"),
                  film_color_type=stock.get("colorType", ""), format=stock.get("format", ""))
    return MetadataConfig(**kw)


# ---------------------------------------------------------------------------
# 6. Auto straighten
# ---------------------------------------------------------------------------
def auto_straighten(path: Path, st: Settings) -> float | None:
    """Angle (degrees) that squares the film frame's edges, or None if no edge is found.

    Uses NegPy's frame-edge detector on the uncropped picture; it needs some film
    border or frame edge in the scan to measure against.
    """
    from negpy.features.geometry.skew import measure_frame_skew

    trial = st.copy()
    trial.fine_rotation = 0.0
    img = preview(path, trial, 1200, uncropped=True).astype(np.float32) / 255.0
    skew = measure_frame_skew(img, fit_keystone=False)
    if skew is None or skew.confidence < 0.2:
        return None
    return float(np.clip(skew.fine_rotation, -45, 45))


def auto_level(path: Path, st: Settings) -> float | None:
    """The angle Auto crop levels a tilted scan by, so no light around the film is left in the
    corners; None when the frame is straight already or its edges cannot be trusted.

    Stricter than auto_straighten: it is applied without anyone watching, so NegPy must have
    found the film's own frame, not a line inside the picture.
    """
    from negpy.features.geometry.skew import trusted_frame_skew

    trial = st.copy()
    trial.fine_rotation = 0.0
    img = preview(path, trial, 1200, uncropped=True).astype(np.float32) / 255.0
    skew = trusted_frame_skew(img, fit_keystone=False)
    if skew is None or skew.confidence < 0.3:
        return None
    angle = round(float(skew.fine_rotation), 1)
    return angle if 0.2 <= abs(angle) <= 10.0 else None


# ---------------------------------------------------------------------------
# 7. Contact sheet
# ---------------------------------------------------------------------------
# A whole roll on one sheet of paper, as a darkroom contact print: the strips of film lie on
# the paper at their real size. NegPy lays out and draws the sheet (services/export/contact_sheet*);
# here the frames are rendered small, with their adjustments, and handed over.

# key -> (menu label, width in mm, height in mm). The sheet is turned the way that needs fewer pages.
SHEET_PAPERS = {
    "24x30": ("24 × 30.5 cm  (a whole 36-exposure roll)", 241.3, 304.8),
    "a4": ("A4  (21 × 29.7 cm)", 210.0, 297.0),
    "letter": ("US Letter  (8.5 × 11 in)", 215.9, 279.4),
    "20x25": ("20 × 25 cm  (8 × 10 in)", 203.2, 254.0),
    "a3": ("A3  (29.7 × 42 cm)", 297.0, 420.0),
    "30x40": ("30 × 40 cm  (12 × 16 in)", 304.8, 406.4),
}
SHEET_FILMS = {"auto": "Detect from the frames", "35mm": "35mm", "half": "35mm half frame",
               "6×4.5": "120  ·  6 × 4.5", "6×6": "120  ·  6 × 6", "6×7": "120  ·  6 × 7", "6×9": "120  ·  6 × 9"}
SHEET_DPI = {150: "150 dpi  (screen, sharing)", 300: "300 dpi  (printing)"}


@dataclass
class SheetOptions:
    paper: str = "24x30"             # key of SHEET_PAPERS
    film: str = "auto"               # key of SHEET_FILMS
    film_edges: bool = True          # strips of film with perforations and edge print; False = upright frames in rows
    white_paper: bool = False        # white around the film (no ink on a home printer) instead of black
    label: bool = True               # the roll's name, film, camera and date above the frames
    dpi: int = 300


def sheet_label(roll_name: str, stock: dict | None = None, push_pull: int = 0, camera: str = "", date: str = "") -> str:
    """The line printed above the frames: roll · film (and what it was pushed to) · camera · date."""
    film = (stock or {}).get("displayName", "")
    if film and push_pull and (stock or {}).get("iso"):
        film += f" @ EI {int(stock['iso'] * 2 ** push_pull)}"
    return " · ".join(part.strip() for part in (roll_name, film, camera, date) if part and part.strip())


class _StockLibrary:
    """NegPy's bundled film stocks, as its edge-print code looks them up (a respooled film prints its source's edge)."""

    def __init__(self):
        from negpy.features.metadata.gear_models import FilmStock

        self.film_stocks = [FilmStock.from_dict(d) for d in json.loads((_GEAR_DIR / "film_stocks.json").read_text())]

    def get_film_stock(self, stock_id: str):
        return next((s for s in self.film_stocks if s.id == stock_id), None)


def _sheet_tile(path: Path, st: Settings, long_edge: int):
    """(the frame as it would export, small; the settings NegPy rendered it with)"""
    from negpy.services.export.contact_sheet_roll import tile_params

    cfg = to_config(st, path=path)
    tile = _processor().render_display_array(str(path), tile_params(cfg), source_hash(path), long_edge,
                                             prefer_gpu=False, fast_decode=True)
    return tile, cfg


def contact_sheet(frames: list[tuple[Path, Settings]], out_dir: Path, name: str, opts: SheetOptions | None = None,
                  label: str = "", stock: dict | None = None, progress=None) -> list[Path]:
    """Write the roll's contact sheet as JPEG (sRGB), one file per page; returns the files.

    `frames` are in roll order, with the settings each one is shown with. `label` is printed above
    the frames (see sheet_label); `stock` (a film_stocks() entry) decides the edge print and whether the
    film base prints warm (colour) or neutral (black and white). `progress(done, total)` is called per frame.
    A frame that cannot be rendered prints as an unexposed one, so the numbering stays right.
    """
    import statistics

    from negpy.features.metadata.gear_models import FilmStock
    from negpy.services.export.contact_sheet import ContactSheetService
    from negpy.services.export.contact_sheet_layout import MM_PER_INCH, better_orientation, plan_rows, plan_sheets
    from negpy.services.export.contact_sheet_roll import (FrameFacts, SheetFrame, SheetLook, _nearest_frame_120,
                                                          read_scan_size, turns_for)
    from negpy.services.export.encoders import Resolution, encode_jpeg
    from negpy.services.film_edge.edge import edge_style
    from negpy.services.film_edge.geometry import SheetFormat, film_geometry
    from negpy.services.film_edge.layouts import LAYOUTS

    opts = opts or SheetOptions()
    if not frames:
        raise RuntimeError("There are no frames to print.")
    dpi = opts.dpi if opts.dpi in SHEET_DPI else 300
    px_per_mm = dpi / MM_PER_INCH

    # 1. every frame, small. Big enough for the largest window a frame can lie in (6×9: 84 mm).
    known = {"35mm": 36.0, "half": 24.0, "6×4.5": 56.0, "6×6": 56.0, "6×7": 70.0, "6×9": 84.0}
    long_edge = int(known.get(opts.film, 84.0) * px_per_mm * 1.5) + 1
    tiles, sheet_frames = [], []
    for i, (path, st) in enumerate(frames):
        try:
            tile, cfg = _sheet_tile(path, st, long_edge)
        except Exception:
            tile, cfg = None, to_config(st)
        finally:
            _processor().release_source_cache()
        tiles.append(tile)
        asset = {"path": str(path), "name": path.name}
        sheet_frames.append(SheetFrame(asset, cfg, FrameFacts(scan_size=read_scan_size(asset))))
        if progress:
            progress(i + 1, len(frames))
    if all(t is None for t in tiles):
        raise RuntimeError("None of the frames could be rendered.")
    aspects = [(t.shape[1] / t.shape[0]) if t is not None else 1.5 for t in tiles]     # width / height, upright

    # 2. the film: chosen, or told from the frames' shape (a 6×9 roll reads as 35mm: both are 3:2)
    if opts.film == "auto":
        median = statistics.median(max(a, 1 / a) for a, t in zip(aspects, tiles) if t is not None)
        fmt, size = (SheetFormat.MEDIUM, _nearest_frame_120(median)) if median <= 1.3 else (SheetFormat.FULL_FRAME, "")
    elif opts.film == "35mm":
        fmt, size = SheetFormat.FULL_FRAME, ""
    elif opts.film == "half":
        fmt, size = SheetFormat.HALF_FRAME, ""
    else:
        fmt, size = SheetFormat.MEDIUM, opts.film
    geometry = film_geometry(fmt, size) if size else film_geometry(fmt)

    # 3. the look: film base colour and edge print
    color = (stock.get("colorType") == "ColorNegative") if stock else frames[0][1].mode == "color"
    palette = "color" if color else "bw"
    library = _StockLibrary()
    edge = edge_style(library.get_film_stock(stock["id"]) or FilmStock.from_dict(stock), library) if stock else edge_style(None)
    edge = replace(edge, dx=fmt != SheetFormat.MEDIUM and palette in LAYOUTS[edge.layout].dx)
    look = SheetLook(palette, 0, edge, label if opts.label else "", opts.white_paper)

    # 4. the layout on the paper
    _, w, h = SHEET_PAPERS.get(opts.paper, SHEET_PAPERS["24x30"])
    if opts.film_edges:
        w, h = better_orientation(w, h, geometry, len(frames), opts.label)
        plan = plan_sheets(w, h, geometry, len(frames), opts.label)
    else:
        w, h = better_orientation(w, h, geometry, len(frames), opts.label, aspects)
        plan = plan_rows(w, h, geometry, aspects, opts.label)
    if not plan.pages:
        raise RuntimeError(plan.reason or "The frames do not fit on this paper.")
    turns = [turns_for(f, geometry, t.shape[:2]) if t is not None else 0 for f, t in zip(sheet_frames, tiles)]

    # 5. draw and save. Written under a temporary name first, so a failure never leaves half a set.
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = file_name("{name} contact sheet", {"name": name}) if name.strip() else "contact sheet"
    n = len(plan.pages)
    icc = _srgb_icc()
    outs, parts = [], []
    try:
        for page in range(n):
            sheet = ContactSheetService.render_sheet(plan, page, tiles, turns, px_per_mm, look)
            out = unique_path(out_dir / (f"{stem}.jpg" if n == 1 else f"{stem} {page + 1} of {n}.jpg"), False)
            part = out.with_name(out.name + ".part")
            parts.append(part)
            part.write_bytes(encode_jpeg(sheet, icc=icc, resolution=Resolution.from_dpi(dpi), quality=95))
            outs.append(out)
        for part, out in zip(parts, outs):
            part.replace(out)
    finally:
        for part in parts:
            part.unlink(missing_ok=True)
    return outs
