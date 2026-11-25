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
from negpy.features.process.models import scan_setup_values
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


# Lab print: the colour a C-41 negative takes when it is printed on photo paper, which a straight
# scan conversion leaves out. Measured with a spectral simulation from the manufacturers' data
# sheets (spektrafilm's digitised Kodak / Fujifilm curves, CC BY-SA 4.0, github.com/andreavolpato/
# spektrafilm): a ColorChecker exposed on each film, scanned under a high-CRI white LED by real
# camera sensors and converted here, against the same negative printed on the paper its maker
# pairs it with. Per colour-mixer band (red .. magenta): the hue turn in degrees and the chroma
# factor taking this conversion to the print. Stocks without data get the Kodak average (the
# stocks most people shoot). tools/print_character/README.md tells how the numbers were made.
PRINT_CHARACTER = {
    "Kodak Gold 200 (approx)": ((-6.3, -8.3, -12.0, -5.5, -2.9, -8.8, -3.0, -8.7),
                                (1.01, 0.95, 0.82, 0.68, 1.06, 1.01, 0.85, 1.05)),
    "Kodak Ultra Max 400 (approx)": ((-5.2, -9.0, -11.5, -4.8, -3.2, -10.1, -0.6, -6.5),
                                     (1.05, 0.96, 0.83, 0.70, 1.06, 0.97, 0.83, 1.07)),
    "Kodak Ektacolor Pro 160 / Portra 160 (approx)": ((-3.4, -9.3, -14.8, -8.9, -7.7, -16.8, -11.7, -6.9),
                                                      (1.08, 0.99, 0.87, 0.71, 0.94, 0.87, 0.85, 1.00)),
    "Kodak Ektacolor Pro 400 / Portra 400 (approx)": ((-4.1, -11.4, -16.7, -8.7, -6.8, -15.0, -7.4, -6.0),
                                                      (0.97, 0.93, 0.81, 0.66, 0.99, 0.83, 0.74, 0.92)),
    "Kodak Ektacolor Pro 800 / Portra 800 (approx)": ((-4.9, -10.0, -15.0, -7.2, -6.8, -13.0, -3.2, -5.9),
                                                      (0.97, 0.91, 0.80, 0.64, 0.95, 0.84, 0.69, 0.98)),
    "Kodak Ektar 100 (approx)": ((-3.6, -7.7, -15.7, -7.9, -7.6, -14.4, -2.0, -11.9),
                                 (1.12, 1.03, 0.89, 0.70, 1.04, 1.08, 0.78, 1.13)),
    "Fujicolor C200 (approx)": ((0.6, -7.7, 2.4, 5.8, -7.2, -13.2, 7.9, 18.7),
                                (1.21, 0.96, 0.90, 0.84, 1.09, 0.79, 0.75, 1.08)),
    "Fujicolor Superia X-TRA 400 (approx)": ((0.4, -7.2, 2.4, 5.7, -8.1, -8.5, 9.6, 19.4),
                                             (1.31, 1.00, 0.97, 0.92, 1.15, 0.81, 0.80, 1.13)),
    "Fujicolor Pro 400H (approx)": ((-1.3, -11.6, -5.9, -0.1, -8.9, -8.6, 5.1, 9.5),
                                    (1.16, 0.93, 0.85, 0.74, 1.13, 0.83, 0.78, 1.07)),
    # Cinema negative (and its respools: CineStill, Flic Film, Silbersalz35) through a C-41 lab onto
    # RA-4 paper (Supra Endura): hues stay near neutral, colour is calmer than a still film's
    "Kodak VISION3 50D 5203/7203 (approx)": ((-2.2, -2.3, -4.9, 1.7, -4.1, -1.6, 2.3, -0.5),
                                             (0.87, 0.85, 0.83, 0.76, 0.92, 0.83, 0.76, 0.90)),
    "Kodak VISION3 200T 5213/7213 (approx)": ((-3.3, -4.0, -4.4, 3.2, -5.1, -2.7, 4.1, 2.9),
                                              (0.83, 0.81, 0.77, 0.73, 0.87, 0.76, 0.73, 0.84)),
    "Kodak VISION3 250D 5207/7207 (approx)": ((-4.2, -5.1, -7.5, -0.3, -5.9, -2.8, 3.0, -1.1),
                                              (0.89, 0.85, 0.79, 0.71, 0.89, 0.83, 0.69, 0.92)),
    "Kodak VISION3 500T 5219/7219 (approx)": ((-3.1, -9.0, -12.4, -2.6, -6.0, -6.9, -0.4, 1.4),
                                              (0.84, 0.84, 0.74, 0.63, 0.84, 0.77, 0.61, 0.80)),
}
PRINT_CHARACTER_DEFAULT = ((-4.6, -9.3, -14.3, -7.2, -5.8, -13.0, -4.7, -7.7),
                           (1.03, 0.96, 0.84, 0.68, 1.01, 0.93, 0.79, 1.03))
# The print is ~9 % less colourful than this conversion overall (Kodak average): that part is left
# out, so the look turns hues rather than dulling; a stock keeps its own difference from the average.
PRINT_CHROMA_REF = 0.91
PRINT_STRENGTH = 0.7    # of the measured difference: the look, kept subtle (sliders adjust on top)
LAB_PRINT = "lab"


def print_character(film: str) -> tuple[tuple, tuple]:
    """The Lab print look of a film as colour-mixer values (hue, saturation per band)."""
    hue, chroma = PRINT_CHARACTER.get(film, PRINT_CHARACTER_DEFAULT)
    from negpy.features.lab.logic import HSL_HUE_DEGREES
    return (tuple(PRINT_STRENGTH * h / HSL_HUE_DEGREES for h in hue),
            tuple(PRINT_STRENGTH * (c / PRINT_CHROMA_REF - 1.0) for c in chroma))


def _mixer(st: "Settings") -> tuple[tuple, tuple]:
    """The colour mixer NegPy gets: the photo's own, on top of the Lab print look (colour film only)."""
    hue, sat = tuple(st.hsl_hue), tuple(st.hsl_saturation)
    if st.paper != LAB_PRINT or st.mode != "color" or st.grayscale:
        return hue, sat
    ph, ps = print_character(st.film)
    return (tuple(float(np.clip(u + p, -1.0, 1.0)) for u, p in zip(hue, ph)),     # turns add up
            tuple(float(np.clip((1 + u) * (1 + p) - 1, -1.0, 1.0)) for u, p in zip(sat, ps)))   # chroma scales multiply


def print_looks(grayscale: bool) -> list[tuple[str, str]]:
    """(key, name) of the print looks for colour or B&W film, "None" first: NegPy's paper profiles,
    and for colour film the Lab print look right after None."""
    from negpy.features.exposure.papers import profiles_for_mode
    out = []
    for key, p in profiles_for_mode(MODES["bw" if grayscale else "color"]):
        name = "None" if key == "neutral" else p.label.replace("Digital (", "").rstrip(")") if p.label.startswith("Digital (") else p.label
        out.append((key, name))
    if not grayscale:
        out.insert(1, (LAB_PRINT, "Lab print"))
    return out


def film_matrix(dyes: tuple) -> tuple:
    """The film's dye-unmix matrix alone (rows sum to 1): what an RGB light scan needs."""
    d = np.array(dyes, dtype=np.float64).reshape(3, 3)
    return tuple(float(v) for v in (d / d.sum(axis=1, keepdims=True)).ravel())


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
    # Highlights and Shadows, -1 .. 1, like Lightroom: darker ← → brighter in that part of the tones
    highlight_tone: float = 0.0
    shadow_tone: float = 0.0
    # White balance on the positive, like a raw converter's: -1 .. 1 (NegPy lab wb_temp / wb_tint)
    temp: float = 0.0              # cooler (blue) ← → warmer (amber)
    tint: float = 0.0              # greener ← → more magenta
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
    exposure: float = 0.0          # -1 .. 1: the positive's exposure, ±EXPOSURE_STOPS at the ends (not D)
    white_point: float = 0.0       # -1 .. 1: Whites, the light end, dimmer ← → brighter
    black_point: float = 0.0       # -1 .. 1: Blacks, the dark end, deeper ← → lifted
    # Colour mixer, per band (red, orange, yellow, green, aqua, blue, purple, magenta), each -1 .. 1
    hsl_hue: tuple = (0.0,) * 8
    hsl_saturation: tuple = (0.0,) * 8
    hsl_luminance: tuple = (0.0,) * 8
    # Auto crop, once found: the rectangle (0..1, like `crop`) and NegPy's key for the framing it was
    # found on. Kept so every render (draft, full, export) uses the same crop instead of looking again.
    auto_rect: tuple | None = None
    auto_key: str = ""
    # Set automatically, not in the panel
    film: str = "Generic C41"    # crosstalk profile (dye unmix matrix); set from the roll's film stock
    paper: str = "lab"           # print look: "lab" = Lab print (PRINT_CHARACTER), "neutral" = none, else a NegPy paper
    # The roll's one white balance (R, G, B multipliers) its camera RAWs decode on: a camera in auto
    # white balance balances each frame differently, and the same film base would read differently
    raw_wb: tuple | None = None
    light: str = "white"         # the scanning light, from Settings: "white" (broadband panel) or "rgb" (narrowband LEDs)
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
    roll_slope: tuple | None = None  # the roll's dye contrast per channel (NegPy pool_dye_slopes; None = all alike)

    def copy(self) -> "Settings":
        return Settings(**asdict(self))

    @property
    def mode(self) -> str:
        return "bw" if self.grayscale else "color"


# ---------------------------------------------------------------------------
# 2. Settings -> NegPy config
# ---------------------------------------------------------------------------
EXPOSURE_STOPS = 2.0  # Exposure at the end of its slider, in stops (one slider step = 0.1 stop)


def to_config(st: Settings, uncropped: bool = False, path: Path | None = None,
              long_edge: int | None = None) -> WorkspaceConfig:
    """`uncropped=True` renders the whole frame (used while editing the crop).
    `path` is the scan being rendered and `long_edge` its render size (None: the whole decode):
    Contrast measures the scan as that render will (see _with_contrast)."""
    flat = {
        "process_mode": MODES[st.mode],
        # The print look: "neutral" is NegPy's "Digital", an ideal print with no paper dyes. A look of
        # the other kind of film (colour paper on B&W) falls back to it inside NegPy.
        "paper_profile": "neutral" if st.paper == LAB_PRINT else st.paper,
        # Print Density (exposure time under the enlarger). Auto stays on.
        "density": st.density,
        # ISO-R grade: lower number = more contrast, 115 is normal.
        # With a scan to measure, Contrast is applied afterwards on the frame's own automatic contrast.
        "grade": 115.0 if path is not None else float(np.clip(115.0 - 70.0 * (st.contrast - 1.0), 50.0, 180.0)),
        "saturation": st.saturation,
        "cast_removal_strength": st.cast_removal,
        "skin_protection": st.skin_protection,
        # White light mixes the dye layers; extra separation gives back what it loses. RGB light does not.
        "dye_separation": st.dye_separation if st.light != "rgb" else 1.0,
        # A camera under RGB light: no camera white balance in the decode (NegPy's scanning-setup rule)
        "linear_raw": scan_setup_values("camera", "narrowband" if st.light == "rgb" else "white")[0],
        "wb_cyan": CMY_SCALE * CMY_BALANCE[0] * (st.cyan - 1.0),
        "wb_magenta": CMY_SCALE * CMY_BALANCE[1] * (st.magenta - 1.0),
        "wb_yellow": CMY_SCALE * CMY_BALANCE[2] * (st.yellow - 1.0),
        # Split grade, ISO-R points: negative = harder, like the main grade
        # Zone filtration (1.0 = 20 CC). Warmth = yellow plus a little red, tint = magenta
        "wb_temp": st.temp,
        "wb_tint": st.tint,
        "glow_amount": st.glow,
        "sharpen": st.sharpen,
        "chroma_denoise": 3.0 * st.color_noise,
        # Light: Highlights, Shadows, Whites, Blacks, Lightroom-like on the finished positive (lab.apply_tone)
        "tone_highlights": st.highlight_tone,      # Lightroom-like, on the finished positive
        "tone_shadows": st.shadow_tone,
        "tone_whites": st.white_point,
        "tone_blacks": st.black_point,
        "exposure_ev": EXPOSURE_STOPS * st.exposure,
        "hsl_hue": _mixer(st)[0],
        "hsl_saturation": _mixer(st)[1],
        "hsl_luminance": tuple(st.hsl_luminance),
        # A spot is a search area: NegPy repairs only what stands out from the film around it.
        # An area is filled with the same repair, band by band (_inside_strokes).
        "manual_heal_strokes": [([[x, y]], float(size), 0.0, 0.0) for x, y, size in st.heal_spots]
        + (_inside_strokes(_as_tuples(st.heal_areas)) if st.heal_areas else []),
        "analysis_buffer": st.border,
        "analysis_rect": st.analysis_rect,
        # The film is read once, on the scan as loaded: crop, turn and straighten only frame it.
        # Re-measure Inside Crop (analysis_rect) is the one way to meter the framed picture.
        "meter_as_loaded": True,
        # The black and white points skip the most extreme 0.05 % of the picture (on top of NegPy's
        # own 0.01 %): a few dust specks then cannot set them, so healing them leaves the rest alone
        "luma_range_clip": 0.05,
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
        # The film's own dye matrix under either light; white light differs only in the extra
        # dye separation above and the camera white balance in the decode.
        flat["crosstalk_matrix"] = film_matrix(tuple(matrix))
        flat["crosstalk_profile"] = st.film
        flat["crosstalk_process"] = "Color Negative"
    if st.raw_wb:
        flat["decode_wb"] = tuple(st.raw_wb)
    if st.roll_floors and st.roll_ceils:
        # Colour from the whole roll (film base + cast are the same on every frame),
        # brightness range per frame, like a lab scanner reading the whole order first.
        flat["locked_floors"] = tuple(st.roll_floors)
        flat["locked_ceils"] = tuple(st.roll_ceils)
        flat["locked_slope"] = st.roll_slope     # how the roll's colour follows each frame's exposure
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
        elif st.auto_crop and st.auto_rect:
            # Auto crop is found once and then kept like a crop of your own: turning and mirroring
            # move it with the picture, straightening leaves it; only Refresh looks again
            flat["crop_rect"] = tuple(st.auto_rect)
        elif st.auto_crop:
            flat["crop_from_auto"] = True        # until the search (find_auto_crop) answers
    cfg = WorkspaceConfig.from_flat_dict(flat)
    if path is not None and abs(st.contrast - 1.0) > 1e-6:
        cfg = _with_contrast(cfg, path, st, uncropped, long_edge)
    return cfg

# ---------------------------------------------------------------------------
# Cleaning areas
# ---------------------------------------------------------------------------
# An area to clean is filled with heal bands, row by row, the same repair a heal spot does: it
# fixes only what stands out from the film around it, so clean film inside the shape stays as it is.
HEAL_AREA_BAND = 24.0     # band diameter, px at NegPy's HEAL_SIZE_REF (1600 px long edge)


def _as_tuples(v):
    return tuple(_as_tuples(x) for x in v) if isinstance(v, (list, tuple)) else v


@functools.lru_cache(maxsize=64)
def _inside_strokes(areas: tuple) -> list:
    """Heal bands (NegPy manual_heal_strokes) filling the inside of each shape (scan 0..1)."""
    from negpy.features.retouch.models import HEAL_SIZE_REF
    reach = HEAL_AREA_BAND / HEAL_SIZE_REF        # a band's diameter, as a share of the long edge
    step = reach * 0.6                            # rows overlap on any frame shape
    strokes = []
    for poly in areas:
        ys = [q[1] for q in poly]
        y = min(ys) + step / 2
        while y < max(ys):
            xs = []
            for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
                if (y0 <= y) != (y1 <= y):
                    xs.append(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
            xs.sort()
            for a, b in zip(xs[0::2], xs[1::2]):
                a, b = a + reach / 2, b - reach / 2    # the round ends stay inside the shape
                if b <= a:
                    a = b = (a + b) / 2
                strokes.append(([[round(a, 5), round(y, 5)], [round(b, 5), round(y, 5)]], HEAL_AREA_BAND, 0.0, 0.0))
            y += step
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


def _widen_grade_ladder():
    """The slope is printed as an ISO R grade on the frame's own density range, and NegPy clamps
    grades to R50..R180. On a thin or flat negative the automatic slope needs a grade outside that
    ladder, so the clamp gave a different curve and every fader step the same one. The ladder is
    widened past anything slope_min..slope_max can need; at the default grade nothing changes."""
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS

    EXPOSURE_CONSTANTS["iso_r_min"] = min(EXPOSURE_CONSTANTS["iso_r_min"], 5.0)
    EXPOSURE_CONSTANTS["iso_r_max"] = max(EXPOSURE_CONSTANTS["iso_r_max"], 600.0)


_widen_grade_ladder()
# Settings that change the picture's look but not what the film measures (the meter ignores them)
_LOOK_FIELDS = ("contrast", "density", "saturation", "cyan", "magenta", "yellow", "highlight_tone", "shadow_tone",
                "temp", "tint", "glow", "sharpen", "paper",
                "color_noise", "heal_spots", "heal_areas", "white_point", "black_point", "exposure",
                "hsl_hue", "hsl_saturation", "hsl_luminance")
_meters: dict[tuple, dict] = {}
_meters_lock = threading.Lock()


def _meter(path: Path, st: Settings, uncropped: bool = False, long_edge: int | None = None) -> dict:
    """What NegPy's light meter reads on this frame, rendered at `long_edge` (cached: it only
    changes with the framing, the film, the roll and the render size, not with the look).
    The reading depends on the size and on how the scan was shrunk, so it is taken on the
    same downsample the render uses; a reading from another size prints another curve."""
    neutral, defaults = st.copy(), Settings()
    for f in _LOOK_FIELDS:
        setattr(neutral, f, getattr(defaults, f))
    key = (source_hash(path), flat_profile_id(neutral), repr(neutral), uncropped, long_edge)
    with _meters_lock:
        if key in _meters:
            return _meters[key]
    from negpy.services.rendering.image_processor import _downsample_to_long_edge

    cfg = to_config(neutral, uncropped)
    ip = _processor()
    # A sized render decodes fast (half-size raw); a full-size one (zoom, export) decodes fully
    src, _, _ = ip._load_source_f32(str(path), cfg, fast_decode=long_edge is not None)
    small = np.ascontiguousarray(_downsample_to_long_edge(src, long_edge or 0))
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


def _with_contrast(cfg: WorkspaceConfig, path: Path, st: Settings, uncropped: bool = False,
                   long_edge: int | None = None) -> WorkspaceConfig:
    from dataclasses import replace

    from negpy.features.exposure import auto_sliders
    from negpy.features.exposure.logic import slope_to_grade
    from negpy.features.exposure.models import EXPOSURE_CONSTANTS
    from negpy.features.exposure.papers import slope_gain

    m = _meter(path, st, uncropped, long_edge)
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
# else runs (before film-base detection too, so the corners cannot fool it).
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


def _srgb_oklab_ab(rgb) -> tuple[float, float]:
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    lms = np.cbrt(np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                            [0.0883024619, 0.2817188376, 0.6299787005]]) @ lin)
    return (1.9779984951 * lms[0] - 2.4285922050 * lms[1] + 0.4505937099 * lms[2],
            0.0259040371 * lms[0] + 0.7827717662 * lms[1] - 0.8086757660 * lms[2])


def mixer_band(rgb) -> int | None:
    """The colour mixer band (0 red .. 7 magenta) a colour picked on the picture (sRGB 0..255) belongs
    to: the one with the most weight at its hue, as NegPy's mixer weighs it. None for a grey."""
    from negpy.features.lab.logic import _band_weights
    a, b = _srgb_oklab_ab(rgb)
    if np.hypot(a, b) < 0.02:
        return None
    k0, k1, s = _band_weights(np.array([np.degrees(np.arctan2(b, a))]))
    return int(k1[0] if s[0] > 0.5 else k0[0])


def neutral_temp_tint(rgb, temp: float, tint: float) -> tuple[float, float]:
    """Temp and Tint that make a spot of the picture (sRGB 0..255, shown with `temp` and `tint`) grey,
    like a raw converter's white-balance dropper. Clicking again on the same spot refines it."""
    from negpy.features.lab.logic import WB_TEMP_STOPS, WB_TINT_STOPS
    c = np.clip(np.asarray(rgb, dtype=np.float64) / 255.0, 1 / 255, 1.0)
    r, g, b = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    temp += np.log2(b / r) / (2 * WB_TEMP_STOPS)
    tint -= np.log2(np.sqrt(r * b) / g) / WB_TINT_STOPS
    return float(np.clip(round(temp, 2), -1, 1)), float(np.clip(round(tint, 2), -1, 1))


def source_hash(path: Path) -> str:
    """NegPy caches the decoded scan under this key; changes when the file does."""
    s = path.stat()
    return f"{path}:{s.st_mtime_ns}:{s.st_size}"


# The viewer (preview and zoom) renders on the graphics card when there is one: the same conversion
# as the CPU, several times faster on large renders. Thumbnails, measuring and export stay on the
# CPU, the reference. A card that fails once is not asked again this session.
USE_GPU = True
_gpu_failed = False
_gpu_processors: dict[int, ImageProcessor] = {}


def _gpu_processor() -> ImageProcessor | None:
    """This thread's GPU-backed processor, or None when the card is off, missing or failed."""
    global _gpu_failed
    if not USE_GPU or _gpu_failed:
        return None
    tid = threading.get_ident()
    with _processors_lock:
        ip = _gpu_processors.get(tid)
        if ip is None:
            try:
                ip = ImageProcessor(use_gpu=True)
            except Exception:
                ip = None
            if ip is None or ip.engine_gpu is None:
                _gpu_failed = True
                return None
            _gpu_processors[tid] = ip
    return ip


def gpu_active() -> bool:
    return USE_GPU and not _gpu_failed


def _display_render(path: Path, cfg: WorkspaceConfig, long_edge: int, fast_decode: bool, keep_source: bool,
                    keep_small: bool = False) -> np.ndarray:
    """A viewer render: on the card when it works, else (and from then on) on the CPU."""
    global _gpu_failed
    ip = _gpu_processor()
    if ip is not None:
        try:
            arr = ip.render_display_array(str(path), cfg, source_hash(path), long_edge, prefer_gpu=True,
                                          fast_decode=fast_decode, keep_source=keep_source, keep_small=keep_small)
        except Exception:
            arr = None
        if arr is not None:
            return arr
        _gpu_failed = True                       # the card could not do it: the CPU from now on
        print("DumbConvert: the graphics card failed a render; using the CPU from now on", file=sys.stderr)
    arr = _processor().render_display_array(str(path), cfg, source_hash(path), long_edge, prefer_gpu=False,
                                            fast_decode=fast_decode, keep_source=keep_source, keep_small=keep_small)
    if arr is None:
        raise RuntimeError(f"NegPy could not render {path.name}")
    return arr


def prefetch_preview(path: Path, st: Settings, long_edge: int):
    """Get a frame ready for the viewer: its scan decoded and shrunk to the viewer's size in NegPy's
    shared cache (about 20 MB at 1600 px), so switching to it renders at once."""
    _processor().prefetch_display_source(str(path), to_config(st, False), long_edge, fast_decode=True)


def preview(path: Path, st: Settings, long_edge: int, uncropped: bool = False,
            keep_source: bool = False, gpu: bool = False) -> np.ndarray:
    """Render `path` to an sRGB uint8 array, long edge about `long_edge` px.

    `keep_source=True` keeps the decoded scan in this thread's memory, so the next render
    of the same file skips decoding (the live preview); others release it. `gpu`: the viewer's
    render, on the graphics card when there is one (see USE_GPU).
    """
    cfg = to_config(st, uncropped, path, long_edge)
    if gpu:
        # the viewer: its shrunk decode is kept (shared with prefetch_preview), the full one is not needed
        return _display_render(path, cfg, long_edge, True, False, keep_small=True)
    arr = _processor().render_display_array(
        str(path), cfg, source_hash(path), long_edge,
        prefer_gpu=False, fast_decode=True, keep_source=keep_source,
    )
    if arr is None:
        raise RuntimeError(f"NegPy could not render {path.name}")
    return arr



def original_preview(path: Path, st: Settings, long_edge: int) -> np.ndarray:
    """The scan as it came from the camera, before any conversion: the negative itself, turned and
    flipped like the frame (no crop, no flat field), its light panel brought up to near white."""
    from negpy.kernel.image.logic import apply_geometry_orientation
    from negpy.services.rendering.image_processor import _downsample_to_long_edge

    ip = _processor()
    src, _, _ = ip._load_source_f32(str(path), to_config(Settings()), fast_decode=True, unmix=False)
    img = _downsample_to_long_edge(np.ascontiguousarray(src[..., :3], dtype=np.float32), long_edge)
    img = apply_geometry_orientation(img, st.rotation % 4, st.mirror, False)
    white = float(np.percentile(img, 99.5)) or 1.0
    shown = np.clip(img / white, 0.0, 1.0) ** (1 / 2.2)          # display encoding of linear light
    return np.ascontiguousarray((shown * 255 + 0.5).astype(np.uint8))


def thumbnail(path: Path, st: Settings, long_edge: int) -> np.ndarray:
    """A filmstrip thumbnail: like preview, but the shrunk decode is kept (NegPy keep_small), so
    redrawing it after an edit costs only the conversion, not decoding the scan again."""
    arr = _processor().render_display_array(
        str(path), to_config(st, False, path, long_edge), source_hash(path), long_edge,
        prefer_gpu=False, fast_decode=True, keep_small=True,
    )
    if arr is None:
        raise RuntimeError(f"NegPy could not render {path.name}")
    return arr


def preview_full(path: Path, st: Settings) -> np.ndarray:
    """The picture at full resolution (a full raw decode, no half-size shortcut): for zooming in.
    Keeps the decode in this thread's memory for the next zoomed render; release_cache() frees it."""
    return _display_render(path, to_config(st, path=path), 1_000_000, False, True)


_exposures: dict = {}


def scan_exposure(path: Path) -> tuple[str, str, str]:
    """The scan's own exposure (the camera that photographed the negative): shutter, aperture, ISO,
    each "" when the file does not say (a manual lens has no aperture). Read once per file."""
    if path in _exposures:
        return _exposures[path]
    shutter = aperture = iso = ""
    try:
        from negpy.features.metadata.exif_read import rational_to_float, rational_to_int
        from negpy.infrastructure.loaders.helpers import read_exif_from_file

        tags = (read_exif_from_file(str(path)) or {}).get("Exif", {}) or {}
        t = rational_to_float(tags.get(33434))                     # ExposureTime
        if t:
            shutter = f"1/{round(1 / t)}" if t < 0.5 else f"{t:g}s"
        f = rational_to_float(tags.get(33437))                     # FNumber (0: a manual lens)
        if f:
            aperture = f"f/{f:g}"
        n = tags.get(34855)                                        # ISOSpeedRatings
        n = n if isinstance(n, int) else rational_to_int(n) if n is not None else None
        if n:
            iso = f"ISO {n}"
    except Exception:
        pass
    _exposures[path] = (shutter, aperture, iso)
    return _exposures[path]


_details: dict = {}


def scan_details(path: Path) -> dict:
    """What the scan file says about itself, for the frame's info: the camera and lens that
    photographed the negative, its size in pixels, the file's size and type. Read once per file."""
    if path in _details:
        return _details[path]
    d = {"camera": "", "lens": "", "size": (0, 0), "bytes": 0, "type": path.suffix.lstrip(".").upper()}
    try:
        d["bytes"] = path.stat().st_size
    except OSError:
        pass
    try:
        from negpy.infrastructure.loaders.helpers import read_exif_from_file
        exif = read_exif_from_file(str(path)) or {}
        text = lambda v: (v.decode(errors="ignore") if isinstance(v, bytes) else str(v or "")).strip("\x00 ")  # noqa: E731
        make, model = text(exif.get("0th", {}).get(271)), text(exif.get("0th", {}).get(272))
        d["camera"] = model if make and model.lower().startswith(make.split()[0].lower()) else f"{make} {model}".strip()
        lens = text(exif.get("Exif", {}).get(42036))                         # LensModel
        d["lens"] = "" if lens.upper().replace("-", "").replace(" ", "") in ("", "NOLENS", "UNKNOWN") or set(lens) <= {"-", "0", " "} else lens
    except Exception:
        pass
    try:
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            from PIL import Image
            with Image.open(path) as im:
                d["size"] = im.size
        else:
            import rawpy
            with rawpy.imread(str(path)) as raw:
                d["size"] = (raw.sizes.width, raw.sizes.height)
    except Exception:
        pass
    _details[path] = d
    return d


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
    ip = _gpu_processors.get(threading.get_ident())
    if ip is not None:
        ip.release_source_cache()


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
    border_px: int = 0               # a white border this many pixels wide around the picture (not on DNG)


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


def _with_border(cfg: WorkspaceConfig, export_cfg: ExportConfig, px: int) -> WorkspaceConfig:
    """A plain white border of `px` pixels, through NegPy's print border. NegPy sizes it in cm at
    the export's own DPI (full resolution: added around the picture; a set size: fitted inside it)."""
    from dataclasses import replace

    from negpy.services.export.print import PrintService

    dpi = PrintService.effective_dpi(export_cfg)
    cm = (px + 0.5) * 2.54 / max(dpi, 1)       # NegPy truncates cm -> px: the half pixel lands it on px
    return replace(cfg, finish=replace(cfg.finish, border_size=cm, border_color="#ffffff",
                                       border_bottom_weight=1.0, border_match_paper=False))


def _negpy_export(path: Path, st: Settings, fmt: ExportFormat, opts: ExportOptions,
                  color_space: str | None = None) -> bytes:
    """Full-resolution render through NegPy, encoded by NegPy (crop/rotation/size applied)."""
    ip = _processor()
    cfg, export_cfg = to_config(st, path=path), _export_config(fmt, opts, color_space)
    if opts.border_px > 0:
        cfg = _with_border(cfg, export_cfg, opts.border_px)
    try:
        data, _ = ip.process_export(str(path), cfg, export_cfg, source_hash(path), prefer_gpu=False)
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


# NegPy's film catalogue: next to the code when run from source, at the bundle root in the packaged app
_GEAR_DIR = (Path(sys._MEIPASS) / "gear" if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")
             else Path(__file__).resolve().parent.parent / "vendor" / "negpy" / "gear")


def frame_bounds(path: Path, st: Settings) -> tuple[tuple, tuple, tuple | None]:
    """Measure one frame exactly as NegPy's renderer does: its per-channel film floors and
    ceils, and its grey axis (where its neutral tones sit; None when it has none)."""
    per_frame = st.copy()
    per_frame.roll_floors = per_frame.roll_ceils = per_frame.roll_axis = per_frame.roll_slope = None
    cfg = to_config(per_frame, uncropped=False)
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


def roll_white_balance(paths: list[Path], progress=None) -> tuple | None:
    """One white balance for a roll's camera RAWs: the median of their as-shot ones (R, 1, B over
    green). None when the roll has no camera RAWs (a scanner's TIFFs carry their own balance).
    `progress(done, total)` is called as it goes, with the files read so far."""
    import rawpy
    found = []
    for i, p in enumerate(paths):
        if progress:
            progress(i, len(paths))
        try:
            with rawpy.imread(str(p)) as raw:
                wb = [float(v) for v in raw.camera_whitebalance[:3]]
        except Exception:
            continue
        if wb[1] > 0 and wb[0] > 0 and wb[2] > 0:
            found.append((wb[0] / wb[1], 1.0, wb[2] / wb[1]))
    if not found:
        return None
    m = np.median(np.array(found), axis=0)
    return (round(float(m[0]), 4), 1.0, round(float(m[2]), 4))


def analyze_roll(frames: list[tuple[Path, Settings]], progress=None) -> tuple[tuple, tuple, tuple, list[bool], tuple | None]:
    """One colour baseline for a whole roll: measure every frame, then pool (NegPy's pool_dye_slopes).

    Returns (floors, ceils, dye slope, outliers, grey axis). Outliers (a sunset, a red wall, a
    frame scanned differently) are left out of the baseline and keep their own measurement.
    `progress(done, total)` is called as it goes, with the frames measured so far.
    """
    from negpy.features.exposure.normalization import pool_dye_slopes, pool_neutral_axis
    from negpy.features.process.models import neutral_axis_tuple

    floors, ceils, axes = [], [], []
    for i, (path, st) in enumerate(frames):
        if progress:
            progress(i, len(frames))
        f, c, a = frame_bounds(path, st)
        floors.append(f)
        ceils.append(c)
        axes.append(a if st.mode == "color" else None)
    pooled, slope, outliers = pool_dye_slopes(np.array(floors), np.array(ceils))
    axis = pool_neutral_axis(axes, outliers)
    return (tuple(pooled.floors), tuple(pooled.ceils), slope, [bool(o) for o in outliers],
            neutral_axis_tuple(axis) if axis is not None else None)


# For a roll whose film is not known: choosing the film is required before a roll is analysed,
# so these let it go ahead (colour: Generic C41 and the average Lab print).
OTHER_STOCKS = [
    {"id": "other-color", "displayName": "Other colour negative", "manufacturer": "", "stockName": "",
     "colorType": "ColorNegative", "notes": "A colour negative not in the list"},
    {"id": "other-bw", "displayName": "Other B&W negative", "manufacturer": "", "stockName": "",
     "colorType": "B&W Negative", "notes": "A black-and-white negative not in the list"},
]


def film_stocks() -> list[dict]:
    """NegPy's film catalogue, negatives only (what SimpleNeg converts), sorted by name; the two
    "Other …" entries come last."""
    data = json.loads((_GEAR_DIR / "film_stocks.json").read_text())
    stocks = [d for d in data if d.get("colorType") in ("ColorNegative", "B&W Negative")]
    return sorted(stocks, key=lambda d: d["displayName"].casefold()) + OTHER_STOCKS


def stock_by_id(stock_id: str) -> dict | None:
    return next((d for d in film_stocks() if d["id"] == stock_id), None)


_MAKERS = {"kodak": "kodak", "fujifilm": "fuji", "fuji": "fuji", "fujicolor": "fuji"}
# Respools of Kodak's cinema film: the same emulsion (Vision3) with the rem-jet backing taken off.
# CineStill 800T is sold as a motion-picture stock rated at 800; it is widely documented as Vision3
# 500T. A respool named with Vision3's own speed and light (Flic Film Cine Colour 250D, Silbersalz35
# 500T) is matched by the CINE_RESPOOLERS rule below.
STOCK_ALIASES = {"CineStill 800T": "Vision3 500T", "CineStill 50D": "Vision3 50D"}
CINE_RESPOOLERS = ("cinestill", "flic film cine", "silbersalz35")


def _words(text: str) -> set[str]:
    import re
    text = re.sub(r"\b[57]2\d\d\b", " ", text.lower().replace("(approx)", ""))    # Kodak's 5207/7207 codes
    return {_MAKERS.get(w, w) for w in re.findall(r"[a-z0-9]+", text)}


def _profile_names(name: str) -> list[tuple[set[str], str]]:
    """What a film profile answers to, per name ("Ektacolor Pro 400 / Portra 400"): its words without
    the maker, unless only a speed would be left ("Fujicolor 400" needs the "fuji"), and its name
    words run together in order ("ultramax"), for stocks that write them as one."""
    import re
    out = []
    for alt in re.sub(r"\b[57]2\d\d/[57]2\d\d\b", " ", name).split("/"):
        w = _words(alt)
        named = {x for x in w if x not in _MAKERS.values()}
        w = named if any(not x.isdigit() for x in named) else w
        order = [_MAKERS.get(x, x) for x in re.findall(r"[a-z0-9]+", alt.lower().replace("(approx)", ""))]
        out.append((w, "".join(x for x in order if x in w and not x.isdigit())))
    return [(w, j) for w, j in out if j]


def crosstalk_for_stock(stock: dict) -> str:
    """The crosstalk profile NegPy ships for this stock, else Generic C41.

    Matched by name: every word of one of the profile's names must be in the stock's name, with the
    maker only when the name is just a speed: "Kodak Gold 200 (approx)" -> Kodak Gold 200, but
    Fujicolor 400 is not every 400 film. "UltraMax" and "Ultra Max" read the same; respools of Vision3
    (STOCK_ALIASES, CINE_RESPOOLERS) find its profile.
    """
    display = stock.get("displayName", "")
    text = f"{display} {stock.get('stockName', '')}"
    text = STOCK_ALIASES.get(display, text)
    if any(r in display.lower() for r in CINE_RESPOOLERS):
        text += " vision3"
    stock_words = _words(text) | {_MAKERS.get(display.split()[0].lower(), "")} if display else _words(text)
    compact = "".join(text.lower().split())
    best, best_len = CrosstalkProfiles.DEFAULT_NAME, 0
    for name in films():
        for w, joined in _profile_names(name):
            hit = w <= stock_words or (joined in compact and {x for x in w if x.isdigit()} <= stock_words)
            if hit and len(w) > best_len:
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

    cfg = to_config(st, path=path, long_edge=long_edge)
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
