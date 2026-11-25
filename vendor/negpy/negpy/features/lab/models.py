from dataclasses import dataclass
from enum import StrEnum


class SharpenMethod(StrEnum):
    USM = "usm"
    RL = "rl"


@dataclass(frozen=True)
class LabConfig:
    """
    Scanner emulation (Sharpening, CLAHE).

    Spectral crosstalk moved to ProcessConfig (capture-side, negative-density
    domain) — `color_separation`/`crosstalk_*` here are migrated by
    WorkspaceConfig.from_flat_dict.
    """

    saturation: float = 1.0
    skin_protection: float = 0.5
    clahe_strength: float = 0.0
    sharpen: float = 0.25
    sharpen_method: SharpenMethod = SharpenMethod.USM
    sharpen_radius: float = 1.0
    sharpen_masking: float = 0.1
    # On, sharpen_masking is an offset on the grain measured in the buffer being sharpened.
    sharpen_auto_masking: bool = False
    chroma_denoise: float = 2.5
    glow_amount: float = 0.0
    halation_strength: float = 0.0
    # Colour mixer: per hue band (HSL_BANDS: red, orange, yellow, green, aqua, blue, purple, magenta),
    # a hue shift, a saturation change and a luminance change, each -1..1. CPU only (see hsl_active).
    hsl_hue: tuple = (0.0,) * 8
    hsl_saturation: tuple = (0.0,) * 8
    hsl_luminance: tuple = (0.0,) * 8
    # Exposure of the positive, in stops, and the four tone ranges (-1..1), applied last
    # (see logic.apply_exposure and logic.apply_tone): Lightroom-like Highlights, Shadows, Whites, Blacks
    exposure_ev: float = 0.0
    tone_highlights: float = 0.0
    tone_shadows: float = 0.0
    tone_whites: float = 0.0
    tone_blacks: float = 0.0
    # White balance on the positive, like a raw converter's Temp and Tint (-1..1, logic.apply_wb)
    wb_temp: float = 0.0
    wb_tint: float = 0.0

    def __post_init__(self) -> None:
        for name in ("hsl_hue", "hsl_saturation", "hsl_luminance"):
            values = tuple(float(v) for v in getattr(self, name))
            object.__setattr__(self, name, (values + (0.0,) * 8)[:8])


def hsl_active(lab: "LabConfig") -> bool:
    """Whether the colour mixer changes anything."""
    return any(v != 0.0 for v in lab.hsl_hue + lab.hsl_saturation + lab.hsl_luminance)
