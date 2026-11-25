import math
from typing import Callable, Optional

import negpy_core
import numpy as np

from negpy.domain.types import ImageBuffer
from negpy.features.lab.models import LabConfig, SharpenMethod
from negpy.kernel.image.logic import (
    WORKING_INV_GAMMA,
    WORKING_LAB,
    working_oetf_decode,
    working_oetf_encode,
)


def apply_clahe(img: ImageBuffer, strength: float) -> ImageBuffer:
    """
    L-channel Contrast Limited Adaptive Histogram Equalization.
    Fixed 8x8 tile grid over the full frame at every scale; mirrors the
    clahe_*.wgsl shaders bin-for-bin so CPU and GPU stay in parity.
    """
    if strength <= 0:
        return img
    return negpy_core.clahe(np.ascontiguousarray(img, dtype=np.float32), strength, *WORKING_LAB)


# Sharpen constants, mirrored as WGSL consts in shaders/lab.wgsl; the USM gate, overshoot and
# edge-mask limits live in negpy_core/src/lab.rs.
# Gain rolls off toward paper black, where the thin negative's grain lives: floor at
# L* 0, full gain from SHARPEN_SHADOW_L_HI up (Gallagher & Gindele, US 7,228,004).
SHARPEN_SHADOW_FLOOR = 1.0 / 3.0
SHARPEN_SHADOW_L_HI = 35.0


def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """
    Single source of truth for the sharpen blur taps: the CPU path convolves
    with this array and the GPU path uploads it verbatim (sharpen_k buffer),
    so kernel support and weights match bit-for-bit on both sides.
    """
    r = max(1, min(255, int(math.ceil(2.5 * sigma))))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x * x) / np.float32(2.0 * sigma * sigma)).astype(np.float32)
    return k / np.float32(k.sum())


# Richardson-Lucy noise floor (linear luminance), mirroring RL_EPS in the WGSL shaders.
RL_EPS = 1e-6
# Adobe RGB (1998) -> XYZ D65 luminance row (Yn = 1). Mirrors lab_sharpen_h.wgsl / rl_init.wgsl.
LUM_R, LUM_G, LUM_B = 0.2973769, 0.6273491, 0.0752741


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - np.float32(e0)) / np.float32(e1 - e0), 0.0, 1.0)
    return t * t * (np.float32(3.0) - np.float32(2.0) * t)


def sharpen_shadow_gain(l_chan: np.ndarray) -> np.ndarray:
    """Per-pixel sharpen gain multiplier from L*; mirrors sharpen_shadow_gain in lab.wgsl."""
    f = np.float32(SHARPEN_SHADOW_FLOOR)
    return f + (np.float32(1.0) - f) * _smoothstep(0.0, SHARPEN_SHADOW_L_HI, l_chan)


def rl_iterations(radius: float) -> int:
    """Deterministic RL iteration count from the user radius (not the scaled σ),
    so preview and export run identical counts. Shared by CPU and GPU."""
    return int(np.clip(int(round(10.0 * radius)), 5, 20))


def _map_row_blocks(img: ImageBuffer, fn: Callable[[ImageBuffer], ImageBuffer], halo: int = 0) -> ImageBuffer:
    rows = max(1, 1048576 // img.shape[1])
    if img.shape[0] <= rows:
        return fn(img)
    output = np.empty(img.shape, dtype=np.float32)
    for start in range(0, img.shape[0], rows):
        end = min(start + rows, img.shape[0])
        low, high = max(0, start - halo), min(img.shape[0], end + halo)
        block = fn(img[low:high])
        output[start:end] = block[start - low : end - low]
    return output


def apply_usm_sharpening(img: ImageBuffer, amount: float, radius: float = 1.0, masking: float = 0.0) -> ImageBuffer:
    if amount <= 0:
        return img
    # Gaussian support, local range, and Sobel-plus-box support must cross block edges.
    halo = max(len(gaussian_kernel_1d(radius)) // 2, 2)
    return _map_row_blocks(img, lambda block: _usm_sharpening_block(block, amount, radius, masking), halo)


# Auto Masking: the masking that holds back the grain is this gain times the measured grain
# gradient. sharpen_grain_pick.wgsl mirrors both.
AUTO_MASKING_GAIN = 0.27
NEUTRAL_MASKING = LabConfig().sharpen_masking


def auto_masking(img: ImageBuffer, config: LabConfig) -> Optional[float]:
    """The masking Auto Masking picks for this buffer, or None when it is off. Without a measure it
    picks the neutral masking."""
    if not config.sharpen_auto_masking:
        return None
    lum = (np.float32(LUM_R), np.float32(LUM_G), np.float32(LUM_B))
    grain = negpy_core.sharpen_grain(np.ascontiguousarray(img, dtype=np.float32), lum)
    return NEUTRAL_MASKING if grain is None else min(AUTO_MASKING_GAIN * grain, 1.0)


def effective_masking(config: LabConfig, auto: Optional[float]) -> float:
    if auto is None:
        return config.sharpen_masking
    return min(max(auto + config.sharpen_masking - NEUTRAL_MASKING, 0.0), 1.0)


def apply_lab_sharpening(img: ImageBuffer, config: LabConfig, auto: Optional[float] = None) -> ImageBuffer:
    """The Lab Sharpening of `config` on linear working RGB, at this buffer's pixel scale. `auto`
    is auto_masking of this buffer."""
    sharpen = apply_rl_sharpening if config.sharpen_method == SharpenMethod.RL else apply_usm_sharpening
    return sharpen(img, config.sharpen, radius=config.sharpen_radius, masking=effective_masking(config, auto))


def apply_resize_sharpening(encoded: np.ndarray, config: LabConfig) -> np.ndarray:
    """
    The Lab Sharpening on export content already scaled to its final size, so its radius is in
    exported pixels. Takes and returns working-OETF code values; a channel after RGB passes through.
    """
    if config.sharpen <= 0:
        return encoded
    linear = working_oetf_decode(np.ascontiguousarray(encoded[..., :3]))
    sharpened = working_oetf_encode(apply_lab_sharpening(linear, config, auto_masking(linear, config)))
    if encoded.shape[2] == 3:
        return sharpened
    return np.concatenate([sharpened, encoded[..., 3:]], axis=2)


def _usm_sharpening_block(
    img: ImageBuffer,
    amount: float,
    radius: float = 1.0,
    masking: float = 0.0,
) -> ImageBuffer:
    """L-channel soft-gated USM, its overshoot clamped to the local 3x3 range, with an optional edge mask for flat
    areas. Radius is in pixels of this buffer, before any export resize; judge sharpening at 1:1.
    lab_sharpen_h/v.wgsl and lab.wgsl mirror this."""
    if amount <= 0:
        return img
    k = gaussian_kernel_1d(radius)
    return negpy_core.usm_sharpen(np.ascontiguousarray(img, dtype=np.float32), np.float32(amount * 2.5), masking, k, *WORKING_LAB)


def apply_rl_sharpening(
    img: ImageBuffer,
    amount: float,
    radius: float = 1.0,
    masking: float = 0.0,
) -> ImageBuffer:
    """Richardson-Lucy deconvolution of linear Y (Gaussian PSF of width radius), applied as an RGB ratio so chroma
    holds. Iterations are fixed by radius and the edge mask governs grain, so there is no damping.
    rl_*.wgsl mirrors this."""
    if amount <= 0:
        return img
    lum = (np.float32(LUM_R), np.float32(LUM_G), np.float32(LUM_B))
    k = gaussian_kernel_1d(radius)
    return negpy_core.rl_sharpen(
        np.ascontiguousarray(img, dtype=np.float32), np.float32(amount), masking, k, rl_iterations(radius), lum, np.float32(RL_EPS)
    )


def apply_saturation(img: ImageBuffer, saturation: float, skin_protection: float = 0.0) -> ImageBuffer:
    """
    Scales chroma in Oklab, which holds hue where CIELAB turns blue toward purple, and keeps the
    pixel's CIELAB L*, so the tone tools and the sharpen state read the same lightness. Above 1.0
    a pixel whose full push leaves the gamut takes a soft knee toward its own headroom, because
    an RGB clamp shifts hue. Skin Protection then reins skin chroma in CIELAB; it runs at 1.0 too.
    """
    if saturation == 1.0 and skin_protection <= 0.0:
        return img
    return negpy_core.saturation(
        np.ascontiguousarray(img, dtype=np.float32), np.float32(saturation), np.float32(skin_protection), *WORKING_LAB
    )


def apply_exposure(img: ImageBuffer, ev: float) -> ImageBuffer:
    """Exposure in stops on the linear positive, like a raw converter's: everything scales by 2^ev.
    Brighter, the values that would pass white roll off into it instead of clipping, above a knee
    that comes down from white as the gain grows (no change at all at 0). lab.wgsl mirrors it."""
    if ev == 0.0:
        return img
    g = np.float32(2.0 ** ev)
    y = np.maximum(np.asarray(img, dtype=np.float32), 0.0) * g
    if g <= 1.0:
        return y
    k = np.float32(1.0 - 0.25 * min(1.0, (float(g) - 1.0) / 0.5))
    span = np.float32(max(1.0 - float(k), 1e-6))
    return np.where(y > k, k + span * (1.0 - np.exp(-(y - k) / span)), y).astype(np.float32)


TONE_LUMA = np.array([0.2974, 0.6273, 0.0753], dtype=np.float32)   # the working space's luminance row
TONE_RANGE = (0.18, 0.18, 0.15, 0.15)    # highlights, shadows, whites, blacks: lightness change at a slider's end


def apply_tone(img: ImageBuffer, highlights: float, shadows: float, whites: float, blacks: float) -> ImageBuffer:
    """Lightroom-like tone controls on the linear positive, each -1..1, on lightness only (colours keep
    their chroma). In perceptual lightness t (luminance^(1/2.2)): Shadows bends the lower tones
    (peak at t = 1/3), Highlights the upper tones (peak at t = 2/3), both leaving black and white
    where they are; Whites moves the light end itself, Blacks the dark end. lab.wgsl mirrors it."""
    if not (highlights or shadows or whites or blacks):
        return img
    rgb = np.maximum(np.asarray(img, dtype=np.float32), 0.0)
    y = rgb @ TONE_LUMA
    t = np.power(np.maximum(y, 0.0), 1.0 / 2.2)
    hr, sr, wr, br = TONE_RANGE
    u = 1.0 - t
    t2 = (t + shadows * sr * 6.75 * t * u * u + highlights * hr * 6.75 * t * t * u
          + whites * wr * t * t * t + blacks * br * u * u * u)
    t2 = np.maximum(t2, 0.0)
    y2 = np.power(t2, 2.2)
    ratio = np.where(y > 1e-6, y2 / np.maximum(y, 1e-6), 1.0)
    out = rgb * ratio[..., None]
    lifted = y <= 1e-6                                   # pure black lifted by Blacks: a neutral grey
    if lifted.any():
        out[lifted] = y2[lifted][:, None]
    return np.ascontiguousarray(out, dtype=np.float32)


# Colour mixer bands: their centres on the Oklab hue circle (degrees), in order.
HSL_BANDS = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")
HSL_CENTERS = np.array([29.0, 60.0, 105.0, 142.0, 195.0, 264.0, 300.0, 335.0])
HSL_HUE_DEGREES = 30.0        # the hue shift at the end of its slider
HSL_LUMINANCE = 0.25          # the lightness change (fraction of L) at the end of its slider
# Adobe RGB (1998) linear -> XYZ (D65), then Ottosson's Oklab matrices
_RGB_TO_XYZ = np.array([[0.5767309, 0.1855540, 0.1881852], [0.2973769, 0.6273491, 0.0752741],
                        [0.0270343, 0.0706872, 0.9911085]])
_XYZ_TO_LMS = np.array([[0.8189330101, 0.3618667424, -0.1288597137], [0.0329845436, 0.9293118715, 0.0361456387],
                        [0.0482003018, 0.2643662691, 0.6338517070]])
_LMS_TO_LAB = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                        [0.0259040371, 0.7827717662, -0.8086757660]])
_RGB_TO_LMS = _XYZ_TO_LMS @ _RGB_TO_XYZ
_LMS_TO_RGB = np.linalg.inv(_RGB_TO_LMS)
_LAB_TO_LMS = np.linalg.inv(_LMS_TO_LAB)


def _band_weights(hue: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each pixel's hue: the two neighbouring bands and how much of the second applies (0..1),
    eased so a colour between two bands takes some of each and no band edge shows."""
    centers = HSL_CENTERS
    n = len(centers)
    ext = np.concatenate([centers, [centers[0] + 360.0]])
    h = np.mod(hue - centers[0], 360.0) + centers[0]            # unwrap so the first centre is the start
    k = np.clip(np.searchsorted(ext, h, side="right") - 1, 0, n - 1)
    lo, hi = ext[k], ext[k + 1]
    t = np.clip((h - lo) / (hi - lo), 0.0, 1.0)
    s = t * t * (3.0 - 2.0 * t)                                 # smoothstep
    return k, (k + 1) % n, s


def apply_hsl(img: ImageBuffer, hue: tuple, saturation: tuple, luminance: tuple) -> ImageBuffer:
    """The colour mixer, in Oklab on the linear working space: per hue band, turn the hue (up to
    HSL_HUE_DEGREES), scale its chroma (-1 removes it, +1 doubles it) and lighten or darken it (up to
    HSL_LUMINANCE). Greys have no hue and stay as they are: every change fades out below a small chroma."""
    hue_a, sat_a, lum_a = (np.asarray(v, dtype=np.float32) for v in (hue, saturation, luminance))
    if not (hue_a.any() or sat_a.any() or lum_a.any()):
        return img
    rgb = np.asarray(img, dtype=np.float32)
    lms = np.cbrt(np.maximum(rgb @ _RGB_TO_LMS.T.astype(np.float32), 0.0))
    lab = lms @ _LMS_TO_LAB.T.astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b)
    angle = np.degrees(np.arctan2(b, a))
    k0, k1, s = _band_weights(angle)
    mix = lambda arr: arr[k0] * (1.0 - s) + arr[k1] * s          # noqa: E731
    grey = np.clip(chroma / 0.04, 0.0, 1.0)                     # a colour, not a grey: changes fade in
    dh, ds, dl = mix(hue_a) * grey, mix(sat_a) * grey, mix(lum_a) * grey
    angle = np.radians(angle + dh * HSL_HUE_DEGREES)
    chroma = chroma * np.clip(1.0 + ds, 0.0, 2.0)
    L = L * (1.0 + dl * HSL_LUMINANCE)
    lab = np.stack([L, chroma * np.cos(angle), chroma * np.sin(angle)], axis=-1)
    lms = lab @ _LAB_TO_LMS.T.astype(np.float32)
    out = (lms * lms * lms) @ _LMS_TO_RGB.T.astype(np.float32)
    return np.ascontiguousarray(np.maximum(out, 0.0), dtype=np.float32)


WB_TEMP_STOPS = 0.5    # red up and blue down (warmer) by this many stops at Temp's end
WB_TINT_STOPS = 0.4    # green down (more magenta) by this many stops at Tint's end


def wb_gains(temp: float, tint: float) -> np.ndarray:
    """Temp and Tint as per-channel gains on the linear positive, scaled so a grey keeps its luminance."""
    g = np.exp2(np.array([WB_TEMP_STOPS * temp, -WB_TINT_STOPS * tint, -WB_TEMP_STOPS * temp], dtype=np.float64))
    return (g / float(g @ TONE_LUMA.astype(np.float64))).astype(np.float32)


def apply_wb(img: ImageBuffer, temp: float, tint: float) -> ImageBuffer:
    """White balance like a raw converter's: the channels scale, before exposure and the tones. lab.wgsl
    takes the gains from wb_gains."""
    if not (temp or tint):
        return img
    return np.ascontiguousarray(np.asarray(img, dtype=np.float32) * wb_gains(temp, tint), dtype=np.float32)


# Below this blur sigma (px) both engines skip the stage.
CHROMA_DENOISE_MIN_SIGMA = 0.25


def chroma_denoise_radius(radius: float, mono: bool) -> float:
    """B&W has no chroma to denoise, so the stage never runs there."""
    return 0.0 if mono else radius


def chroma_denoise_sigma(radius: float, scale_factor: float) -> float:
    """The low-pass sigma in render pixels for both engines; 0.0 when the stage does not run."""
    sigma = radius * scale_factor
    return sigma if sigma >= CHROMA_DENOISE_MIN_SIGMA else 0.0


def apply_chroma_denoise(img: ImageBuffer, radius: float, scale_factor: float = 1.0) -> ImageBuffer:
    """
    Band split of CIELAB a*/b* (Fuji, US 5,739,922): chroma above a Gaussian cutoff is noise or
    grain, except where it belongs to a color edge. L* is untouched, so grain stays as luminance.
    """
    sigma = chroma_denoise_sigma(radius, scale_factor)
    if sigma == 0.0:
        return img

    k = gaussian_kernel_1d(sigma)
    return negpy_core.chroma_denoise(np.ascontiguousarray(img, dtype=np.float32), sigma, k, *WORKING_LAB)


# Base-reflection bounces, (weight, sigma over HALATION_SIGMA). Mirrored in shaders/lab.wgsl.
HALATION_SIGMA = 0.0027  # of the short side: 65 um on 35 mm
HALATION_BOUNCES = ((4.0 / 7.0, 1.0), (2.0 / 7.0, math.sqrt(2.0)), (1.0 / 7.0, math.sqrt(3.0)))
HALATION_AMOUNT_MAX = 0.2
HALATION_TINT = (1.0, 0.3, 0.05)


def halation_amounts(strength: float, bw: bool) -> tuple[float, float, float]:
    """Per-channel mix at a Halation value; neutral, same luminance, on B&W."""
    r, g, b = (HALATION_AMOUNT_MAX * float(strength) * t for t in HALATION_TINT)
    if bw:
        y = r * LUM_R + g * LUM_G + b * LUM_B
        return (y, y, y)
    return (r, g, b)


def halation_sigma_px(shape: tuple[int, ...]) -> float:
    return HALATION_SIGMA * min(shape[:2])


def halation_reach_px(shape: tuple[int, ...]) -> int:
    return int(math.ceil(3.0 * HALATION_BOUNCES[-1][1] * halation_sigma_px(shape)))


# Glow: a mist filter's scatter, Gaussian fit of spektrafilm's Black Pro-Mist halo and bloom
# without its core, as (weight, sigma over the short side). Mirrored in shaders/lab.wgsl.
GLOW_PSF = ((0.1104, 0.001582), (0.3559, 0.004122), (0.4014, 0.010181), (0.1323, 0.024798))
GLOW_AMOUNT_MAX = 0.25
GLOW_THRESHOLD = 0.5  # display-encoded luma where the highlights start to scatter


def glow_reach_px(shape: tuple[int, ...]) -> int:
    return int(math.ceil(3.0 * GLOW_PSF[-1][1] * min(shape[:2])))


def apply_glow_and_halation(img: ImageBuffer, glow_amount: float, halation_strength: float, bw: bool = False) -> ImageBuffer:
    """Glow spreads the highlights, halation all light, each I + a·(K*L − L), so a flat area keeps its value."""
    if glow_amount == 0.0 and halation_strength == 0.0:
        return img

    glow = halation = None
    if glow_amount > 0.0:
        a = GLOW_AMOUNT_MAX * float(glow_amount)
        glow = (((a, a, a), [(w, s * min(img.shape[:2])) for w, s in GLOW_PSF]), GLOW_THRESHOLD)
    if halation_strength > 0.0:
        sigma = halation_sigma_px(img.shape)
        halation = (halation_amounts(halation_strength, bw), [(w, k * sigma) for w, k in HALATION_BOUNCES])
    lum = (np.float32(LUM_R), np.float32(LUM_G), np.float32(LUM_B))
    return negpy_core.glow_and_halation(np.ascontiguousarray(img, dtype=np.float32), glow, halation, lum, WORKING_INV_GAMMA)
