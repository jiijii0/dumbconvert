"""Per-stock colour character from manufacturer data (spektrafilm's digitised Kodak/Fuji datasheets).

Chain, all spectral (380-780 nm, 5 nm):
  scene  : ColorChecker 24 (BabelColor average) under D55, the film's reference daylight
  film   : spectral sensitivity -> log exposure (mid grey = 0) -> characteristic curves -> dye densities
  scan   : negative transmittance x scan light x camera sensitivities, white balanced on the bare light
           -> 16-bit TIFF -> the real DumbConvert engine (neutral print, roll defaults)
  print  : the same negative on the stock's target paper (manufacturer pairing), tungsten enlarger,
           filtration solved so the grey patch prints neutral -> reflectance under D50
Compared in CIELAB (D50). The difference print - DumbConvert is the stock's designed character.
"""
import json
import sys
from pathlib import Path

import numpy as np

S = Path(__file__).parent
SPEC = json.load(open(S / "specdata/spectral.json"))
WL = np.array(SPEC["wl"])                       # 380..780 step 5, 81 samples
CMF = np.array(SPEC["cmf"])
ILL = {k[4:]: np.array(v) for k, v in SPEC.items() if k.startswith("ill_")}
CHECKER = {k: np.array(v) for k, v in SPEC["checker"].items()}
NAMES = list(CHECKER)
GREY = "neutral 5 (.70 D)"


def fill(a):
    a = np.array(a, dtype=float)
    out = a.copy()
    for k in range(a.shape[1] if a.ndim > 1 else 1):
        col = out[:, k] if a.ndim > 1 else out
        ok = np.isfinite(col)
        col[~ok] = np.interp(WL[~ok], WL[ok], col[ok])
    return out


def stock(name):
    d = json.load(open(S / "spk" / f"{name}.json"))
    a = d["data"]
    return dict(
        name=d["info"]["name"], target=d["info"].get("target_print"),
        sens=10 ** fill(a["log_sensitivity"]),                    # R, G, B layers
        dye=np.nan_to_num(np.array(a["channel_density"], float)),  # C, M, Y dye spectra
        base=fill(np.array(a["base_density"], float)[:, None])[:, 0] if a["base_density"] else np.zeros(81),
        le=np.array(a["log_exposure"]), dc=np.array(a["density_curves"], float),
    )


def camera(path):
    rows = [l.split(",") for l in open(path) if l.strip() and not l.startswith("#")]
    num = []
    for r in rows:
        try:
            num.append([float(x) for x in r[:4]])
        except ValueError:
            pass
    m = np.array(num)
    if m.shape[1] < 4:          # column-major: first row wavelengths
        raise SystemExit(f"unexpected camera file {path}")
    return np.stack([np.interp(WL, m[:, 0], m[:, k], left=0, right=0) for k in (1, 2, 3)], -1)


def xyz(spd, white):
    k = 100.0 / (white @ CMF[:, 1])
    return k * (spd @ CMF)


def lab(XYZ, white_xyz):
    t = XYZ / white_xyz
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])


def adapt(XYZ, src_w, dst_w):
    s, d = BRADFORD @ src_w, BRADFORD @ dst_w
    M = np.linalg.inv(BRADFORD) @ np.diag(d / s) @ BRADFORD
    return XYZ @ M.T


D50W = xyz(ILL["D50"], ILL["D50"])


def scene_lab():
    w = ILL["D55"]
    X = np.array([xyz(CHECKER[n] * w, w) for n in NAMES])
    return lab(adapt(X, xyz(w, w), D50W), D50W)


# DIR couplers (interimage), spektrafilm's default model: donor row -> receiver column, generic magnitudes
COUPLERS = np.diag([0.341, 0.324, 0.273]) + np.array([[0, 0.355, 0.305], [0.154, 0, 0.358], [0.171, 0.225, 0]])
COUPLER_AMOUNT = 1.0


def curves_before_couplers(film, M):
    le, dc = film["le"], film["dc"]
    le0 = le[:, None] - dc @ M
    return np.stack([np.interp(le, le0[:, i], dc[:, i]) for i in range(3)], -1)


def expose(film, refl, light=ILL["D55"], couplers=None):
    """Layer densities (over D-min) for each reflectance; mid grey gives log H = 0 on every layer.
    With couplers, colours inhibit their neighbour layers while greys keep the datasheet curves."""
    amount = COUPLER_AMOUNT if couplers is None else couplers
    H = np.array([(r * light) @ film["sens"] for r in refl])
    Hg = (CHECKER[GREY] * light) @ film["sens"] / 0.95     # BabelColor N5 is ~19 %: put 18 % at 0
    logH = np.log10(H / Hg)
    D = np.stack([np.interp(logH[:, k], film["le"], film["dc"][:, k]) for k in range(3)], -1)
    if amount <= 0:
        return D
    M = COUPLERS * amount
    dc0 = curves_before_couplers(film, M)
    logH0 = logH - D @ M
    return np.stack([np.interp(logH0[:, k], film["le"], dc0[:, k]) for k in range(3)], -1)


def transmittance(film, dens):
    return 10 ** -(film["base"][None] + dens @ film["dye"].T)


def scan(T, light, cam):
    raw = (T * light) @ cam
    bare = light @ cam                    # camera white balance on the light (white-light decode)
    return raw / bare


def print_lab(film, paper, T):
    """Optical print: tungsten enlarger, per-channel filtration solved so N5 prints neutral at its scene L*."""
    A = ILL["A"]
    H = np.log10((T * A) @ paper["sens"])
    target = scene_lab()[NAMES.index(GREY)] * np.array([1, 0, 0])

    def render(c):
        D = np.stack([np.interp(H[:, k] + c[k], paper["le"], paper["dc"][:, k]) for k in range(3)], -1)
        R = 10 ** -(paper["base"][None] + D @ paper["dye"].T)
        return lab(np.array([xyz(r * ILL["D50"], ILL["D50"]) for r in R]), D50W)

    c = -H[NAMES.index(GREY)].copy()
    for _ in range(40):                                   # Gauss-Newton on the grey patch
        g = NAMES.index(GREY)
        f0 = render(c)[g] - target
        J = np.zeros((3, 3))
        for k in range(3):
            dc = np.zeros(3); dc[k] = 1e-3
            J[:, k] = (render(c + dc)[g] - target - f0) / 1e-3
        c -= np.linalg.lstsq(J, f0, rcond=None)[0]
        if np.abs(f0).max() < 1e-3:
            break
    return render(c)


def write_negative(path, rgb_patches, base_rgb, greys_only=False):
    """A scan of a chart negative: light panel border, film border (D-min), 6x4 patches, grey gaps."""
    import tifffile
    h, w = 1200, 1800
    img = np.ones((h, w, 3))
    m = (int(h * 0.08), int(w * 0.07))
    img[int(h * .04):h - int(h * .04), int(w * .035):w - int(w * .035)] = base_rgb
    ih, iw = h - 2 * m[0], w - 2 * m[1]
    pic = np.empty((ih, iw, 3)); pic[:] = rgb_patches[NAMES.index(GREY)]
    centres = []
    for i, p in enumerate(rgb_patches):
        if greys_only:
            p = rgb_patches[18 + (i % 6)]
        r, c = divmod(i, 6)
        y0, x0 = int(ih * (r + 0.1) / 4), int(iw * (c + 0.1) / 6)
        y1, x1 = int(ih * (r + 0.9) / 4), int(iw * (c + 0.9) / 6)
        pic[y0:y1, x0:x1] = p
        centres.append(((m[1] + (x0 + x1) / 2) / w, (m[0] + (y0 + y1) / 2) / h))
    img[m[0]:m[0] + ih, m[1]:m[1] + iw] = pic
    img *= 0.92
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, (np.clip(img, 0, 1) * 65535).astype(np.uint16), photometric="rgb")
    return centres


def srgb_to_lab(rgb8):
    v = rgb8 / 255.0
    lin = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    X = lin @ M.T * 100
    d65 = np.array([95.047, 100.0, 108.883])
    return lab(adapt(X, d65, D50W), D50W)


def dumbconvert_lab(path, centres, film_profile, settings=None, grey_card=None):
    """grey_card: a negative of only the grey scale, same stock and scan. Read as a one-frame roll,
    it fixes the film's colour (like a lab calibrating a stock) so the chart's own colours cannot."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from simpleneg import engine
    st = engine.Settings(film=film_profile, **(settings or {}))
    if grey_card is not None:
        from negpy.features.process.models import neutral_axis_tuple
        fl, ce, ax = engine.frame_bounds(grey_card, st)
        st = engine.Settings(film=film_profile, roll_floors=tuple(fl), roll_ceils=tuple(ce),
                             roll_axis=neutral_axis_tuple(ax) if ax is not None else None, **(settings or {}))
    img = engine.preview(path, st, 1200).astype(float)
    h, w = img.shape[:2]
    out = []
    for cx, cy in centres:
        x, y = int(cx * w), int(cy * h)
        out.append(img[y - 8:y + 8, x - 8:x + 8].reshape(-1, 3).mean(0))
    return srgb_to_lab(np.array(out))


def chroma(L):
    return np.hypot(L[..., 1], L[..., 2])
