import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import filmsim as fs
from negpy.features.lab.logic import HSL_BANDS, _band_weights, _XYZ_TO_LMS, _LMS_TO_LAB

R = json.load(open(fs.S / "simout/results.json"))
D65 = np.array([95.047, 100.0, 108.883])

def lab_to_xyz(L, white):
    fy = (L[..., 0] + 16) / 116; fx = fy + L[..., 1] / 500; fz = fy - L[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    t = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29))
    return t * white

def oklab(L):
    X = fs.adapt(lab_to_xyz(np.asarray(L), fs.D50W), fs.D50W, D65) / 100
    return np.cbrt(X @ _XYZ_TO_LMS.T) @ _LMS_TO_LAB.T

GREYS = slice(19, 23); COL = np.arange(18)
def neutralise(o):
    o = o.copy(); o[:, 1:] -= o[GREYS, 1:].mean(0); return o

def band_stats(src, dst):
    """Per mixer band: hue turn (deg) and chroma ratio taking src to dst, weighted by band membership."""
    hs, cs = np.degrees(np.arctan2(src[:, 2], src[:, 1])), np.hypot(src[:, 1], src[:, 2])
    hd, cd = np.degrees(np.arctan2(dst[:, 2], dst[:, 1])), np.hypot(dst[:, 1], dst[:, 2])
    k0, k1, s = _band_weights(hs)
    W = np.zeros((len(src), 8)); W[np.arange(len(src)), k0] += 1 - s; W[np.arange(len(src)), k1] += s
    dh = (hd - hs + 180) % 360 - 180; lr = np.log(cd / cs)
    out = []
    for b in range(8):
        w = W[:, b]
        out.append((np.nan, np.nan, 0) if w.sum() < 0.3 else ((w @ dh) / w.sum(), np.exp((w @ lr) / w.sum()), w.sum()))
    return out, np.exp(lr.mean())

scene = oklab(np.array(R["scene"]))[COL]
print("ColorChecker colours (18), Oklab. dc = DumbConvert today, white LED + 2 cameras, greys neutralised\n")
rows = {}
for k, r in R["stocks"].items():
    pr = neutralise(oklab(np.array(r["print"])))[COL]
    dw = neutralise(oklab(np.array(r["white LED 5100K (CIE LED-B4)"])))[COL]
    dr = neutralise(oklab(np.array(r["RGB LED (CIE LED-RGB1)"])))[COL]
    bands, sat = band_stats(dw, pr)
    _, rgb_vs_white = band_stats(dw, dr)
    rows[k] = dict(bands=bands, sat=sat, rgb=rgb_vs_white, pr=pr, dw=dw)
mean_pr = np.mean([v["pr"] for v in rows.values()], 0)
hdr = "".join(f"{b[:6]:>13}" for b in HSL_BANDS)
print(f"{'stock':<20}{'sat':>6}{'rgb/w':>7}  per band: hue turn deg / chroma x   (print vs dc)\n{'':<33}{hdr}")
for k, v in rows.items():
    cells = "".join("          -  " if np.isnan(h) else f"{h:+6.1f}/{c:4.2f}  " for h, c, _ in v["bands"])
    print(f"{k:<20}{v['sat']:6.2f}{v['rgb']:7.2f}  {cells}")
print("\nStock character relative to the average stock (print vs mean print):")
for k, v in rows.items():
    bands, sat = band_stats(mean_pr, v["pr"])
    cells = "".join("          -  " if np.isnan(h) else f"{h:+6.1f}/{c:4.2f}  " for h, c, _ in bands)
    print(f"{k:<20}{sat:6.2f}         {cells}")
