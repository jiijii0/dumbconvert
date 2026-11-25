import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import filmsim as fs

STOCKS = {
    "kodak_gold_200": "Kodak Gold 200 (approx)",
    "kodak_ultramax_400": "Kodak Ultra Max 400 (approx)",
    "kodak_portra_160": "Kodak Ektacolor Pro 160 / Portra 160 (approx)",
    "kodak_portra_400": "Kodak Ektacolor Pro 400 / Portra 400 (approx)",
    "kodak_portra_800": "Kodak Ektacolor Pro 800 / Portra 800 (approx)",
    "kodak_ektar_100": "Kodak Ektar 100 (approx)",
    "fujifilm_c200": "Fujicolor C200 (approx)",
    "fujifilm_xtra_400": "Fujicolor Superia X-TRA 400 (approx)",
    "fujifilm_pro_400h": "Fujicolor Pro 400H (approx)",
    "kodak_vision3_50d": "Kodak VISION3 50D 5203/7203 (approx)",
    "kodak_vision3_200t": "Kodak VISION3 200T 5213/7213 (approx)",
    "kodak_vision3_250d": "Kodak VISION3 250D 5207/7207 (approx)",
    "kodak_vision3_500t": "Kodak VISION3 500T 5219/7219 (approx)",
}
if sys.argv[1:]:                                   # only these stocks; the others keep their last results
    STOCKS = {k: v for k, v in STOCKS.items() if k in sys.argv[1:]}
PAPER = {"kodak_gold_200": "kodak_ektacolor_edge", "kodak_ultramax_400": "kodak_ektacolor_edge",
         "kodak_ektar_100": "kodak_supra_endura",
         # cinema stocks shot as stills go through a C-41 lab and onto RA-4 paper, not cinema print film
         "kodak_vision3_50d": "kodak_supra_endura", "kodak_vision3_200t": "kodak_supra_endura",
         "kodak_vision3_250d": "kodak_supra_endura", "kodak_vision3_500t": "kodak_supra_endura"}
LIGHTS = {"white LED 5100K (CIE LED-B4)": fs.ILL["LED-B4"], "RGB LED (CIE LED-RGB1)": fs.ILL["LED-RGB1"]}
CAMS = sorted((fs.S / "specdata").glob("cam_*.csv"))
OUT = fs.S / "simout"
SKIN = [fs.NAMES.index(n) for n in ("dark skin", "light skin")]
COLOURS = list(range(18))

scene = fs.scene_lab()
old = OUT / "results.json"
results = json.load(open(old))["stocks"] if old.exists() and sys.argv[1:] else {}
for key, prof in STOCKS.items():
    film = fs.stock(key)
    # Kodak's own pairing: consumer stocks are "optimized for KODAK EDGE / ROYAL paper" (E-7022, E-7023)
    paper = fs.stock(PAPER.get(key, film["target"]))
    refl = np.array([fs.CHECKER[n] for n in fs.NAMES])
    dens = fs.expose(film, refl)
    T = fs.transmittance(film, dens)
    base_T = fs.transmittance(film, np.zeros((1, 3)))
    pr = fs.print_lab(film, paper, T)
    row = {"print": pr.tolist(), "target": film["target"]}
    for lname, light in LIGHTS.items():
        labs = []
        for cam in CAMS[:2]:
            c = fs.camera(cam)
            rgb = fs.scan(T, light, c)
            b = fs.scan(base_T, light, c)[0]
            p = OUT / f"{key}-{lname[:3]}-{cam.stem}.tif"
            centres = fs.write_negative(p, rgb, b)
            g = p.with_name(p.stem + "-grey.tif")
            fs.write_negative(g, rgb, b, greys_only=True)
            labs.append(fs.dumbconvert_lab(p, centres, prof, {"light": "rgb"} if lname.startswith("RGB") else None, grey_card=g))
        row[lname] = np.mean(labs, 0).tolist()
    results[key] = row
    dc = np.array(row["white LED 5100K (CIE LED-B4)"])
    cp, cd, cs = fs.chroma(pr[COLOURS]), fs.chroma(dc[COLOURS]), fs.chroma(scene[COLOURS])
    print(f"{film['name']:<18} chroma  scene {cs.mean():5.1f}  print {cp.mean():5.1f}  dumbconvert {cd.mean():5.1f}"
          f"  | skin b*: scene {scene[SKIN,2].mean():5.1f} print {pr[SKIN,2].mean():5.1f} dc {dc[SKIN,2].mean():5.1f}"
          f"  | greys a*,b* print {pr[19:23,1].mean():+.1f},{pr[19:23,2].mean():+.1f} dc {dc[19:23,1].mean():+.1f},{dc[19:23,2].mean():+.1f}")
json.dump({"scene": scene.tolist(), "names": fs.NAMES, "stocks": results}, open(OUT / "results.json", "w"))
