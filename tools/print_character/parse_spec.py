"""Pull literal spectral tables out of colour-science source text (parsed, never executed)."""
import ast, json, sys
import numpy as np
D = sys.argv[1]
def lit(path, name):
    tree = ast.parse(open(f"{D}/{path}").read())
    for n in tree.body:
        t = n.targets[0] if isinstance(n, ast.Assign) else getattr(n, "target", None)
        if isinstance(t, ast.Name) and t.id == name:
            return ast.literal_eval(n.value)
    raise KeyError(name)
wl = np.arange(380, 781, 5.0)
def onto(d):
    k = np.array(sorted(d)); v = np.array([d[x] for x in k], float)
    return np.stack([np.interp(wl, k, v[:, i] if v.ndim > 1 else v) for i in range(v.shape[1] if v.ndim > 1 else 1)], -1)
out = {"wl": wl.tolist()}
cmf = lit("cmfs.py", "DATA_CMFS_STANDARD_OBSERVER")
out["cmf"] = onto(cmf["CIE 1931 2 Degree Standard Observer"]).tolist()
basis = lit("sds_d_illuminant_series.py", "DATA_BASIS_FUNCTIONS_CIE_ILLUMINANT_D_SERIES")
out["S"] = [onto(basis[k])[:, 0].tolist() for k in ("S0", "S1", "S2")]
ill = lit("illum.py", "DATA_ILLUMINANTS_CIE")
print("illuminants:", [k for k in ill][:40])
for k in ("D50", "D55", "D65", "A", "FL3.15", "LED-B1", "LED-B2", "LED-B3", "LED-B4", "LED-B5", "LED-BH1", "LED-RGB1", "LED-V1", "LED-V2"):
    if k in ill: out["ill_" + k] = onto(ill[k])[:, 0].tolist()
tree = ast.parse(open(f"{D}/checker.py").read())
names = [n.targets[0].id if isinstance(n, ast.Assign) else n.target.id for n in tree.body
         if isinstance(n, (ast.Assign, ast.AnnAssign)) and isinstance(getattr(n, "targets", [getattr(n, "target", None)])[0], ast.Name)]
print("checker tables:", [n for n in names if n.startswith("DATA")])
for nm in names:
    if nm == "DATA_BABELCOLOR_AVERAGE":
        try:
            t = lit("checker.py", nm)
            out["checker"] = {k: onto(v)[:, 0].tolist() for k, v in t.items()}; out["checker_src"] = nm
            break
        except Exception as e:
            print("skip", nm, e)
print("checker from", out.get("checker_src"), len(out.get("checker", {})))
json.dump(out, open(f"{D}/spectral.json", "w"))
