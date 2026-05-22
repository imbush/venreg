"""Local web server for venreg: serves the viewer and exposes load / register / save APIs.
Pure stdlib HTTP (no Flask). Run via `python run.py` at the repo root."""
import os, io, json, glob, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
from PIL import Image

from . import imaging, registration as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
SESS = os.path.join(ROOT, ".session")
OUT = os.path.join(ROOT, "output")
DATA_DIRS = [os.path.join(ROOT, "data"), os.path.join(ROOT, "example")]
REPS = ["fluorescence", "vesselness", "mask", "intersection"]
S = {}            # global single-user session
_lock = threading.Lock()


def _png(arr01, size=512):
    im = (np.clip(arr01, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(im).resize((size, size), Image.BILINEAR)


def _write_slices(tag, reps, outdir):
    for rep in REPS:
        d = os.path.join(outdir, tag, rep); os.makedirs(d, exist_ok=True)
        vol = reps[rep]
        for z in range(vol.shape[0]):
            _png(vol[z]).save(os.path.join(d, f"z{z:03d}.png"))


def _prep_volume(path):
    ds, info = imaging.load_volume(path)
    reps = imaging.representations(ds)
    ori = imaging.inplane_orientation(reps["vesselness"])
    return dict(ds=ds, reps=reps, ori=ori, info=info, shape=list(ds.shape))


def api_list():
    files = []
    for d in DATA_DIRS:
        for f in sorted(glob.glob(os.path.join(d, "*.tif")) + glob.glob(os.path.join(d, "*.tiff"))):
            files.append(os.path.relpath(f, ROOT))
    return {"files": files}


def api_session():
    """Report the currently-loaded pair so the browser can resume on open."""
    if "A" not in S:
        return {"loaded": False}
    return {"loaded": True, "A": {"name": S["A"]["name"], "nz": S["A"]["shape"][0], "shape": S["A"]["shape"]},
            "B": {"name": S["B"]["name"], "nz": S["B"]["shape"][0], "shape": S["B"]["shape"]},
            "work_xy": imaging.WORK_XY, "reps": REPS,
            "results": list(S.get("results", {}).keys())}


def api_load(body):
    a, b = body["fileA"], body["fileB"]
    pa = a if os.path.isabs(a) else os.path.join(ROOT, a)
    pb = b if os.path.isabs(b) else os.path.join(ROOT, b)
    with _lock:
        S.clear()
        S["A"] = _prep_volume(pa); S["A"]["name"] = os.path.basename(a)
        S["B"] = _prep_volume(pb); S["B"]["name"] = os.path.basename(b)
        S["results"] = {}
        # write display slices
        import shutil
        if os.path.isdir(os.path.join(SESS, "slices")):
            shutil.rmtree(os.path.join(SESS, "slices"))
        _write_slices("A", S["A"]["reps"], os.path.join(SESS, "slices"))
        _write_slices("B", S["B"]["reps"], os.path.join(SESS, "slices"))
    return {"A": {"name": S["A"]["name"], "nz": S["A"]["shape"][0], "shape": S["A"]["shape"]},
            "B": {"name": S["B"]["name"], "nz": S["B"]["shape"][0], "shape": S["B"]["shape"]},
            "work_xy": imaging.WORK_XY, "reps": REPS}


def _btree():
    from scipy.spatial import cKDTree
    if "_btree" not in S:
        ves = S["B"]["reps"]["vesselness"]; m = ves > 0.12
        pts = np.argwhere(m)[:, ::-1].astype(float)   # x,y,z
        S["_bpts"] = pts; S["_btree"] = cKDTree(pts)
        S["_bori"] = S["B"]["ori"][np.argwhere(m)[:, 0], np.argwhere(m)[:, 1], np.argwhere(m)[:, 2]]
    return S["_btree"], S["_bpts"], S["_bori"]


def _eval_BtoA(M4):
    """Volumetric B->A oriented inlier (are B's vessels covered by aligned A?)."""
    from scipy.spatial import cKDTree
    A = S["A"]; mA = A["reps"]["vesselness"] > 0.12
    aidx = np.argwhere(mA); apts = aidx[:, ::-1].astype(float)
    aori = A["ori"][aidx[:, 0], aidx[:, 1], aidx[:, 2]]
    treeA = cKDTree(apts)
    B = S["B"]; mB = B["reps"]["vesselness"] > 0.12
    bidx = np.argwhere(mB); bpts = bidx[:, ::-1].astype(float)
    bori = B["ori"][bidx[:, 0], bidx[:, 1], bidx[:, 2]]
    Minv = np.linalg.inv(np.asarray(M4))
    r = R.oriented_inlier(bpts, bori, treeA, apts, aori, Minv, A["shape"])
    return r


def api_register(body):
    L = np.asarray(body["landmarks"], float)   # working-voxel coords
    mode = body.get("mode", "rigid")
    if len(L) < 2:
        return {"error": "need >=2 landmark pairs"}
    with _lock:
        outshape = S["B"]["shape"]
        if mode == "rigid":
            fit = R.fit_rigid(L); M = np.asarray(fit["matrix"]); field = None
        else:
            fit = R.fit_affine(L) if len(L) >= 3 else R.fit_rigid(L)
            M = np.asarray(fit["matrix"])
            warpedV = R.warp_volume(S["A"]["reps"]["vesselness"], M, outshape)
            field = R.deformable(S["B"]["reps"]["vesselness"], warpedV)
        # warp every rep into B grid and write overlay slices
        tag = f"regA_{mode}"
        import shutil
        d0 = os.path.join(SESS, "slices", tag)
        if os.path.isdir(d0):
            shutil.rmtree(d0)
        warped_reps = {}
        for rep in REPS:
            warped_reps[rep] = R.apply_warp(S["A"]["reps"][rep], M, outshape, field)
        _write_slices(tag, warped_reps, os.path.join(SESS, "slices"))
        ev = _eval_BtoA(M)
        S["results"][mode] = dict(matrix=M.tolist(), field=field, fit=fit,
                                  warped_fluor=warped_reps["fluorescence"])
        S["results"][mode]["BtoA_oriented_inlier"] = ev["frac"]
    out = {"mode": mode, "tag": tag, "BtoA_oriented_inlier": round(ev["frac"], 3),
           "n_inl": ev["n_inl"], "n_in": ev["n_in"]}
    out.update({k: fit[k] for k in fit if k not in ("matrix", "residuals_px")})
    return out


def api_save(body):
    what = body["what"]; mode = body.get("mode", "rigid")
    os.makedirs(OUT, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    if what == "landmarks":
        L = np.asarray(body["landmarks"], float)
        p = os.path.join(OUT, f"landmarks_{ts}.csv")
        np.savetxt(p, L, delimiter=",", header="xA,yA,zA,xB,yB,zB", comments="")
        return {"saved": os.path.relpath(p, ROOT)}
    res = S.get("results", {}).get(mode)
    if not res:
        return {"error": f"no {mode} result yet — generate it first"}
    if what == "transform":
        p = os.path.join(OUT, f"transform_{mode}_{ts}.json")
        rec = dict(mode=mode, matrix=res["matrix"], fit={k: v for k, v in res["fit"].items() if k != "matrix"},
                   has_deformable=res["field"] is not None)
        json.dump(rec, open(p, "w"), indent=2)
        np.save(os.path.join(OUT, f"transform_{mode}_{ts}_matrix.npy"), np.asarray(res["matrix"]))
        if res["field"] is not None:
            import SimpleITK as sitk
            sitk.WriteImage(res["field"], os.path.join(OUT, f"transform_{mode}_{ts}_field.nii.gz"))
        return {"saved": os.path.relpath(p, ROOT)}
    if what == "tif":
        import tifffile
        p = os.path.join(OUT, f"A_registered_to_B_{mode}_{ts}.tif")
        vol = (np.clip(res["warped_fluor"], 0, 1) * 255).astype(np.uint8)
        tifffile.imwrite(p, vol)
        return {"saved": os.path.relpath(p, ROOT)}
    return {"error": "unknown 'what'"}


ROUTES = {"/api/list": lambda b: api_list(), "/api/load": api_load,
          "/api/register": api_register, "/api/save": api_save}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, data):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store"); self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            path = "/index.html"
        if path == "/api/list":
            return self._send(200, "application/json", json.dumps(api_list()).encode())
        if path == "/api/session":
            return self._send(200, "application/json", json.dumps(api_session()).encode())
        if path.startswith("/slices/"):
            fp = os.path.join(SESS, path[1:])
        else:
            fp = os.path.join(WEB, path.lstrip("/"))
        if os.path.isfile(fp):
            ext = os.path.splitext(fp)[1]
            ctype = {".html": "text/html", ".js": "application/javascript",
                     ".css": "text/css", ".png": "image/png"}.get(ext, "application/octet-stream")
            with open(fp, "rb") as f:
                return self._send(200, ctype, f.read())
        self._send(404, "text/plain", b"not found")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        fn = ROUTES.get(self.path.split("?")[0])
        if not fn:
            return self._send(404, "application/json", b'{"error":"no route"}')
        try:
            res = fn(body)
        except Exception as e:
            import traceback; traceback.print_exc()
            return self._send(500, "application/json", json.dumps({"error": str(e)}).encode())
        self._send(200, "application/json", json.dumps(res).encode())


def serve(port=8000):
    os.makedirs(SESS, exist_ok=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"venreg running →  http://127.0.0.1:{port}/")
    print("Open it in a browser. Ctrl-C to stop.")
    httpd.serve_forever()
