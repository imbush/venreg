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
REPS = ["fluorescence"]  # the one server-rendered image; brightness/blur are client-side
DISPLAY_XY = 1024  # fluorescence display slices rendered at this in-plane size: sharp &
                   # zoomable (e.g. a 4000px mosaic -> 4x downsample instead of 16x), while
                   # the vesselness used for registration stays on the cheaper WORK_XY grid.
S = {}            # global single-user session
_lock = threading.Lock()


def _png(arr01, size=512):
    im = (np.clip(arr01, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(im).resize((size, size), Image.BILINEAR)


def _write_slices(tag, fluor, outdir, sub="fluorescence"):
    """Write the per-slice display PNGs for `fluor` (high-res for A/B, warped for a
    registration result) plus a max-Z projection (`proj.png`) for the z-proj overlay,
    under outdir/tag/sub. Brightness/blur are client-side, so only raw intensity is
    rendered; panels use a per-channel sub ('ch{c}'), results use 'fluorescence'."""
    size = fluor.shape[-1]
    d = os.path.join(outdir, tag, sub); os.makedirs(d, exist_ok=True)
    for z in range(fluor.shape[0]):
        _png(fluor[z], size).save(os.path.join(d, f"z{z:03d}.png"))
    _png(fluor.max(axis=0), size).save(os.path.join(d, "proj.png"))


def _prep_volume(path, channel=0):
    ds, info = imaging.load_volume(path, channel=channel)
    reps = imaging.representations(ds)
    ori = imaging.inplane_orientation(reps["vesselness"])
    meta = imaging.tiff_info(path)
    # high-res copy of the chosen channel, for display slices only (not registration);
    # capped at the image's native in-plane size so small stacks aren't upsampled.
    disp_xy = min(DISPLAY_XY, info["orig_shape"][2])
    hires = imaging.load_volume(path, channel=channel, work_xy=disp_xy)[0]
    # auto-exposure: vessel fluorescence is sparse/dim, so suggest a default display gain
    # that maps the 99th pct to ~0.85 (keeps background dark, makes vessels pop). The
    # client seeds the brightness slider with it; registration is unaffected.
    p99 = float(np.percentile(hires, 99))
    dispgain = round(float(np.clip(0.85 / (p99 + 1e-6), 1.0, 6.0)), 2)
    return dict(ds=ds, reps=reps, ori=ori, info=info, shape=list(ds.shape),
                channel=int(channel), hires_fluor=hires, path=path, disp_xy=int(disp_xy),
                nchan=int(meta["n_channels"]), labels=meta.get("channel_labels"), dispgain=dispgain)


def api_list():
    files = []
    for d in DATA_DIRS:
        for f in sorted(glob.glob(os.path.join(d, "*.tif")) + glob.glob(os.path.join(d, "*.tiff"))):
            rel = os.path.relpath(f, ROOT)
            try:
                meta = imaging.tiff_info(f)
            except Exception:
                meta = {"n_channels": 1, "channel_labels": None}
            files.append({"path": rel, "n_channels": meta["n_channels"],
                          "channel_labels": meta.get("channel_labels")})
    return {"files": files}


def api_session():
    """Report the currently-loaded pair so the browser can resume on open."""
    if "A" not in S:
        return {"loaded": False}
    return {"loaded": True, "A": _imeta("A"), "B": _imeta("B"),
            "work_xy": imaging.WORK_XY, "reps": REPS,
            "results": list(S.get("results", {}).keys())}


def api_load(body):
    a, b = body["fileA"], body["fileB"]
    ca, cb = int(body.get("channelA", 0)), int(body.get("channelB", 0))
    pa = a if os.path.isabs(a) else os.path.join(ROOT, a)
    pb = b if os.path.isabs(b) else os.path.join(ROOT, b)
    with _lock:
        S.clear()
        S["A"] = _prep_volume(pa, ca); S["A"]["name"] = os.path.basename(a)
        S["B"] = _prep_volume(pb, cb); S["B"]["name"] = os.path.basename(b)
        S["results"] = {}
        # write display slices
        import shutil
        if os.path.isdir(os.path.join(SESS, "slices")):
            shutil.rmtree(os.path.join(SESS, "slices"))
        _write_slices("A", S["A"].pop("hires_fluor"), os.path.join(SESS, "slices"), f"ch{ca}")
        _write_slices("B", S["B"].pop("hires_fluor"), os.path.join(SESS, "slices"), f"ch{cb}")
    return {"A": _imeta("A"), "B": _imeta("B"), "work_xy": imaging.WORK_XY, "reps": REPS}


def _imeta(v):
    """Per-image metadata for the client (loaded channel + channels available to toggle)."""
    s = S[v]
    return {"name": s["name"], "nz": s["shape"][0], "shape": s["shape"],
            "channel": s["channel"], "nchan": s.get("nchan", 1), "labels": s.get("labels"),
            "dispgain": s.get("dispgain", 1.0)}


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
        # warp A's fluorescence into B's grid and write overlay slices (+ proj)
        tag = f"regA_{mode}"
        import shutil
        d0 = os.path.join(SESS, "slices", tag)
        if os.path.isdir(d0):
            shutil.rmtree(d0)
        warped_fluor = R.apply_warp(S["A"]["reps"]["fluorescence"], M, outshape, field)
        _write_slices(tag, warped_fluor, os.path.join(SESS, "slices"), f"ch{S['A']['channel']}")
        ev = _eval_BtoA(M)
        S["results"][mode] = dict(matrix=M.tolist(), field=field, fit=fit,
                                  warped_fluor=warped_fluor)
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


def api_channel(body):
    """Render display slices for another channel of an already-loaded image, on demand
    (lazy), so the viewer can toggle channels without reloading. Registration still uses
    the channel chosen at load time."""
    which, c = body["which"], int(body["channel"])
    v = S.get(which)
    if not v:
        return {"error": "not loaded"}
    if not (0 <= c < v.get("nchan", 1)):
        return {"error": "bad channel"}
    with _lock:
        d = os.path.join(SESS, "slices", which, f"ch{c}")
        if not os.path.isdir(d):                       # render once, then it's cached on disk
            hires = imaging.load_volume(v["path"], channel=c, work_xy=v["disp_xy"])[0]
            _write_slices(which, hires, os.path.join(SESS, "slices"), f"ch{c}")
    return {"ok": True, "which": which, "channel": c}


def api_result_channel(body):
    """Warp another A channel through a stored registration result, on demand, so the
    result overlay can follow A's displayed channel (registration itself is unchanged)."""
    mode, c = body.get("mode", "rigid"), int(body["channel"])
    res = S.get("results", {}).get(mode)
    A = S.get("A")
    if not res or not A:
        return {"error": f"no {mode} result yet"}
    if not (0 <= c < A.get("nchan", 1)):
        return {"error": "bad channel"}
    tag = f"regA_{mode}"
    with _lock:
        d = os.path.join(SESS, "slices", tag, f"ch{c}")
        if not os.path.isdir(d):                       # warp once, then cached on disk
            volc = imaging.load_volume(A["path"], channel=c, work_xy=imaging.WORK_XY)[0]
            warped = R.apply_warp(volc, np.asarray(res["matrix"]), S["B"]["shape"], res["field"])
            _write_slices(tag, warped, os.path.join(SESS, "slices"), f"ch{c}")
    return {"ok": True, "mode": mode, "channel": c}


def api_landmarks_list():
    """Saved landmark CSVs (working-voxel coords) available to reload, newest first."""
    fs = sorted(glob.glob(os.path.join(OUT, "*.csv")), key=os.path.getmtime, reverse=True)
    return {"files": [os.path.relpath(f, ROOT) for f in fs]}


def api_load_landmarks(body):
    """Read a saved landmark CSV (xA,yA,zA,xB,yB,zB) -> (N,6) rows. Restricted to output/."""
    p = os.path.normpath(os.path.join(ROOT, body["file"]))
    if not (p.startswith(OUT + os.sep) and os.path.isfile(p)):
        return {"error": "file not found"}
    L = np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
    return {"landmarks": L.tolist()}


def api_unwarp(body):
    """Map a point in B's frame back to A using a stored registration result, so an
    overlay-pick made while a rigid/warp result is displayed lands on the correct A
    location. The overlay shows volA[ inv(M) @ (pB + d(pB)) ], so pA = inv(M) @ T(pB)
    where T is the (affine+deformable) resampling transform that produced it."""
    mode = body.get("mode", "rigid")
    res = S.get("results", {}).get(mode)
    if not res:
        return {"error": f"no {mode} result yet — generate it first"}
    pB = (float(body["x"]), float(body["y"]), float(body["z"]))
    qx, qy, qz = pB
    if res.get("field") is not None:                  # add the deformable displacement
        import SimpleITK as sitk
        disp = sitk.DisplacementFieldTransform(sitk.Cast(res["field"], sitk.sitkVectorFloat64))
        qx, qy, qz = disp.TransformPoint(pB)
    pa = np.linalg.inv(np.asarray(res["matrix"], float)) @ np.array([qx, qy, qz, 1.0])
    return {"x": float(pa[0]), "y": float(pa[1]), "z": float(pa[2])}


ROUTES = {"/api/list": lambda b: api_list(), "/api/load": api_load,
          "/api/register": api_register, "/api/save": api_save, "/api/unwarp": api_unwarp,
          "/api/load_landmarks": api_load_landmarks, "/api/channel": api_channel,
          "/api/result_channel": api_result_channel}


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
        if path == "/api/landmarks":
            return self._send(200, "application/json", json.dumps(api_landmarks_list()).encode())
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
