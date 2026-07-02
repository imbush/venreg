"""Headless venreg pipeline: pre-register on the venation channel with the most-recent
saved landmarks (best WARP = reflected affine + 3-D diffeomorphic Demons), then apply a
TRU-FACT-style Soma-print match on the cell (GCaMP) channels to register/match cells.

    python -m venreg.pipeline                      # uses the defaults below
    python -m venreg.pipeline --A data/AVG_2X_2_hyperstack_manual.tif --B data/reference_mosaic.tif \
        --veins-a 0 --veins-b 0 --cells-a 0 --cells-b 1

Defaults: landmarks = most recent output/*.csv; masks = <stack>_ch<cells>_masks.tif next
to each stack. Cell masks are shadow-filtered by local contrast unless --no-filter. Writes
matches CSV (+ per-match likelihood ratio / posterior), the transform, and a colored
matched-cell overlay TIFF to output/.
"""
import argparse
import glob
import os
import time
import numpy as np
import tifffile

from . import imaging, registration as R, cells as C
from .filter_masks import local_contrast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")


def _recent_landmarks():
    # only the app's saved landmark files (landmarks_<ts>.csv), not our own outputs
    fs = sorted(glob.glob(os.path.join(OUT, "landmarks_*.csv")), key=os.path.getmtime, reverse=True)
    return fs[0] if fs else None


def _cell_centroids(mask_path, stack, channel, nz_work, do_filter, thresh, surround, work_xy):
    """Load a mask, optionally drop shadow cells (darker than local surround) using the
    stack's channel, and return centroids in the working grid (work_xy) + native labels."""
    native = C.read_masks(open(mask_path, "rb").read(), mask_path)
    mz, my, mx = native.shape
    dropped = 0
    if do_filter:
        chan = imaging.load_volume(stack, channel=channel, work_xy=mx, normalize=False)[0]
        ids = np.unique(native); ids = ids[ids > 0]
        con = local_contrast(chan, native, ids, surround)
        drop = ids[con < thresh]; dropped = len(drop)
        if dropped:
            native = native.copy(); native[np.isin(native, drop)] = 0
    ids, cen, sizes = C.centroids(native)
    if len(cen):
        cen = cen * np.array([work_xy / mx, work_xy / my, nz_work / mz])
    return ids, cen, native, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--A", default="data/AVG_2X_2_hyperstack_manual.tif")
    ap.add_argument("--B", default="data/reference_mosaic.tif")
    ap.add_argument("--veins-a", type=int, default=1, help="A venation channel (registration)")
    ap.add_argument("--veins-b", type=int, default=0, help="B venation channel (registration)")
    ap.add_argument("--cells-a", type=int, default=0, help="A cell/GCaMP (soma) channel")
    ap.add_argument("--cells-b", type=int, default=1, help="B cell/GCaMP channel")
    ap.add_argument("--sp-radius", type=float, default=None, help="Soma-print candidate radius (working px; default auto)")
    ap.add_argument("--sp-m-a", type=int, default=15, help="Soma-print neighbour pool for A")
    ap.add_argument("--sp-m-b", type=int, default=30, help="Soma-print neighbour pool for B (denser)")
    ap.add_argument("--sp-n", type=int, default=10, help="Soma-print vectors scored per pair")
    ap.add_argument("--sp-lr", type=float, default=0.05, help="accept matches with likelihood ratio < this")
    ap.add_argument("--landmarks", default=None, help="landmark CSV (default: most recent in output/)")
    ap.add_argument("--masks-a", default=None, help="A mask TIFF (default: <A>_ch<cells-a>_masks.tif)")
    ap.add_argument("--masks-b", default=None, help="B mask TIFF (default: <B>_ch<cells-b>_masks.tif)")
    ap.add_argument("--no-filter", action="store_true", help="skip the shadow-cell (dark-than-surround) filter")
    ap.add_argument("--filter-thresh", type=float, default=0.0)
    ap.add_argument("--out", default=None, help="output prefix (default: output/pipeline_<timestamp>)")
    ap.add_argument("--no-overlay", action="store_true", help="skip the colored matched-cell overlay TIFF")
    ap.add_argument("--reg-res", type=int, default=imaging.WORK_XY,
                    help="in-plane resolution for the fit + matching (default 256; higher = finer "
                         "but vesselness/Demons get much slower)")
    args = ap.parse_args()
    W = args.reg_res

    A = args.A if os.path.isabs(args.A) else os.path.join(ROOT, args.A)
    B = args.B if os.path.isabs(args.B) else os.path.join(ROOT, args.B)
    lm = args.landmarks or _recent_landmarks()
    if not lm:
        raise SystemExit("no landmark CSV found in output/ — save landmarks in the app first, or pass --landmarks")
    L = np.loadtxt(lm, delimiter=",", skiprows=1, ndmin=2)
    os.makedirs(OUT, exist_ok=True)
    out = args.out or os.path.join(OUT, "pipeline_" + time.strftime("%Y%m%d_%H%M%S"))
    print(f"landmarks: {os.path.relpath(lm, ROOT)}  ({len(L)} pairs)")

    # 1) pre-register on the venation channel: best WARP (affine + Demons), at reg-res W
    print(f"registering veins:  A ch{args.veins_a} <- B ch{args.veins_b}  @ {W}px …")
    t = time.time()
    dsA, infoA = imaging.load_volume(A, channel=args.veins_a, work_xy=W)
    dsB, infoB = imaging.load_volume(B, channel=args.veins_b, work_xy=W)
    nzA, nzB = dsA.shape[0], dsB.shape[0]
    sig = tuple(x * W / imaging.WORK_XY for x in (1.5, 2.5, 3.5))   # scale ridge sigmas to reg-res
    vesA, vesB = imaging.vesselness(dsA, sigmas=sig), imaging.vesselness(dsB, sigmas=sig)
    Lw = L.copy(); Lw[:, [0, 1, 3, 4]] *= W / imaging.WORK_XY   # landmarks are saved in 256 coords
    fit = R.fit_affine(Lw) if len(Lw) >= 3 else R.fit_rigid(Lw)
    M = np.asarray(fit["matrix"])
    outshape = list(dsB.shape)
    field = R.deformable(vesB, R.warp_volume(vesA, M, outshape))
    print(f"  warp fit ({fit['kind']}) + deformable in {time.time()-t:.1f}s")

    # 2) cell centroids from the GCaMP masks (shadow-filtered), in the working grid
    ma = args.masks_a or os.path.splitext(A)[0] + f"_ch{args.cells_a}_masks.tif"
    mb = args.masks_b or os.path.splitext(B)[0] + f"_ch{args.cells_b}_masks.tif"
    for p in (ma, mb):
        if not os.path.isfile(p):
            raise SystemExit(f"mask not found: {p} (run venreg.segment_cells or pass --masks-a/-b)")
    do_filter = not args.no_filter
    idsA, cA, labA, dropA = _cell_centroids(ma, A, args.cells_a, nzA, do_filter, args.filter_thresh, 51, W)
    idsB, cB, labB, dropB = _cell_centroids(mb, B, args.cells_b, nzB, do_filter, args.filter_thresh, 51, W)
    print(f"cells: A ch{args.cells_a} {len(idsA)} (dropped {dropA} shadows)  |  "
          f"B ch{args.cells_b} {len(idsB)} (dropped {dropB} shadows)")

    # 3) Soma-print match through the warp
    print("Soma-print matching …")
    t = time.time()
    pairs, info = C.soma_print_match(cA, cB, M, field, m_a=args.sp_m_a, m_b=args.sp_m_b,
                                     n=args.sp_n, radius=args.sp_radius, lr_thresh=args.sp_lr)
    probs = np.array([info[p]["prob_correct"] for p in pairs]) if pairs else np.array([])
    print(f"  matched {len(pairs)} cells in {time.time()-t:.1f}s  "
          f"(A {len(idsA)} / B {len(idsB)}; median P(correct)={np.median(probs):.3f})" if pairs else "  no matches")

    # 4) save results
    rows = []
    for (i, j) in pairs:
        rows.append([int(idsA[i]), int(idsB[j]), *np.round(cA[i], 2), *np.round(cB[j], 2),
                     round(info[(i, j)]["lr"], 5), round(info[(i, j)]["prob_correct"], 5)])
    csv = out + "_cellmatches.csv"
    np.savetxt(csv, np.array(rows) if rows else np.zeros((0, 10)), delimiter=",",
               header="idA,idB,Ax,Ay,Az,Bx,By,Bz,likelihood_ratio,prob_correct", comments="", fmt="%.5g")
    np.save(out + "_transform_matrix.npy", M)
    import SimpleITK as sitk
    sitk.WriteImage(field, out + "_deformation.nii.gz")
    print(f"  matches   -> {os.path.relpath(csv, ROOT)}")
    print(f"  transform -> {os.path.relpath(out + '_transform_matrix.npy', ROOT)} (+ _deformation.nii.gz)")

    if not args.no_overlay and pairs:
        from scipy.ndimage import zoom
        labAw = zoom(labA, (nzA / labA.shape[0], W / labA.shape[1], W / labA.shape[2]), order=0).astype(np.int32)
        labBw = zoom(labB, (nzB / labB.shape[0], W / labB.shape[1], W / labB.shape[2]), order=0).astype(np.int32)
        colA, colB = C.assign_colors(idsA, idsB, pairs)
        rgbA, rgbB = C.colorize(labAw, colA), C.colorize(labBw, colB)
        warpedA = np.stack([R.apply_warp(rgbA[..., k], M, outshape, field) for k in range(3)], axis=-1)
        overlay = np.maximum(rgbB, warpedA)                       # matched pairs share a hue, coincide
        ov = (np.clip(overlay, 0, 1) * 255).astype(np.uint8).transpose(0, 3, 1, 2)  # ZCYX
        tifffile.imwrite(out + "_matched_overlay.tif", ov, imagej=True, metadata={"axes": "ZCYX"})
        print(f"  overlay   -> {os.path.relpath(out + '_matched_overlay.tif', ROOT)}")


if __name__ == "__main__":
    main()
