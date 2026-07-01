"""Merge a stack's original channels + all its produced cell masks into one hyperstack.

Builds a ZCYX ImageJ TIFF where each original channel and each mask is a separate channel,
all on the mask's working grid (so the masks overlay the channels exactly). Masks default
to the sibling files named ``<stack>_ch<c>_masks.tif`` (e.g. from venreg.segment_cells).

    python -m venreg.merge_masks data/AVG_2X_2_hyperstack_manual.tif
    python -m venreg.merge_masks data/reference_mosaic.tif --masks data/reference_mosaic_ch1_masks.tif data/reference_mosaic_ch3_masks.tif

Original channels keep raw intensity; mask channels hold the integer cell labels. Output
is uint16 (or uint32 if labels exceed 65535) -> <stack>_merged.tif next to the input.
"""
import argparse
import glob
import os
import numpy as np
import tifffile

from . import imaging, cells as C


def main():
    ap = argparse.ArgumentParser(description="Merge original channels + cell masks into one hyperstack")
    ap.add_argument("tiff", help="the original TIFF hyperstack")
    ap.add_argument("--masks", nargs="*", default=None, help="mask TIFF/_seg.npy files (default: <stack>_ch*_masks.tif)")
    ap.add_argument("--out", default=None, help="output path (default: <stack>_merged.tif)")
    args = ap.parse_args()

    stem = os.path.splitext(args.tiff)[0]
    mask_files = args.masks or sorted(glob.glob(stem + "_ch*_masks.tif"))
    if not mask_files:
        raise SystemExit(f"no mask files found (looked for {stem}_ch*_masks.tif); pass --masks")

    # working grid = the masks' in-plane size (they must all match)
    mvols = [C.read_masks(open(m, "rb").read(), m) for m in mask_files]
    ws = {v.shape[-1] for v in mvols} | {v.shape[-2] for v in mvols}
    if len(ws) != 1:
        raise SystemExit(f"masks have differing in-plane sizes {ws}; re-segment at one --res")
    w = ws.pop()

    info = imaging.tiff_info(args.tiff)
    nz = info["shape"][info["axes"].index("Z")] if "Z" in info["axes"] else 1
    nchan = info["n_channels"]
    print(f"{os.path.basename(args.tiff)}: {nchan} orig channels + {len(mask_files)} masks @ {w}px, {nz} z")

    maxlabel = max((int(v.max()) for v in mvols), default=0)
    dtype = np.uint16 if maxlabel <= 65535 else np.uint32
    hi = np.iinfo(dtype).max

    chans, labels = [], []
    for c in range(nchan):                                  # original channels, raw intensity
        vol = imaging.load_volume(args.tiff, channel=c, work_xy=w, normalize=False)[0]
        chans.append(np.clip(np.rint(vol), 0, hi).astype(dtype))
        lab = info.get("channel_labels")
        labels.append(f"ch{c}" + (f" {lab[c]}" if lab and c < len(lab) else ""))
    for m, v in zip(mask_files, mvols):                     # mask channels, integer labels
        if v.shape[0] != nz:                               # match z if a mask was made on a subset
            from scipy.ndimage import zoom
            v = zoom(v, (nz / v.shape[0], 1, 1), order=0)
        chans.append(v.astype(dtype))
        labels.append("mask:" + os.path.basename(m))

    stack = np.stack(chans, axis=1)                         # (Z, C, Y, X)
    out = args.out or stem + "_merged.tif"
    tifffile.imwrite(out, stack, imagej=True, metadata={"axes": "ZCYX", "Labels": labels})
    print(f"wrote {out}  shape={stack.shape} dtype={dtype.__name__}  ({stack.nbytes/1e6:.0f} MB)")
    for i, l in enumerate(labels):
        print(f"  channel {i}: {l}")


if __name__ == "__main__":
    main()
