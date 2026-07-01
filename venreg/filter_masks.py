"""Drop "shadow" cells from a mask using LOCAL contrast (polarity), not absolute brightness.

Cellpose segments by shape, so with background tissue fluorescence, non-labelled cells get
segmented too — they show up as *shadows* (darker than the surrounding background), whereas
real GCaMP+ cells are *brighter* than their surroundings. So the right test is local
contrast: for each segmented cell, compare its intensity to a local background estimated
from the nearby non-cell pixels, and drop cells that are darker than their surroundings.
(An absolute-brightness cut wrongly removes good cells that sit in dim regions.)

    python -m venreg.filter_masks data/reference_mosaic.tif --channel 1 \
        --mask data/reference_mosaic_ch1_masks.tif
    # -> <mask>_filtered.tif  + <mask>_contrast.png (histogram; keep = contrast >= thresh)

--thresh sets the contrast margin (default 0: keep cells at least as bright as their
surroundings). --surround px sets the local-background window size.
"""
import argparse
import os
import numpy as np
import tifffile

from . import imaging, cells as C


def local_contrast(chan, labels, ids, surround=51):
    """Per-cell (cell_mean - local_background), where local background is the mean of the
    nearby NON-cell pixels (per z-slice) — so dense neighbours don't bias it."""
    from scipy.ndimage import uniform_filter, mean as ndi_mean
    notcell = (labels == 0).astype(np.float32)
    bg = np.empty_like(chan, dtype=np.float32)
    for z in range(chan.shape[0]):
        num = uniform_filter(chan[z] * notcell[z], surround, mode="reflect")
        den = uniform_filter(notcell[z], surround, mode="reflect")
        bg[z] = num / np.maximum(den, 1e-6)
    cell_mean = np.asarray(ndi_mean(chan, labels, ids))
    surround_mean = np.asarray(ndi_mean(bg, labels, ids))
    return cell_mean - surround_mean


def main():
    ap = argparse.ArgumentParser(description="Drop shadow cells (darker than surroundings) from a mask")
    ap.add_argument("tiff", help="the original hyperstack the mask came from")
    ap.add_argument("--mask", required=True, help="the mask TIFF/_seg.npy to filter")
    ap.add_argument("--channel", type=int, required=True, help="channel index the cells were segmented from")
    ap.add_argument("--thresh", type=float, default=0.0, help="keep cells with (cell-surround) >= this (default 0)")
    ap.add_argument("--surround", type=int, default=51, help="local-background window size in px (default 51)")
    ap.add_argument("--out", default=None, help="output mask (default: <mask>_filtered.tif)")
    args = ap.parse_args()

    labels = C.read_masks(open(args.mask, "rb").read(), args.mask)
    w = labels.shape[-1]
    chan = imaging.load_volume(args.tiff, channel=args.channel, work_xy=w, normalize=False)[0]
    if labels.shape[0] != chan.shape[0]:
        raise SystemExit(f"z mismatch: mask {labels.shape} vs channel {chan.shape}")

    ids = np.unique(labels); ids = ids[ids > 0]
    contrast = local_contrast(chan, labels, ids, surround=args.surround)
    keep = contrast >= args.thresh
    drop_ids = ids[~keep]

    filt = labels.copy()
    filt[np.isin(filt, drop_ids)] = 0
    out = args.out or os.path.splitext(args.mask)[0] + "_filtered.tif"
    tifffile.imwrite(out, filt.astype(labels.dtype))
    print(f"cells: {len(ids)} -> kept {int(keep.sum())}  dropped {int((~keep).sum())}  "
          f"(darker-than-surround; thresh={args.thresh}, surround={args.surround}px)")
    print(f"local contrast (cell-surround): p10={np.percentile(contrast,10):.3g} "
          f"p50={np.percentile(contrast,50):.3g} p90={np.percentile(contrast,90):.3g}  "
          f"negative (darker)={int((contrast<0).sum())}")
    print(f"wrote {out}")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        lim = np.percentile(np.abs(contrast), 99)
        ax.hist(np.clip(contrast, -lim, lim), bins=80, color="#0072B2")
        ax.axvline(args.thresh, color="#D55E00", ls="--",
                   label=f"thresh {args.thresh:g}\nkeep {int(keep.sum())} / drop {int((~keep).sum())}")
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("cell − local surround (brighter →)"); ax.set_ylabel("# cells")
        ax.legend(); ax.set_title(os.path.basename(args.mask))
        fig.tight_layout(); png = os.path.splitext(out)[0].replace("_filtered", "") + "_contrast.png"
        fig.savefig(png, dpi=110); print(f"histogram -> {png}")
    except Exception as e:
        print("(no histogram:", e, ")")


if __name__ == "__main__":
    main()
