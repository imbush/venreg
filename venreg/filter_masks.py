"""Drop dim cells from a mask by measuring each cell's signal in its source channel.

Cellpose segments by shape, not brightness, so background fluorescence makes dim "shadow"
cells get segmented alongside the real bright ones. This measures every segmented cell's
intensity in the channel it came from (above a local background estimate) and keeps only
cells above a threshold -- no re-segmentation needed.

    python -m venreg.filter_masks data/reference_mosaic.tif --channel 1 \
        --mask data/reference_mosaic_ch1_masks.tif
    # -> <mask>_filtered.tif  + a <mask>_intensity.png histogram to check the threshold

Threshold: Otsu on the per-cell signal by default; override with --thresh T, or
--keep-percentile P (drop the dimmest P%). --no-bg measures raw intensity (no background
subtraction). --bg-sigma sets the background length scale (px).
"""
import argparse
import os
import numpy as np
import tifffile

from . import imaging, cells as C


def main():
    ap = argparse.ArgumentParser(description="Filter dim cells out of a mask by channel intensity")
    ap.add_argument("tiff", help="the original hyperstack the mask came from")
    ap.add_argument("--mask", required=True, help="the mask TIFF/_seg.npy to filter")
    ap.add_argument("--channel", type=int, required=True, help="channel index the cells were segmented from")
    ap.add_argument("--thresh", type=float, default=None, help="keep cells with signal >= this (default: Otsu)")
    ap.add_argument("--keep-percentile", type=float, default=None, help="instead, drop the dimmest P%% of cells")
    ap.add_argument("--bg-sigma", type=float, default=25.0, help="background length scale in px (0 = raw)")
    ap.add_argument("--no-bg", action="store_true", help="measure raw intensity (no background subtraction)")
    ap.add_argument("--out", default=None, help="output mask (default: <mask>_filtered.tif)")
    args = ap.parse_args()

    labels = C.read_masks(open(args.mask, "rb").read(), args.mask)
    w = labels.shape[-1]
    chan = imaging.load_volume(args.tiff, channel=args.channel, work_xy=w, normalize=False)[0]
    if labels.shape[0] != chan.shape[0]:
        raise SystemExit(f"z mismatch: mask {labels.shape} vs channel {chan.shape}")

    sig_img = chan
    if not args.no_bg and args.bg_sigma > 0:                 # subtract per-slice low-freq background
        from scipy.ndimage import gaussian_filter
        bg = np.stack([gaussian_filter(chan[z], args.bg_sigma) for z in range(chan.shape[0])])
        sig_img = np.clip(chan - bg, 0, None)

    from scipy import ndimage as ndi
    ids = np.unique(labels); ids = ids[ids > 0]
    signal = np.asarray(ndi.mean(sig_img, labels, ids))      # per-cell mean signal

    if args.keep_percentile is not None:
        thr = np.percentile(signal, args.keep_percentile)
    elif args.thresh is not None:
        thr = args.thresh
    else:
        from skimage.filters import threshold_otsu
        thr = float(threshold_otsu(signal))
    keep = signal >= thr
    drop_ids = ids[~keep]

    filt = labels.copy()
    filt[np.isin(filt, drop_ids)] = 0
    out = args.out or os.path.splitext(args.mask)[0] + "_filtered.tif"
    tifffile.imwrite(out, filt.astype(labels.dtype))
    print(f"cells: {len(ids)} -> kept {int(keep.sum())}  dropped {int((~keep).sum())}  "
          f"(threshold={thr:.3g}, {'raw' if args.no_bg else f'bg-sub sigma={args.bg_sigma}'})")
    print(f"per-cell signal percentiles: p10={np.percentile(signal,10):.3g} p50={np.percentile(signal,50):.3g} "
          f"p90={np.percentile(signal,90):.3g} max={signal.max():.3g}")
    print(f"wrote {out}")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(signal, bins=80, color="#0072B2")
        ax.axvline(thr, color="#D55E00", ls="--", label=f"threshold {thr:.3g}\nkeep {int(keep.sum())} / drop {int((~keep).sum())}")
        ax.set_xlabel("per-cell signal" + ("" if args.no_bg else " (background-subtracted)"))
        ax.set_ylabel("# cells"); ax.legend(); ax.set_title(os.path.basename(args.mask))
        fig.tight_layout(); png = os.path.splitext(out)[0].replace("_filtered", "") + "_intensity.png"
        fig.savefig(png, dpi=110); print(f"histogram -> {png}")
    except Exception as e:
        print("(no histogram:", e, ")")


if __name__ == "__main__":
    main()
