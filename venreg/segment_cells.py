"""Generate Cellpose cell-mask files to import into venreg's Step 5.

Runs **outside** the web app (it's the only thing that needs Cellpose). It reads one
channel of a TIFF hyperstack at a chosen in-plane resolution (using the same lazy,
memory-safe loader as the app, so multi-GB mosaics are fine), runs Cellpose, and writes a
3-D instance-label TIFF that you then load with **load masks A / B** in the viewer.

    pip install cellpose            # only needed to run this tool
    python -m venreg.segment_cells data/AVG_2X_2_hyperstack_manual.tif --channel 1
    python -m venreg.segment_cells data/reference_mosaic.tif --channel 3 --res 768

Defaults to 2-D-per-slice + z-stitch (fast, yields 3-D labels); --mode 3d is far slower.
Uses Apple MPS / CUDA when available; --cpu forces CPU. The mask covers the same field and
z-range as the stack, so the app can resample it onto its working grid.
"""
import argparse
import os
import numpy as np
import tifffile

from . import imaging


def main():
    ap = argparse.ArgumentParser(description="Cellpose-segment one channel of a TIFF stack -> label TIFF")
    ap.add_argument("tiff", help="path to the TIFF hyperstack")
    ap.add_argument("--channel", type=int, default=0, help="channel index to segment (the somata/nuclei channel)")
    ap.add_argument("--res", type=int, default=512, help="in-plane size to segment at (default 512)")
    ap.add_argument("--mode", choices=["2d", "3d"], default="2d", help="2d = per-slice + z-stitch (fast); 3d = volumetric (slow)")
    ap.add_argument("--diameter", type=float, default=None, help="expected cell diameter in px (default: auto)")
    ap.add_argument("--cpu", action="store_true", help="force CPU (default uses MPS/CUDA if available)")
    ap.add_argument("--out", default=None, help="output label TIFF (default: <stack>_ch<c>_masks.tif next to input)")
    args = ap.parse_args()

    info = imaging.tiff_info(args.tiff)
    print(f"{os.path.basename(args.tiff)}: axes={info['axes']} shape={info['shape']} "
          f"channels={info['n_channels']}  -> segmenting channel {args.channel} at {args.res}px ({args.mode})")
    vol = imaging.load_volume(args.tiff, channel=args.channel, work_xy=args.res)[0]   # (nz, res, res) in [0,1]
    img = (np.clip(vol, 0, 1) * 255).astype(np.uint8)

    from cellpose import models
    if hasattr(models, "Cellpose"):                    # Cellpose v2 / v3
        model = models.Cellpose(gpu=not args.cpu, model_type="cyto3")
        kw = dict(channels=[0, 0], diameter=args.diameter or 0)
    else:                                              # v4 / cellpose-SAM
        model = models.CellposeModel(gpu=not args.cpu)
        kw = dict(diameter=args.diameter)
    if args.mode == "3d":
        masks = model.eval(img, do_3D=True, z_axis=0, **kw)[0]
    else:
        masks = model.eval(img, do_3D=False, z_axis=0, stitch_threshold=0.5, **kw)[0]
    masks = np.asarray(masks).astype(np.int32)

    out = args.out or os.path.splitext(args.tiff)[0] + f"_ch{args.channel}_masks.tif"
    tifffile.imwrite(out, masks)
    print(f"{int(masks.max())} cells -> {out}")


if __name__ == "__main__":
    main()
