# venreg — landmark-driven 3D venation registration

A local web app for registering two 3-D TIFF z-stacks of a vessel/venation network
(e.g. the same tissue imaged before vs. after processing, with different stain, scale,
rotation, depth range — **and even a mirror flip**). You place a few corresponding
landmarks in a side-by-side viewer; the app fits the best **rigid** (similarity, with
automatic reflection detection) and best **warping** (affine + diffeomorphic deformable)
registration from those landmarks, lets you inspect the overlay per depth, and saves the
registered volume, the transforms, and the landmarks.

Registration is derived **purely from the landmark correspondences** — no voxel-size
metadata is required, so it works on any pair of stacks.

> Why landmarks + reflection: the venation mesh is dense and self-similar, so fully
> automatic intensity/feature methods fail (they can't tell one loop from another), and
> samples are often **mirror-flipped** during processing — which proper-rotation methods
> can never recover. A handful of confident landmarks plus automatic reflection detection
> solves both. (See `exploration/REGISTRATION_REPORT.md` for the full investigation.)

## Install

Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Launch the web app

```bash
python run.py            # then open http://127.0.0.1:8000/  (use --port to change)
```

Put your `.tif` / `.tiff` stacks in **`data/`** (or use the bundled ones in `example/`);
they appear in the file dropdowns automatically.

## Workflow

1. **Load a pair of TIFFs.** Pick file A and file B in the top bar and click **Load**
   (preprocessing takes ~10–30 s; it builds fluorescence / vesselness / mask /
   intersection representations and per-slice views).
2. **Label corresponding points.** In the two-panel viewer: **wheel = zoom, drag = pan,
   click = drop a landmark, ↑/↓ = change z slice**. Click the same feature in A and in B
   (they share the *current pair #*), then **+ new pair** and repeat. Aim for **≥4 well-
   spread pairs at varied depths**. The live preview (B panel) overlays A on B in real
   time (red = B, green = A, yellow = match) and reports scale / rotation / **FLIP** /
   RMS residual so you can spot a bad pick (large residual in the table).
   - *Tip:* turn on **overlay-pick** to correct the registration directly on the overlay —
     click a green (warped-A) vessel, then the red (B) vessel it belongs on.
3. **Generate registration.** Click **Generate best RIGID** (similarity + reflection +
   z-map) or **Generate best WARP** (reflected affine + 3-D diffeomorphic Demons). The
   result is computed server-side from your landmarks and shown as an overlay; the status
   line reports the **B→A oriented-inlier** score (fraction of B's vessels covered by the
   aligned A with matching orientation; chance ≈ 0.10).
4. **Save.** Choose `rigid` or `warp` and save the **registered TIFF**, the **transform**
   (4×4 matrix JSON + `.npy`, plus the deformation field for warps), or the **landmarks**
   (CSV). Files land in **`output/`**.

Evaluate per **depth slice**, never by max-Z projection (a projection makes vessels at
different depths look aligned when they aren't).

## Example

`example/` contains a ready pair (`405_488_560_640_2um_1025-1.tif` = A,
`JG910_hyperstack_shiftcorrected-1-1.tif` = B) plus the landmarks and fitted transforms:

- `landmarks_fullres.csv`, `landmarks_work256.json`
- `transform_rigid.json` / `_matrix.npy` — flip + scale ≈ 1.30 + rot ≈ −64° + z-flip
- `transform_warp.json` / `_matrix.npy` — reflected affine + deformable
- `A_registered_to_B_rigid.tif`, `A_registered_to_B_warp.tif`

To reproduce in the app: Load A and B from the dropdowns, click **load example
landmarks**, then **Generate best RIGID / WARP**.

## Layout

```
run.py                 launch script
venreg/
  imaging.py           TIFF load + downsample, vesselness / mask / orientation
  registration.py      landmark fits (reflected similarity / affine, z-map), warp, Demons, eval
  server.py            stdlib HTTP API (list / load / register / save) + static serving
web/                   index.html, app.js, style.css  (the viewer)
data/                  put your TIFFs here
example/               bundled demo pair + landmarks + transforms
output/                saved results land here
```

## Notes

- Works in a downsampled working grid (in-plane 256 px) for responsiveness; saved TIFFs
  are at that working resolution.
- Registration is in voxel/pixel space and needs no calibration; the in-plane scale it
  recovers tells you the relative pixel size of the two stacks.
- TIFFs and deformation fields are git-ignored (too large); the small landmark/transform
  files in `example/` are committed.
