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
   (preprocessing builds the per-slice fluorescence views; vesselness is computed
   internally to drive the registration but isn't shown). Each panel has its own
   **brightness** (auto-set on load to a sensible exposure, since vessel fluorescence is
   dim — adjust as needed), **blur**, and **zoom** sliders, a **z-proj** toggle (additively
   overlays that stack's full-depth max-Z projection for context), and a **marks** toggle
   (hide/show the landmark markers when they obscure the image). These are display-only and
   applied live in the browser.
   - *Multi-channel stacks:* for an ImageJ hyperstack (e.g. `ZCYX`) a **ch** dropdown
     appears next to each file (with the channel's ImageJ label when present) — pick the
     **vessel/venation channel** for each before loading; that channel drives the
     registration. Once loaded, per-panel **channel buttons** let you toggle which channel
     is *displayed* (e.g. switch to DAPI/GCaMP for context while landmarking); other
     channels are rendered on demand. The registration channel is marked with `*`. Changing
     A's channel updates B's overlay too (the live preview and any result overlay show A's
     displayed channel, warped on demand). Generating a registration resets both panels to
     the channels that were registered.
   - *Large mosaics:* loading reads and downsamples **one z-plane at a time** (a multi-GB
     mosaic never loads in full, so peak RAM stays low). The fluorescence view is rendered
     at a high display resolution (up to 1024 px in-plane, capped at the image's native
     size — sharp & zoomable for landmarking) while vesselness and the registration fit
     run on the cheaper 256-px working grid.
2. **Label corresponding points.** In the two-panel viewer: **wheel = zoom, drag = pan,
   click = drop a landmark, ↑/↓ = change z slice**. Click the same feature in A and in B
   (they share the *current pair #*), then **+ new pair** and repeat. Aim for **≥4 well-
   spread pairs at varied depths**. The live preview (B panel) overlays A on B in real
   time (red = B, green = A, yellow = match) and reports scale / rotation / **FLIP** /
   RMS residual so you can spot a bad pick (large residual in the table).
   - *Tip:* turn on **overlay-pick** to correct the registration directly on the overlay —
     click a green (warped-A) vessel, then the red (B) vessel it belongs on. The green
     click is mapped back to its true A location by inverting whatever is currently shown:
     the live similarity preview, or — when a **rigid/warp result** is displayed — that
     result's actual transform (including the deformable field), so the A landmark lands
     correctly.
3. **Generate registration.** Click **Generate best RIGID** (similarity + reflection +
   z-map) or **Generate best WARP** (reflected affine + 3-D diffeomorphic Demons). The
   result is computed server-side from your landmarks and shown as an overlay; the status
   line reports the **B→A oriented-inlier** score (fraction of B's vessels covered by the
   aligned A with matching orientation; chance ≈ 0.10).
4. **Save.** Choose `rigid` or `warp` and save the **registered TIFF**, the **transform**
   (4×4 matrix JSON + `.npy`, plus the deformation field for warps), or the **landmarks**
   (CSV). Files land in **`output/`**. Saved landmarks can be reloaded later via
   **load saved landmarks** next to *2. Landmarks* — a file picker for any `.csv`/`.json`
   landmark file (parsed in the browser).

5. **Cell matching (optional).** Segment the cell (somata/nuclei) channel **outside the
   app** — e.g. the [Cellpose](https://cellpose.org) GUI or your own pipeline, with your
   tuned parameters and hardware — then **load masks A / load masks B** (a label `.tif`,
   e.g. Cellpose `*_cp_masks.tif`, or a Cellpose `_seg.npy`). The app reads the instance
   labels, resamples them to the working grid, and takes cell centroids. **Match cells**
   then pairs them through the chosen `rigid`/`warp` result. Matching is purely
   **geometric** — A's cell centroids are mapped into B's frame by the registration, then
   paired — because A and B are imaged differently (e.g. before vs after fixation), so
   appearance/overlap is unreliable; only *position* (once aligned) is comparable. Three
   matchers to compare: **hungarian** (globally optimal assignment), **mutual-NN**, and
   **geometry-consistent** (keeps matches whose local displacement agrees with their
   neighbours). The **cell view** then colours each matched A↔B pair the same hue
   (unmatched cells dim grey): panel A shows A's cells, panel B shows B's cells with A's
   warped cells overlaid, so matched cells coincide in colour.
   - Segmentation is done externally, so the app has **no Cellpose/PyTorch dependency**.
     Masks are resampled to the 256-px working grid, so a moderate-resolution mask is fine.
   - *Generating masks:* a helper script is bundled (needs `pip install cellpose`, run
     outside the app). It reads one channel of your hyperstack (memory-safe, so the mosaic
     is fine), runs Cellpose, and writes a label TIFF:

     ```bash
     pip install cellpose
     python -m venreg.segment_cells data/AVG_2X_2_hyperstack_manual.tif --channel 1
     python -m venreg.segment_cells data/reference_mosaic.tif --channel 3 --res 768
     # -> <stack>_ch<c>_masks.tif  (load it with "load masks A/B")
     ```

     `--mode 2d` (default) is per-slice + z-stitch (fast); `--mode 3d` is volumetric and
     much slower. Uses MPS/CUDA when available. Or segment in the **Cellpose GUI** /
     your own pipeline and save the masks as a label TIFF or `_seg.npy` — anything that
     reads back as an instance-label volume of the same field works.

Evaluate per **depth slice**, never by max-Z projection (a projection makes vessels at
different depths look aligned when they aren't).

## Example

`example/` contains a ready pair (`405_488_560_640_2um_1025-1.tif` = A,
`JG910_hyperstack_shiftcorrected-1-1.tif` = B) plus the landmarks and fitted transforms:

- `landmarks_fullres.csv`, `landmarks_work256.json`
- `transform_rigid.json` / `_matrix.npy` — flip + scale ≈ 1.30 + rot ≈ −64° + z-flip
- `transform_warp.json` / `_matrix.npy` — reflected affine + deformable
- `A_registered_to_B_rigid.tif`, `A_registered_to_B_warp.tif`

To reproduce in the app: Load A and B from the dropdowns, click **load saved landmarks**
and pick `example/landmarks_work256.json`, then **Generate best RIGID / WARP**.

## Layout

```
run.py                 launch script
venreg/
  imaging.py           TIFF load + downsample, vesselness / orientation
  registration.py      landmark fits (reflected similarity / affine, z-map), warp, Demons, eval
  cells.py             read pre-computed cell masks + geometric cell matching (step 5)
  server.py            stdlib HTTP API (list / load / channel / register / unwarp /
                       result_channel / save) + static serving
web/                   index.html, app.js, style.css  (the viewer)
data/                  put your TIFFs here
example/               bundled demo pair + landmarks + transforms
output/                saved results land here
```

## Notes

- Registration runs on a downsampled working grid (in-plane 256 px, `WORK_XY` in
  `venreg/imaging.py`) for responsiveness — ample for fitting a global transform from a
  handful of landmarks — and saved TIFFs are at that resolution. The **fluorescence
  display** is rendered separately at a higher resolution (`DISPLAY_XY` in
  `venreg/server.py`, default 1024 px) so large mosaics stay sharp; landmarks are placed
  at sub-grid (0.1 working-px) precision. Raising `WORK_XY` is costly — 3-D ridge
  filtering scales superlinearly with grid size — so prefer raising `DISPLAY_XY` for
  sharper viewing.
- Registration is in voxel/pixel space and needs no calibration; the in-plane scale it
  recovers tells you the relative pixel size of the two stacks.
- TIFFs and deformation fields are git-ignored (too large); the small landmark/transform
  files in `example/` are committed.
