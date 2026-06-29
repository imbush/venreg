"""Cell segmentation (Cellpose, 3-D) + cross-image cell matching for a registered pair.

Matching is purely **geometric** — it uses cell positions after the chosen registration
(and their local spatial configuration), never appearance/intensity. That is deliberate:
A and B are imaged with different strategies (e.g. before vs after fixation), so a cell's
brightness/shape/stain differs between them; only *where the cells are* (once aligned) is
comparable. Several matchers are provided so the result can be compared.
"""
import numpy as np


# ---------------- segmentation ----------------

def segment(vol01, diameter=None, mode="2d", gpu=True):
    """Cellpose instance segmentation of a [0,1] volume (z,y,x) -> int32 3-D labels.

    mode='2d'  : segment each z-slice in 2-D and stitch labels across z (fast — the
                 practical default; full do_3D with cellpose-SAM is extremely slow on
                 CPU/MPS). mode='3d' : true volumetric do_3D (much slower).
    gpu=True uses Apple MPS / CUDA when available (Cellpose falls back to CPU otherwise).
    Handles both Cellpose v2/v3 (``Cellpose`` + cyto3) and v4 / cellpose-SAM."""
    from cellpose import models
    img = (np.clip(vol01, 0, 1) * 255).astype(np.uint8)
    if hasattr(models, "Cellpose"):                 # v2 / v3
        model = models.Cellpose(gpu=gpu, model_type="cyto3")
        kw = dict(channels=[0, 0], diameter=diameter or 0)
    else:                                           # v4 (cellpose-SAM)
        model = models.CellposeModel(gpu=gpu)
        kw = dict(diameter=diameter or None)
    if mode == "3d":
        masks = model.eval(img, do_3D=True, z_axis=0, **kw)[0]
    else:
        masks = model.eval(img, do_3D=False, z_axis=0, stitch_threshold=0.5, **kw)[0]
    return np.asarray(masks, dtype=np.int32)


def centroids(labels):
    """Label volume -> (ids[int], centroids xyz [N,3] voxel coords, sizes[int])."""
    from scipy import ndimage as ndi
    ids = np.unique(labels); ids = ids[ids > 0]
    if ids.size == 0:
        return ids.astype(int), np.zeros((0, 3)), np.zeros(0, int)
    ones = np.ones_like(labels, dtype=np.uint8)
    coms = np.asarray(ndi.center_of_mass(ones, labels, ids))      # (z,y,x)
    sizes = np.asarray(ndi.sum(ones, labels, ids)).astype(int)
    return ids.astype(int), coms[:, ::-1], sizes                  # -> x,y,z


# ---------------- map B-frame points into A's frame ----------------

def map_points(pts_xyz, M, field=None):
    """Map points from B's frame to A's frame using the stored registration (B->A):
    pA = inv(M) @ (pB + d(pB)) — the same transform the overlay 'unwarp' uses."""
    q = np.asarray(pts_xyz, float)
    if q.shape[0] == 0:
        return q
    if field is not None:
        import SimpleITK as sitk
        disp = sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64))
        q = np.array([disp.TransformPoint((float(x), float(y), float(z))) for x, y, z in q])
    Minv = np.linalg.inv(np.asarray(M, float))
    return (np.column_stack([q, np.ones(len(q))]) @ Minv.T)[:, :3]


# ---------------- matchers (all geometric / modality-robust) ----------------

def _mnn(D, max_dist):
    """Mutual nearest neighbours within max_dist."""
    if D.size == 0:
        return []
    nnB, nnA = D.argmin(1), D.argmin(0)
    return [(i, int(nnB[i])) for i in range(D.shape[0])
            if nnA[nnB[i]] == i and D[i, nnB[i]] <= max_dist]


def _hungarian(D, max_dist):
    """Globally-optimal one-to-one assignment, then drop pairs beyond max_dist."""
    from scipy.optimize import linear_sum_assignment
    if D.size == 0:
        return []
    ri, ci = linear_sum_assignment(D)
    return [(int(i), int(j)) for i, j in zip(ri, ci) if D[i, j] <= max_dist]


def _consistent(cA, cBa, D, max_dist, k=8, tol=None):
    """MNN candidates kept only where the A->B displacement agrees with the local
    consensus (median of k nearest matched neighbours). Rejects ambiguous matches in
    dense regions and residual-warp outliers — robust when the two modalities differ."""
    cand = _mnn(D, max_dist * 2)                  # generous candidates first
    if len(cand) < 3:
        return _mnn(D, max_dist)
    ai = np.array([a for a, _ in cand]); bj = np.array([b for _, b in cand])
    disp = cBa[bj] - cA[ai]                        # per-candidate displacement (A frame)
    pos = cA[ai]
    from scipy.spatial import cKDTree
    tree = cKDTree(pos)
    tol = tol if tol is not None else max_dist
    keep = []
    for n in range(len(cand)):
        idx = tree.query(pos[n], k=min(k + 1, len(cand)))[1]
        med = np.median(disp[idx], axis=0)
        if np.linalg.norm(disp[n] - med) <= tol:
            keep.append(cand[n])
    return keep


def match(cA, cB, M, field=None, method="hungarian", max_dist=8.0):
    """Return list of (iA, jB) matched index pairs. cA, cB are centroids (x,y,z) in the
    working grid; cB is mapped into A's frame via the registration before matching."""
    cA = np.asarray(cA, float); cB = np.asarray(cB, float)
    if cA.shape[0] == 0 or cB.shape[0] == 0:
        return []
    cBa = map_points(cB, M, field)
    from scipy.spatial.distance import cdist
    D = cdist(cA, cBa)
    if method == "mnn":
        return _mnn(D, max_dist)
    if method == "consistent":
        return _consistent(cA, cBa, D, max_dist)
    return _hungarian(D, max_dist)               # default


# ---------------- colouring for the visualiser ----------------

def _palette(n):
    """n visually-distinct RGB colours (golden-ratio hue walk)."""
    import colorsys
    return np.array([colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0, 0.85, 1.0)
                     for i in range(max(n, 1))])


def assign_colors(idsA, idsB, pairs):
    """Matched A/B cells share a colour; unmatched cells are dim grey. Returns
    {idA: rgb}, {idB: rgb} dicts (rgb in [0,1])."""
    grey = np.array([0.32, 0.32, 0.32])
    pal = _palette(len(pairs))
    cA = {int(i): grey for i in idsA}
    cB = {int(i): grey for i in idsB}
    for n, (ia, jb) in enumerate(pairs):
        cA[int(idsA[ia])] = pal[n]; cB[int(idsB[jb])] = pal[n]
    return cA, cB


def colorize(labels, color_of):
    """Label volume + {id: rgb} -> RGB float volume (z,y,x,3) in [0,1]."""
    out = np.zeros(labels.shape + (3,), np.float32)
    for cid, rgb in color_of.items():
        out[labels == cid] = rgb
    return out
