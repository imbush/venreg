"""Pre-computed cell masks (loaded, not segmented here) + cross-image cell matching for a
registered pair.

Segmentation is done **outside** the app (e.g. the Cellpose GUI / a GPU box, with your
own tuned parameters); here we just read the instance-label volume. Matching is purely
**geometric** — it uses cell positions after the chosen registration (and their local
spatial configuration), never appearance/intensity. That is deliberate: A and B are imaged
with different strategies (e.g. before vs after fixation), so a cell's brightness/shape/
stain differs between them; only *where the cells are* (once aligned) is comparable.
Several matchers are provided so the result can be compared.
"""
import numpy as np


# ---------------- read pre-computed instance masks ----------------

def read_masks(raw, name):
    """Parse a pre-computed instance-mask file (bytes) -> int32 label volume (z,y,x).
    Accepts a label TIFF (e.g. Cellpose ``*_cp_masks.tif``) or a Cellpose ``*_seg.npy``
    (a pickled dict with a 'masks' array)."""
    import io
    if name.lower().endswith(".npy"):
        obj = np.load(io.BytesIO(raw), allow_pickle=True)
        if obj.dtype == object:                     # _seg.npy: 0-d object array holding a dict
            d = obj.item()
            m = d.get("masks", d) if isinstance(d, dict) else d
        else:
            m = obj
    else:
        import tifffile
        m = tifffile.imread(io.BytesIO(raw))
    m = np.squeeze(np.asarray(m))
    while m.ndim > 3:                               # drop stray channel/time axes
        m = m[0]
    if m.ndim == 2:
        m = m[None]
    return m.astype(np.int32)


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


# ---------------- Soma-print matcher (TRU-FACT, Wang et al.) ----------------
# Register cells by the geometry of their neighbours (a "Soma-print": vectors from a cell
# to its m nearest neighbours), not by appearance — robust across imaging modalities. A
# multi-round, empirical-Bayes design (Wang et al.): round 1 scores candidates by greedy
# vector matching over an m>n neighbour pool; later rounds rebuild each print from the
# *confirmed*-matched neighbours; each accepted match carries a likelihood ratio.

def _neighbor_vectors(pts, k):
    """For each point, indices+vectors to its k nearest neighbours (self excluded)."""
    from scipy.spatial import cKDTree
    k = min(k, len(pts) - 1)
    tree = cKDTree(pts)
    idx = tree.query(pts, k=k + 1)[1][:, 1:]                 # drop self
    vecs = pts[idx] - pts[:, None, :]
    return idx, vecs


def _greedy_mean(U, V, n):
    """Mean of the n greedily best-matched vector-pair distances between prints U and V."""
    from scipy.spatial.distance import cdist
    D = cdist(U, V)
    m = min(n, D.shape[0], D.shape[1])
    if m == 0:
        return np.inf
    tot = 0.0
    for _ in range(m):
        f = int(D.argmin()); r, c = divmod(f, D.shape[1])
        tot += D[r, c]; D[r, :] = np.inf; D[:, c] = np.inf
    return tot / m


def _gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))


def _empirical_bayes_lr(best, second, have, iters=400):
    """Per-match likelihood ratio of incorrect vs. correct (Wang et al. empirical Bayes).
    Works on match COSTS (lower=better). The *incorrect* model is pinned from the 2nd-best
    costs (a clean estimate of mismatches); the best costs are a mixture of that incorrect
    Gaussian + a free 'correct' Gaussian (low cost). Returns LR for every entry (inf where
    unavailable); accept matches with LR < threshold."""
    b, s = best[have], second[have]
    mu_i, sig_i = float(np.mean(s)), float(np.std(s) + 1e-6)      # incorrect (from 2nd-best)
    mu_c, sig_c = float(np.percentile(b, 20)), float(np.std(b) / 2 + 1e-6)
    wc, wi = 0.5, 0.5
    for _ in range(iters):                                         # EM, incorrect comp fixed
        pc, pi = wc * _gauss(b, mu_c, sig_c), wi * _gauss(b, mu_i, sig_i)
        rc = pc / (pc + pi + 1e-300); nc = rc.sum() + 1e-9
        mu_c = (rc * b).sum() / nc
        sig_c = np.sqrt((rc * (b - mu_c) ** 2).sum() / nc) + 1e-6
        wc = nc / len(b); wi = 1 - wc
    num = wi * _gauss(best, mu_i, sig_i)
    den = wc * _gauss(best, mu_c, sig_c)
    lr = np.where(den > 0, num / (den + 1e-300), np.inf)
    lr[~np.isfinite(best)] = np.inf
    return lr


def soma_print_match(cA, cB, M, field=None, m_a=15, m_b=30, n=10, radius=None,
                     beta=1.0, rounds=3, lr_thresh=0.05):
    """Soma-print cell matching. cA,cB: centroids (x,y,z) in the working grid; cB is mapped
    into A's frame first. Returns (pairs [(iA,jB)], info) with per-match likelihood ratio
    and posterior probability of a correct match."""
    from scipy.spatial import cKDTree
    cA = np.asarray(cA, float); cB = np.asarray(cB, float)
    if len(cA) < 5 or len(cB) < 5:
        return [], {}
    cBa = map_points(cB, M, field)
    # local search radius (their "correct matches lie nearby" prior): a few median spacings
    nnA = cKDTree(cA).query(cA, k=2)[0][:, 1]
    if radius is None:
        radius = float(np.median(nnA) * 6)
    treeB = cKDTree(cBa)
    _, vA = _neighbor_vectors(cA, m_a)
    _, vB = _neighbor_vectors(cBa, m_b)
    cand = [np.array(treeB.query_ball_point(cA[i], radius), dtype=int) for i in range(len(cA))]

    matchB = np.full(len(cA), -1, int); lr_of = np.full(len(cA), np.nan); prev_n = -1
    for rnd in range(rounds):
        best_j = np.full(len(cA), -1, int); best_c = np.full(len(cA), np.inf)
        second_c = np.full(len(cA), np.inf)
        if rnd == 0:
            score = lambda i, j: _greedy_mean(vA[i], vB[j], n) + beta * np.linalg.norm(cA[i] - cBa[j])
        else:                                            # rebuild prints from confirmed matches
            conf = np.where(matchB >= 0)[0]
            if len(conf) < 4:
                break
            treeC = cKDTree(cA[conf])
            def score(i, j, _tc=treeC, _conf=conf):
                kk = min(n, len(_conf) - 1)
                nb = _conf[_tc.query(cA[i], k=kk + 1)[1][1:]]      # nearest confirmed A-neighbours
                U = cA[nb] - cA[i]; W = cBa[matchB[nb]] - cBa[j]
                return float(np.linalg.norm(U - W, axis=1).mean()) + beta * np.linalg.norm(cA[i] - cBa[j])
        for i in range(len(cA)):
            for j in cand[i]:
                c = score(i, j)
                if c < best_c[i]:
                    second_c[i] = best_c[i]; best_c[i] = c; best_j[i] = j
                elif c < second_c[i]:
                    second_c[i] = c
        have = np.isfinite(best_c) & np.isfinite(second_c)
        if have.sum() < 10:
            break
        lr = _empirical_bayes_lr(best_c, second_c, have)     # incorrect model pinned from 2nd-best
        # accept, resolving B-cell conflicts by lowest cost
        order = np.argsort(best_c)
        mB = np.full(len(cA), -1, int); lof = np.full(len(cA), np.nan); usedB = set()
        for i in order:
            j = best_j[i]
            if j < 0 or not np.isfinite(best_c[i]) or lr[i] >= lr_thresh or j in usedB:
                continue
            mB[i] = j; lof[i] = lr[i]; usedB.add(int(j))
        n = int((mB >= 0).sum())
        matchB, lr_of = mB, lof
        if rnd > 0 and prev_n > 0 and (n - prev_n) < 0.05 * prev_n:   # <5% new matches -> converged
            prev_n = n; break
        prev_n = n

    pairs = [(int(i), int(matchB[i])) for i in range(len(cA)) if matchB[i] >= 0]
    info = {(int(i), int(matchB[i])): {"lr": float(lr_of[i]),
            "prob_correct": float(1.0 / (1.0 + lr_of[i]))}
            for i in range(len(cA)) if matchB[i] >= 0}
    return pairs, info


def match(cA, cB, M, field=None, method="hungarian", max_dist=8.0):
    """Return list of (iA, jB) matched index pairs. cA, cB are centroids (x,y,z) in the
    working grid; cB is mapped into A's frame via the registration before matching."""
    cA = np.asarray(cA, float); cB = np.asarray(cB, float)
    if cA.shape[0] == 0 or cB.shape[0] == 0:
        return []
    if method == "soma_print":
        return soma_print_match(cA, cB, M, field)[0]
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
    """Label volume + {id: rgb} -> RGB float volume (z,y,x,3) in [0,1]. Uses a colour LUT
    indexed by label id (O(voxels), not O(cells x voxels)) so it stays fast for many cells."""
    maxid = int(labels.max()) if labels.size else 0
    lut = np.zeros((maxid + 1, 3), np.float32)
    for cid, rgb in color_of.items():
        if 0 <= cid <= maxid:
            lut[cid] = rgb
    return lut[labels]
