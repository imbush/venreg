"""Landmark-driven registration (works on any data; pure voxel/pixel space).

fit_rigid  : best similarity (uniform scale + rotation + translation), auto-detecting an
             in-plane REFLECTION (mirror), plus a linear z map. Returns a 4x4 voxel matrix.
fit_warp   : reflected affine (anisotropic) + linear z, then 3D diffeomorphic Demons
             deformable refinement seeded from it. Returns (matrix, displacement field).
Both consume landmarks as an (N,6) array: xA,yA,zA,xB,yB,zB in WORKING voxel coords.
"""
import numpy as np

# ---------------- 2D similarity with optional reflection ----------------

def _similarity_2d(src, dst, reflect):
    s = src.copy()
    if reflect:
        s = s.copy(); s[:, 0] = -s[:, 0]
    mu_s, mu_d = s.mean(0), dst.mean(0)
    sc, dc = s - mu_s, dst - mu_d
    saa = (sc ** 2).sum()
    sab = (sc[:, 0] * dc[:, 0] + sc[:, 1] * dc[:, 1]).sum()
    scr = (sc[:, 0] * dc[:, 1] - sc[:, 1] * dc[:, 0]).sum()
    if saa < 1e-9:
        return None
    a, b = sab / saa, scr / saa
    t = mu_d - np.array([a * mu_s[0] - b * mu_s[1], b * mu_s[0] + a * mu_s[1]])
    # model on ORIGINAL src coords (fold reflection into the matrix)
    if not reflect:
        M = np.array([[a, -b], [b, a]])
    else:
        M = np.array([[-a, -b], [-b, a]])
    pred = src @ M.T + t
    rms = float(np.sqrt(((pred - dst) ** 2).sum(1).mean()))
    return dict(M=M, t=t, scale=float(np.hypot(a, b)),
                rot_deg=float(np.degrees(np.arctan2(b, a))), reflect=reflect, rms=rms)


def fit_similarity(srcA_xy, dstB_xy):
    """Best of proper / reflected 2D similarity (>=2 points)."""
    cands = [c for c in (_similarity_2d(srcA_xy, dstB_xy, r) for r in (False, True)) if c]
    return min(cands, key=lambda c: c["rms"]) if cands else None


def fit_affine_2d(srcA_xy, dstB_xy):
    """Least-squares 2D affine (allows anisotropy + reflection; >=3 points)."""
    X = np.column_stack([srcA_xy, np.ones(len(srcA_xy))])
    sol, *_ = np.linalg.lstsq(X, dstB_xy, rcond=None)
    M = sol[:2].T; t = sol[2]
    pred = srcA_xy @ M.T + t
    rms = float(np.sqrt(((pred - dstB_xy) ** 2).sum(1).mean()))
    return dict(M=M, t=t, rms=rms, det=float(np.linalg.det(M)))


def fit_zmap(zA, zB):
    """Linear z map zB = m*zA + c (>=2 distinct zA), else identity."""
    if len(zA) >= 2 and np.ptp(zA) > 0:
        m, c = np.polyfit(zA, zB, 1)
        return float(m), float(c)
    return 1.0, 0.0


def _matrix4(M2, t2, zm, zc):
    M = np.eye(4)
    M[:2, :2] = M2; M[0, 3] = t2[0]; M[1, 3] = t2[1]
    M[2, 2] = zm; M[2, 3] = zc
    return M


def fit_rigid(landmarks):
    """landmarks: (N,6) xA,yA,zA,xB,yB,zB. Returns dict(matrix 4x4 A->B, params...)."""
    L = np.asarray(landmarks, float)
    sim = fit_similarity(L[:, :2], L[:, 3:5])
    if sim is None:
        return None
    zm, zc = fit_zmap(L[:, 2], L[:, 5])
    M = _matrix4(sim["M"], sim["t"], zm, zc)
    resid = np.linalg.norm(L[:, :2] @ sim["M"].T + sim["t"] - L[:, 3:5], axis=1)
    return dict(kind="rigid", matrix=M.tolist(), scale=sim["scale"], rotation_deg=sim["rot_deg"],
                reflect=bool(sim["reflect"]), z_scale=zm, z_offset=zc,
                rms_px=sim["rms"], residuals_px=resid.tolist(), n=len(L))


def fit_affine(landmarks):
    L = np.asarray(landmarks, float)
    aff = fit_affine_2d(L[:, :2], L[:, 3:5])
    zm, zc = fit_zmap(L[:, 2], L[:, 5])
    M = _matrix4(aff["M"], aff["t"], zm, zc)
    return dict(kind="affine", matrix=M.tolist(), det=aff["det"], reflect=bool(aff["det"] < 0),
                z_scale=zm, z_offset=zc, rms_px=aff["rms"], n=len(L))


# ---------------- volume warping ----------------

def warp_volume(volA, M4, out_shape):
    """Resample A into B's grid using 4x4 voxel matrix M (A->B). out_shape=(z,y,x)."""
    from scipy.ndimage import map_coordinates
    nz, ny, nx = out_shape
    zz, yy, xx = np.mgrid[0:nz, 0:ny, 0:nx]
    P = np.stack([xx.ravel(), yy.ravel(), zz.ravel(), np.ones(xx.size)])
    pa = np.linalg.inv(np.asarray(M4)) @ P
    coords = np.stack([pa[2], pa[1], pa[0]])  # z,y,x into A
    return map_coordinates(volA, coords, order=1, cval=0.0).reshape(out_shape).astype(np.float32)


def deformable(fixed_ves, moving_ves, iters=(40, 40, 40), shrink=(4, 2, 1), smooth=2.0):
    """3D diffeomorphic Demons. fixed/moving must share shape. Returns displacement
    field as (z,y,x,3) array (and a callable resampler via SimpleITK transform)."""
    import SimpleITK as sitk
    fix = sitk.GetImageFromArray(fixed_ves.astype(np.float32))
    mov = sitk.GetImageFromArray(moving_ves.astype(np.float32))
    demons = sitk.DiffeomorphicDemonsRegistrationFilter()
    demons.SetStandardDeviations(smooth)
    field = None
    for sh, it in zip(shrink, iters):
        f = sitk.Shrink(sitk.SmoothingRecursiveGaussian(fix, smooth), [sh] * 3)
        m = sitk.Shrink(sitk.SmoothingRecursiveGaussian(mov, smooth), [sh] * 3)
        if field is None:
            field = sitk.Image(f.GetSize(), sitk.sitkVectorFloat64, 3); field.CopyInformation(f)
        else:
            field = sitk.Resample(field, f)
        demons.SetNumberOfIterations(it)
        field = demons.Execute(f, m, field)
    field = sitk.Resample(field, fix)
    return field  # SimpleITK displacement-field image


def apply_warp(volA, M4, out_shape, field=None):
    """Warp A by rigid/affine M, then optional Demons displacement field (in B grid)."""
    warped = warp_volume(volA, M4, out_shape)
    if field is None:
        return warped
    import SimpleITK as sitk
    disp = sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64))
    img = sitk.GetImageFromArray(warped)
    out = sitk.Resample(img, sitk.GetImageFromArray(np.zeros(out_shape, np.float32)),
                        disp, sitk.sitkLinear, 0.0)
    return sitk.GetArrayFromImage(out).astype(np.float32)


# ---------------- volumetric evaluation (no max-Z) ----------------

def oriented_inlier(srcM, srcOri, dstTree, dstPts, dstOri, M4, out_shape, tau=2.5, tol_deg=25):
    """Fraction of src mask-voxels that map within tau of a dst mask-voxel with matching
    in-plane orientation. Direction-specific (call A->B and B->A)."""
    from scipy.spatial import cKDTree
    M = np.asarray(M4)
    m = srcM @ M[:3, :3].T + M[:3, 3]
    nz, ny, nx = out_shape
    inb = ((m[:, 0] >= 0) & (m[:, 0] < nx) & (m[:, 1] >= 0) & (m[:, 1] < ny) &
           (m[:, 2] >= 0) & (m[:, 2] < nz))
    if inb.sum() < 50:
        return dict(frac=0.0, n_inl=0, n_in=int(inb.sum()))
    d, idx = dstTree.query(m[inb], distance_upper_bound=tau * 2)
    near = d < tau
    rot = np.arctan2(M[1, 0], M[0, 0])
    oa = np.mod(srcOri[inb] + rot, np.pi)
    ob = dstOri[np.where(near, idx, 0)]
    dd = np.abs(oa - ob); dd = np.minimum(dd, np.pi - dd)
    inl = near & (dd < np.deg2rad(tol_deg))
    return dict(frac=float(inl.mean()), n_inl=int(inl.sum()), n_in=int(inb.sum()))
