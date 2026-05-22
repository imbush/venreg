"""TIFF loading, downsampling to a working grid, and structural representations
(fluorescence / vesselness / mask) + in-plane orientation. All in voxel space."""
import numpy as np
import tifffile
from scipy.ndimage import zoom
from skimage.filters import sato, threshold_otsu
from skimage.feature import structure_tensor

WORK_XY = 256  # working in-plane size (display + registration)


def load_volume(path, work_xy=WORK_XY):
    """Load a TIFF z-stack -> (vol float32 [0,1], info). Downsamples XY to work_xy,
    keeps z. Squeezes to 3D (takes first channel if 4D)."""
    vol = tifffile.imread(path)
    vol = np.asarray(vol)
    while vol.ndim > 3:                 # drop channel/time axes (keep largest-Z interpretation)
        vol = vol[0]
    if vol.ndim == 2:
        vol = vol[None]
    nz, ny, nx = vol.shape
    fy, fx = work_xy / ny, work_xy / nx
    ds = zoom(vol.astype(np.float32), (1.0, fy, fx), order=1)
    lo, hi = np.percentile(ds, [1, 99.7])
    ds = np.clip((ds - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)
    info = dict(orig_shape=[int(nz), int(ny), int(nx)], work_shape=list(ds.shape),
                xy_downsample=float(nx / work_xy))
    return ds, info


def vesselness(vol01, sigmas=(1.5, 2.5, 3.5)):
    v = sato(vol01, sigmas=list(sigmas), black_ridges=False)
    return (v / (v.max() + 1e-9)).astype(np.float32)


def mask_of(ves, thr_scale=0.5, min_size=64):
    from skimage.morphology import remove_small_objects
    m = ves > threshold_otsu(ves) * thr_scale
    return remove_small_objects(m, min_size=min_size)


def inplane_orientation(ves):
    """Per-slice 2D ridge orientation (radians mod pi). PSF-robust (ignores z)."""
    th = np.zeros_like(ves)
    for z in range(ves.shape[0]):
        Axx, Axy, Ayy = structure_tensor(ves[z], sigma=2, order='rc')
        th[z] = np.mod(0.5 * np.arctan2(2 * Axy, Axx - Ayy) + np.pi / 2, np.pi)
    return th


def representations(vol01):
    """Return dict of the three display/registration representations."""
    ves = vesselness(vol01)
    msk = mask_of(ves).astype(np.float32)
    inter = (vol01 * ves) * (msk > 0)
    if inter.max() > 0:
        inter = inter / inter.max()
    return dict(fluorescence=vol01, vesselness=ves, mask=msk, intersection=inter.astype(np.float32))
