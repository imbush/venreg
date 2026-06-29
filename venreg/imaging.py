"""TIFF loading, downsampling to a working grid, and structural representations
(fluorescence + vesselness) + in-plane orientation. All in voxel space."""
import numpy as np
import tifffile
from scipy.ndimage import zoom
from skimage.filters import sato, threshold_otsu
from skimage.feature import structure_tensor

WORK_XY = 256  # working in-plane grid for the (expensive) vesselness + registration fit.
               # Display slices are rendered separately at a higher resolution (see
               # DISPLAY_XY in server.py) so large mosaics stay sharp without paying the
               # superlinear cost of 3-D ridge filtering on a big grid.


def tiff_info(path):
    """Metadata only (no pixel data): axes, full shape, channel count + labels.
    Lets the UI offer a channel picker before committing to a (possibly huge) load."""
    with tifffile.TiffFile(path) as tf:
        s = tf.series[0]
        axes = s.axes                       # e.g. 'ZCYX'
        shape = [int(x) for x in s.shape]
        ci = axes.find("C")
        nc = shape[ci] if ci >= 0 else 1
        labels = None
        ij = tf.imagej_metadata or {}
        lbls = ij.get("Labels")
        if lbls and nc > 1:                 # ImageJ Labels are per-(z,c); first nc are channel names
            labels = [str(x) for x in lbls[:nc]]
        return dict(axes=axes, shape=shape, n_channels=int(nc), channel_labels=labels)


def load_volume(path, channel=0, work_xy=WORK_XY):
    """Load one channel of a TIFF z-stack -> (vol float32 [0,1], info).

    Downsamples XY to work_xy (keeps z), reading and downscaling **one z-plane at a
    time via a lazy zarr view** so memory stays small even for multi-GB mosaics
    (we never materialise the full-res volume or a float32 copy of it)."""
    import zarr
    with tifffile.TiffFile(path) as tf:
        s = tf.series[0]
        axes = s.axes
        arr = zarr.open(s.aszarr(), mode="r")
        ax = {c: i for i, c in enumerate(axes)}
        zi, ci, yi, xi = ax.get("Z"), ax.get("C"), ax["Y"], ax["X"]
        nz = arr.shape[zi] if zi is not None else 1
        ny, nx = arr.shape[yi], arr.shape[xi]
        if not (0 <= channel < (arr.shape[ci] if ci is not None else 1)):
            channel = 0
        fy, fx = work_xy / ny, work_xy / nx
        ds = np.empty((nz, work_xy, work_xy), np.float32)
        for z in range(nz):
            sel = [0] * arr.ndim            # pin every axis; 0 for any stray (T/S/…) axis
            if zi is not None:
                sel[zi] = z
            if ci is not None:
                sel[ci] = channel
            sel[yi] = slice(None)
            sel[xi] = slice(None)
            plane = np.asarray(arr[tuple(sel)], dtype=np.float32)
            if yi > xi:                     # normalise to (Y, X)
                plane = plane.T
            ds[z] = zoom(plane, (fy, fx), order=1)
    lo, hi = np.percentile(ds, [1, 99.7])
    ds = np.clip((ds - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)
    info = dict(orig_shape=[int(nz), int(ny), int(nx)], work_shape=list(ds.shape),
                xy_downsample=float(nx / work_xy), channel=int(channel), axes=axes)
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
    """Return the registration/structural representations: the raw fluorescence and the
    vesselness ridge map. Vesselness drives the registration backend (orientation fit,
    B->A inlier eval, deformable demons); it is not shown directly — the viewer offers the
    fluorescence and a gaussian-blurred version (derived at display time)."""
    return dict(fluorescence=vol01, vesselness=vesselness(vol01))
