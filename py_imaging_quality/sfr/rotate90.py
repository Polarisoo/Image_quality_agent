"""90-degree counterclockwise rotation, equivalent to MATLAB rotate90.m."""
import numpy as np


def rotate90(a, n=1):
    """Rotate matrix 90 degrees counterclockwise n times.

    Args:
        a: input matrix (nlin, npix) or (nlin, npix, nc)
        n: number of 90-degree rotations (default 1)

    Returns:
        rotated matrix
    """
    a = np.asarray(a)
    nd = a.ndim
    if nd < 2:
        raise ValueError('Input to rotate90 must be a matrix')

    out = a
    for _ in range(n):
        out = _r90(out)
    return out


def _r90(a):
    """Single 90-degree CCW rotation."""
    if a.ndim == 2:
        return np.flipud(a.T)
    else:
        nlin, npix, nc = a.shape
        out = np.zeros((npix, nlin, nc), dtype=a.dtype)
        for c in range(nc):
            temp = a[:, :, c].T
            out[:, :, c] = temp[::-1, :]
        return np.squeeze(out)
