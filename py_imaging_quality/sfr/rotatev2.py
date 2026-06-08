"""Rotate edge array vertical, equivalent to MATLAB rotatev2.m."""
import numpy as np
from .rotate90 import rotate90


def rotatev2(a):
    """Rotate array so edge feature is in vertical orientation.

    Test based on array values, not dimensions.

    Args:
        a: input array (nlin, npix) or (nlin, npix, ncol)

    Returns:
        a: rotated array
        nlin, npix: dimensions after rotation
        rflag: 0 = no rotation, 1 = rotation performed
    """
    a = np.asarray(a, dtype=float)
    nlin, npix = a.shape[0], a.shape[1]
    dim = a.shape

    has_rgb = len(dim) == 3 and dim[2] >= 3
    mm = 1 if has_rgb else 0

    nn = 3
    if has_rgb:
        testv = abs(np.mean(a[-nn:, :, mm]) - np.mean(a[:nn, :, mm]))
        testh = abs(np.mean(a[:, -nn:, mm]) - np.mean(a[:, :nn, mm]))
    elif len(dim) == 3:
        testv = abs(np.mean(a[-nn:, :, 0]) - np.mean(a[:nn, :, 0]))
        testh = abs(np.mean(a[:, -nn:, 0]) - np.mean(a[:, :nn, 0]))
    else:
        testv = abs(np.mean(a[-nn:, :]) - np.mean(a[:nn, :]))
        testh = abs(np.mean(a[:, -nn:]) - np.mean(a[:, :nn]))

    rflag = 0
    if testv > testh:
        rflag = 1
        a = rotate90(a)
        nlin, npix = npix, nlin

    return a, nlin, npix, rflag
