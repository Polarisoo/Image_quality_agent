"""Clipping detection, equivalent to MATLAB clipping.m."""
import numpy as np


def clipping(a, low, high, thresh1):
    """Check for clipping of data array.

    Args:
        a: array (nlin, npix, ncol)
        low: low clip value
        high: high clip value
        thresh1: threshold fraction for warning

    Returns:
        nlow: fraction of low-clipped pixels per channel
        nhigh: fraction of high-clipped pixels per channel
        status: 1 = OK, 0 = clipping detected
    """
    a = np.asarray(a, dtype=float)
    status = 1
    nlin, npix = a.shape[0], a.shape[1]
    ncol = a.shape[2] if a.ndim == 3 else 1
    n = nlin * npix

    nhigh = np.zeros(ncol)
    nlow = np.zeros(ncol)

    for k in range(ncol):
        if a.ndim == 3:
            ch = a[:, :, k]
        else:
            ch = a
        nlow[k] = np.sum(ch <= low)
        nhigh[k] = np.sum(ch >= high)

    nhigh = nhigh / n

    for k in range(ncol):
        if nlow[k] > thresh1:
            print(f' *** Warning: low clipping in record {k + 1}')
            status = 0
        if nhigh[k] > thresh1:
            print(f' *** Warning: high clipping in record {k + 1}')
            status = 0

    nlow = nlow / n
    return nlow, nhigh, status
