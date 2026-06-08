"""Computes first derivative via FIR filter, equivalent to MATLAB deriv1.m."""
import numpy as np


def deriv1(a, nlin, npix, fil):
    """Compute first derivative via FIR (1xn) filter.

    Edge effects are suppressed and vector size is preserved.
    Filter is applied in the npix direction only.

    Args:
        a: (nlin, npix) data array
        fil: array of filter coefficients, e.g. [-0.5, 0.5]

    Returns:
        b: (nlin, npix) output array
    """
    a = np.asarray(a)
    b = np.zeros((nlin, npix))
    nn = len(fil)
    for i in range(nlin):
        temp = np.convolve(fil, a[i, :])
        b[i, nn:npix] = temp[nn:npix]
        b[i, nn - 1] = b[i, nn]
    return b
