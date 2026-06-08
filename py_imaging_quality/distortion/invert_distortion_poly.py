"""Polynomial inversion for distortion, equivalent to MATLAB invert_distortion_poly.m."""
import numpy as np


def invert_distortion_poly(inverseCoeffs):
    """Invert a radial distortion polynomial.

    Given P where r_u = P(r_d), find Q where r_d = Q(r_u).

    Args:
        inverseCoeffs: polynomial coefficients [a3, a2, a1, a0]

    Returns:
        forwardCoeffs: inverted polynomial (5th order)
    """
    inverseCoeffs = np.asarray(inverseCoeffs)
    rDistorted = np.linspace(0, 1, 10)
    rUndistorted = np.polyval(inverseCoeffs, rDistorted)
    forwardCoeffs = np.polyfit(rUndistorted, rDistorted, 5)
    return forwardCoeffs
