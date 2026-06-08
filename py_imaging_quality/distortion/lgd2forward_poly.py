"""LGD to forward polynomial conversion, equivalent to MATLAB lgd2forward_poly.m."""
import numpy as np


def lgd2forward_poly(lgdCoeffs):
    """Convert Local Geometric Distortion polynomial to forward distortion polynomial.

    LGD(r_d) = 100*(r_d - r_u)/r_u = P(r_d)
    Output: r_u = f_inv(r_d) as 5th order polynomial.

    Args:
        lgdCoeffs: LGD polynomial coefficients

    Returns:
        forwardCoeffs: 5th order polynomial coefficients
    """
    lgdCoeffs = np.asarray(lgdCoeffs)
    rDistorted = np.linspace(0, 1, 100)
    rUndistorted = rDistorted / (np.polyval(lgdCoeffs, rDistorted) / 100 + 1)
    forwardCoeffs = np.polyfit(rUndistorted, rDistorted, 5)
    return forwardCoeffs
