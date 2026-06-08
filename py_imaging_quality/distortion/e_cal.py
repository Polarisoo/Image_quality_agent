"""Error calculation for distortion fit, equivalent to MATLAB e_cal.m."""
import numpy as np


def e_cal(k1, xd, yd, xm, ym):
    """Calculate the straightness error for a set of points.

    After applying distortion correction, the distorted points should lie
    on a straight line. This function measures the deviation.

    Args:
        k1: distortion coefficient (ru = rd + k1*rd^3)
        xd, yd: distorted coordinates
        xm, ym: image dimensions

    Returns:
        e: sum of squared deviations from best-fit straight line
    """
    xd = np.asarray(xd, dtype=float)
    yd = np.asarray(yd, dtype=float)

    scale = 1 + 4 * k1 * (xd ** 2 + yd ** 2) / (xm ** 2 + ym ** 2)
    xu = scale * xd
    yu = scale * yd

    # Least squares fit of a straight line
    n = len(xu)
    if n < 2:
        return 0.0

    k = (n * np.sum(xu * yu) - np.sum(xu) * np.sum(yu)) / \
        (n * np.sum(xu ** 2) - (np.sum(xu)) ** 2)
    b = np.mean(yu) - k * np.mean(xu)

    d = yu - (k * xu + b)
    e = np.sum(d ** 2)
    e = e / (k * k + 1)
    return e
