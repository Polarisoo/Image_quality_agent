"""Global error for distortion optimization, equivalent to MATLAB e_all.m."""
import numpy as np
from .e_cal import e_cal


def e_all(k1, ver_x, ver_y, hor_x, hor_y, xm, ym):
    """Calculate total straightness error for all grid lines.

    Args:
        k1: distortion coefficient
        ver_x: vertical line x-coordinates (n_points, n_lines)
        ver_y: vertical line y-coordinates
        hor_x: horizontal line x-coordinates (n_lines, n_points)
        hor_y: horizontal line y-coordinates
        xm, ym: image dimensions

    Returns:
        e_global: total error
    """
    e_global = 0.0

    # Vertical lines
    yd = ver_y
    nlin = ver_x.shape[0]
    for num in range(nlin):
        xd = ver_x[num, :]
        e_global += e_cal(k1, xd, yd, xm, ym)

    # Horizontal lines
    ncol = hor_y.shape[1]
    xd = hor_x
    for num in range(ncol):
        yd = hor_y[:, num]
        e_global += e_cal(k1, xd, yd, xm, ym)

    return e_global
