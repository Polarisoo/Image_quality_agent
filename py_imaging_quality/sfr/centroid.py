"""Centroid of a vector, equivalent to MATLAB centroid.m."""
import numpy as np


def centroid(x):
    """Return centroid location of a vector.

    Args:
        x: vector

    Returns:
        loc: centroid in units of array index (0 if sum is near zero)
    """
    x = np.asarray(x, dtype=float)
    if np.any(np.isnan(x)):
        return 0
    sumx = np.sum(x)
    if sumx < 1e-4:
        loc = 0
    else:
        n = np.arange(1, len(x) + 1)
        loc = np.sum(n * x) / sumx
    return loc
