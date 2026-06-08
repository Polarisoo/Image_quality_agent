"""Find edge via linear fit, equivalent to MATLAB findedge.m."""
import numpy as np


def findedge(cent_data, nlin):
    """Fit linear equation to edge location array.

    Args:
        cent_data: array of centroid values
        nlin: length of cent

    Returns:
        slope, int: from least-square fit
        Note: this fits index = int + slope * cent(x), inverse of usual form
    """
    index = np.arange(nlin)
    slope, intercept = np.polyfit(index, cent_data, 1)
    return slope, intercept
