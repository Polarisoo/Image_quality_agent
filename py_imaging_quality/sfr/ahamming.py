"""Asymmetric Hamming window, equivalent to MATLAB ahamming.m."""
import numpy as np


def ahamming(n, mid):
    """Generate a general asymmetric Hamming-type window array.

    If mid = (n+1)/2 then the usual symmetric Hamming window is returned.

    Args:
        n: length of array
        mid: midpoint (maximum) of window function

    Returns:
        data: window array (n,)
    """
    data = np.zeros(n)
    wid1 = mid - 1
    wid2 = n - mid
    wid = max(wid1, wid2)
    for i in range(n):
        arg = i - mid
        data[i] = np.cos(np.pi * arg / wid)
    data = 0.54 + 0.46 * data
    return data
