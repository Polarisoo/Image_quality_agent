"""FIR derivative correction for MTF, equivalent to MATLAB fir2fix.m."""
import numpy as np


def fir2fix(n, m):
    """Correction for MTF of derivative (difference) filter.

    Args:
        n: frequency data length [0 to half-sampling (Nyquist) frequency]
        m: length of difference filter (e.g. 2 for 2-point, 3 for 3-point)

    Returns:
        correct: (n,) MTF correction array (limited to max of 10)
    """
    correct = np.ones(n)
    m = m - 1
    scale = 1
    for i in range(1, n):
        correct[i] = abs((np.pi * i * m / (2 * (n + 1))) /
                         np.sin(np.pi * i * m / (2 * (n + 1))))
        correct[i] = 1 + scale * (correct[i] - 1)
        if correct[i] > 10:
            correct[i] = 10
    return correct
