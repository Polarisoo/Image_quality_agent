"""MTF metric calculation (MTF50, MTF30), equivalent to MATLAB mtf_eff.m."""
import numpy as np


def mtf_eff(freq, mtf, mtf_val):
    """Find frequencies where MTF equals specified values.

    Args:
        freq: (n,) frequency array
        mtf: (n, ncol) MTF values per channel
        mtf_val: list/array of MTF threshold values, e.g. [0.5, 0.3]

    Returns:
        freq_val: (num_thresholds, ncol) frequency values at each threshold
    """
    freq = np.asarray(freq)
    mtf = np.asarray(mtf)
    mtf_val = np.atleast_1d(mtf_val)
    num = len(mtf_val)
    ncol = mtf.shape[1]
    freq_val = np.zeros((num, ncol))

    for n in range(num):
        for m in range(ncol):
            idx = np.where(mtf[:, m] < mtf_val[n])[0]
            if len(idx) > 0:
                n1 = idx[0]
                n2 = n1 - 1 if n1 > 0 else 0
                f1 = freq[n1]
                f2 = freq[n2]
                y1 = mtf[n1, m]
                y2 = mtf[n2, m]
                # Guard against zero or very small (y2-y1) to prevent NaN
                if abs(y2 - y1) > 1e-10:
                    freq_val[n, m] = f1 + (f2 - f1) * (mtf_val[n] - y1) / (y2 - y1)
                else:
                    freq_val[n, m] = float('nan')
            else:
                # MTF never drops below threshold — mark as NaN rather than
                # returning freq[-1] which would be a misleading extrapolation
                freq_val[n, m] = float('nan')

    return freq_val
