"""Find frequency for specified SFR value, equivalent to MATLAB findfreq.m."""
import numpy as np


def findfreq(dat, val, imax, fflag=0):
    """Find frequency for specified SFR value.

    Args:
        dat: SFR data n x m array. First column is frequency.
        val: threshold SFR value, e.g. 0.1
        imax: index of half-sampling frequency
        fflag: 1 = filter [1 1 1] SFR data, 0 = no filter

    Returns:
        freqval: frequencies corresponding to val, (nc,)
        sfrval: SFR values corresponding to val, (nc,)
    """
    dat = np.asarray(dat)
    n, m = dat.shape
    nc = m - 1
    freqval = np.zeros(nc)
    sfrval = np.zeros(nc)
    maxf = dat[imax, 0]

    for c in range(nc):
        sfr_col = dat[:, c + 1].copy()
        if fflag != 0:
            fil = np.array([1, 1, 1]) / 3.0
            temp = np.convolve(sfr_col, fil, mode='same')
            sfr_col[1:-1] = temp[1:-1]

        test = sfr_col - val
        x = np.where(test < 0)[0]
        if len(x) == 0 or x[0] == 0:
            s = maxf
            sval = dat[imax, c + 1]
        else:
            x = x[0]
            sval = dat[x, c + 1]
            s = dat[x, 0]
            y = dat[x, c + 1]
            y2 = dat[x + 1, c + 1]
            slope = (y2 - y) / dat[1, 0]
            dely = test[x]
            s = s - dely / slope
            sval = sval - dely

        if s > maxf:
            s = maxf
            sval = dat[imax, c + 1]

        freqval[c] = s
        sfrval[c] = sval

    return freqval, sfrval
