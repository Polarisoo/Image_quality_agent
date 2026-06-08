"""Project data along edge direction, equivalent to MATLAB project.m."""
import numpy as np
from .ahamming import ahamming


def project(bb, loc, slope, fac=4):
    """Project data in bb along the direction defined by slope.

    Data is accumulated in bins of width (1/fac) pixel.

    Args:
        bb: input data array (nlin, npix)
        loc: centroid location per line
        slope: edge slope from least-square fit (x = int + slope*y)
        fac: oversampling (binning) factor, default 4

    Returns:
        point: (nn,) output super-sampled 1D vector
        status: 0 = OK, 1 = zero counts encountered
    """
    status = 0
    nlin, npix = bb.shape
    nn = npix * fac

    slope = 1.0 / slope
    offset = round(fac * (0 - (nlin - 1) / slope))
    del_ = abs(offset)
    if offset > 0:
        offset = 0

    barray = np.zeros((2, nn + del_ + 100))

    for n in range(npix):
        for m in range(nlin):
            x = n
            y = m
            ling = int(np.ceil((x - y / slope) * fac) + 1 - offset)
            barray[0, ling] += 1
            barray[1, ling] += bb[m, n]

    point = np.zeros(nn)
    start = int(1 + round(0.5 * del_))

    nz = 0
    for i in range(start, start + nn):
        if barray[0, i] == 0:
            nz += 1
            status = 0
            if i == 0:
                barray[0, i] = barray[0, i + 1]
            else:
                barray[0, i] = (barray[0, i - 1] + barray[0, i + 1]) / 2

    if status != 0:
        print('                            WARNING')
        print('      Zero count(s) found during projection binning.')

    for i in range(nn):
        point[i] = barray[1, i + start] / barray[0, i + start]

    return point, status
