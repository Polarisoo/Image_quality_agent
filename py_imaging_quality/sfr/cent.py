"""Shift 1D array so center point moves to midpoint, equivalent to MATLAB cent.m."""
import numpy as np


def cent(a, center):
    """Shift array so a[center] is located at b[round((n+1)/2)].

    Written to shift a line-spread function array prior to
    applying a smoothing window.

    Uses linear (non-wrapping) shift — equivalent to MATLAB cent.m:
      if del > 0:   b(1:n-del) = a(del+1:n)         [shift right, drop rightmost del elements]
      if del < 1:   b(-del+1:n) = a(1:n+del)        [shift left, drop leftmost |del| elements]
    Data that falls outside array bounds is discarded (NOT wrapped).

    Args:
        a: input array
        center: location of signal center to be shifted

    Returns:
        b: output shifted array
    """
    a = np.asarray(a, dtype=float)
    n = len(a)
    mid = round((n + 1) / 2)
    del_ = round(center - mid)

    if abs(del_) < 1:
        return a.copy()

    b = np.zeros(n)
    if del_ > 0:
        # Shift right by del_: b[0:n-del] = a[del_:n]
        b[:n - del_] = a[del_:]
    elif del_ < 1:
        # Shift left by |del_|: b[-del_:n] = a[:n+del_]
        # -del_ is positive, n+del_ = n-|del_|
        b[-del_:] = a[:n + del_]

    return b
