"""Find line center positions, equivalent to MATLAB findcenter.m."""
import numpy as np


def findcenter(data, val, tol=2):
    """Find center positions of consecutive segments where data == val.

    Args:
        data: 1D data array
        val: target value (typically 0 or 1)
        tol: minimum segment width

    Returns:
        center: array of center positions
    """
    data = np.asarray(data)
    flag = False
    center = []
    for n in range(len(data)):
        if data[n] == val:
            if not flag:
                flag = True
                n1 = n
        else:
            if flag:
                n2 = n - 1
                loc = round((n1 + n2) / 2)
                if n2 - n1 >= tol:
                    center.append(int(loc))
                    if len(center) > 2:
                        dis1 = center[-1] - center[-2]
                        dis2 = center[-2] - center[-3]
                        if dis1 < dis2 * 0.6:
                            center.pop()
                flag = False

    return np.array(center)
