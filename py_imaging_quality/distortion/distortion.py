"""Distortion measurement, equivalent to MATLAB distortion.m."""
import numpy as np
import sympy as sp
from scipy.optimize import least_squares
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import remove_small_objects
from .findcenter import findcenter
from .e_all import e_all
from ..getmyroi import getmyroi
from ..imageread import imageread


def distortion(img=None):
    """Measure radial lens distortion from a grid pattern image.

    Uses SMIA TV distortion metric.

    Args:
        img: input image. If None, user selects one.

    Returns:
        k1: radial distortion coefficient (ru = rd + k1*rd^3)
        TV: SMIA TV distortion value (fraction, NOT percent)
    """
    if img is None:
        status, img, ftype, pathname, f = imageread()
        if f is None:
            print('No image selected')
            return None, None

    xm, ym = img.shape[1], img.shape[0]
    n_cor = 3 if img.ndim == 3 else 1
    r2 = (xm * xm + ym * ym) / 4

    roi_data, roi = getmyroi(img, 1)
    I = roi_data[0]
    if n_cor == 3:
        I_gray = np.dot(I[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        I_gray = I.copy()

    xc = 0.5 * xm - roi[0][0]
    yc = 0.5 * ym - roi[0][1]

    th = (np.max(I_gray) + np.min(I_gray)) * 0.5
    Iedge = I_gray < th

    # Dilate and erode
    se = np.ones((3, 3), dtype=bool)
    Iedge = binary_dilation(Iedge, se)
    Iedge = binary_erosion(Iedge, se)
    Iedge = remove_small_objects(Iedge, 10)

    temp = Iedge.astype(float)

    # Vertical lines
    ver_sum = np.sum(temp, axis=0)
    blin = ver_sum > np.mean(ver_sum)
    all_ver_y = findcenter(blin, 0)
    ver_y = all_ver_y.copy()

    ver_x = None
    for n in range(len(ver_y)):
        temp1 = findcenter(Iedge[:, int(ver_y[n])], 0)
        if n > 1:
            while len(temp1) < ver_x.shape[0]:
                dup = abs(np.mean(ver_x[0, :]) - temp1[0])
                ddown = abs(np.mean(ver_x[-1, :]) - temp1[-1])
                if dup > ddown:
                    ver_x = ver_x[1:, :]
                else:
                    ver_x = ver_x[:-1, :]
            while len(temp1) > ver_x.shape[0]:
                dup = abs(np.mean(ver_x[0, :]) - temp1[0])
                ddown = abs(np.mean(ver_x[-1, :]) - temp1[-1])
                if dup > ddown:
                    temp1 = temp1[1:]
                else:
                    temp1 = temp1[:-1]

        if ver_x is None:
            ver_x = temp1.reshape(-1, 1)
        else:
            # Ensure same length
            min_len = min(len(temp1), ver_x.shape[0])
            ver_x = ver_x[:min_len, :]
            temp1 = temp1[:min_len]
            ver_x = np.column_stack([ver_x, temp1])

    # Horizontal lines
    col = np.sum(temp, axis=1)
    bcol = col > np.mean(col)
    all_hor_x = findcenter(bcol, 0)
    hor_x = all_hor_x.copy()

    hor_y_list = []
    for n in range(len(hor_x)):
        temp1 = findcenter(Iedge[int(hor_x[n]), :], 0)
        if n > 1:
            while len(temp1) < len(hor_y_list[-1]):
                dleft = abs(np.mean(np.array(hor_y_list)[0, :]) - temp1[0])
                dright = abs(np.mean(np.array(hor_y_list)[-1, :]) - temp1[-1])
                if dleft > dright:
                    hor_y_list = [h[1:] for h in hor_y_list]
                else:
                    hor_y_list = [h[:-1] for h in hor_y_list]
            while len(temp1) > len(hor_y_list[-1]):
                dleft = abs(np.mean(np.array(hor_y_list)[0, :]) - temp1[0])
                dright = abs(np.mean(np.array(hor_y_list)[-1, :]) - temp1[-1])
                if dleft > dright:
                    temp1 = temp1[1:]
                else:
                    temp1 = temp1[:-1]

        hor_y_list.append(temp1)

    if len(hor_y_list) > 0:
        min_len = min(len(h) for h in hor_y_list)
        hor_y = np.array([h[:min_len] for h in hor_y_list])
    else:
        return 0, 0

    # Center coordinates
    ver_x_centered = ver_x - xc
    ver_y_centered = ver_y - yc
    hor_x_centered = hor_x - xc
    hor_y_centered = hor_y - yc

    # Apply edge weighting
    for n in range(len(ver_y)):
        weight = 1.0
        mid = ver_x_centered.shape[0] // 2
        for m in range(mid):
            ver_x_centered[m, n] *= weight
            ver_x_centered[-(m + 1), n] *= weight
            weight *= 0.8

    for n in range(len(hor_x)):
        weight = 1.0
        mid = hor_y_centered.shape[1] // 2
        for m in range(mid):
            hor_y_centered[n, m] *= weight
            hor_y_centered[n, -(m + 1)] *= weight
            weight *= 0.8

    # Optimization
    xGuess = 0.1
    obj = lambda k1_val: _residual(k1_val, ver_x_centered, ver_y_centered,
                                   hor_x_centered, hor_y_centered, xm, ym)
    result = least_squares(obj, xGuess, method='lm')
    k1 = float(result.x[0])

    # Calculate TV distortion using numpy roots (replaced sympy for compatibility)
    # Equation 1: k1*x^3 + x = 0.5^0.5  ->  k1*x^3 + x - 0.5^0.5 = 0
    coeffs1 = [float(k1), 0, 1, -(0.5 ** 0.5)]
    r1_all = np.roots(coeffs1)
    # Pick the real root closest to 0.5^0.5 (expected ~0.707)
    y1 = None
    best_diff = 1e9
    for r in r1_all:
        if abs(np.imag(r)) < 1e-6:
            rv = float(np.real(r))
            if abs(rv) < best_diff or (y1 is None and rv > 0):
                best_diff = abs(rv)
                y1 = rv * (0.5 ** 0.5)

    # Equation 2: k1*x^3 + x = 1  ->  k1*x^3 + x - 1 = 0
    coeffs2 = [float(k1), 0, 1, -1]
    r2_all = np.roots(coeffs2)
    y2 = None
    best_diff = 1e9
    for r in r2_all:
        if abs(np.imag(r)) < 1e-6:
            rv = float(np.real(r))
            if abs(rv) < best_diff or (y2 is None and rv > 0):
                best_diff = abs(rv)
                y2 = rv

    if y1 is not None and y2 is not None and abs(y2) > 1e-9:
        TV = (y1 - y2) / y2
    else:
        TV = 0

    print(f'k1 = {k1:.6f}  (ru = rd + k1*rd^3)')
    print(f'SMIA TV distortion = {100 * TV:.2f}%')

    return k1, TV


def _distortion_from_roi(roi_img):
    """Measure distortion from a pre-selected ROI image (non-interactive).

    Args:
        roi_img: ROI image array (should contain a grid pattern)

    Returns:
        k1, TV
    """
    xm, ym = roi_img.shape[1], roi_img.shape[0]
    n_cor = 3 if roi_img.ndim == 3 else 1

    if n_cor == 3:
        I_gray = np.dot(roi_img[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        I_gray = roi_img.copy()

    th = (np.max(I_gray) + np.min(I_gray)) * 0.5
    Iedge = I_gray < th

    se = np.ones((3, 3), dtype=bool)
    Iedge = binary_dilation(Iedge, se)
    Iedge = binary_erosion(Iedge, se)
    Iedge = remove_small_objects(Iedge, 10)

    temp = Iedge.astype(float)

    # Vertical lines
    ver_sum = np.sum(temp, axis=0)
    blin = ver_sum > np.mean(ver_sum)
    all_ver_y = findcenter(blin, 0)
    ver_y = all_ver_y.copy()

    ver_x_list = []
    for n in range(len(ver_y)):
        temp1 = findcenter(Iedge[:, int(ver_y[n])], 0)
        ver_x_list.append(temp1)

    if len(ver_x_list) < 2:
        return 0, 0

    min_len = min(len(v) for v in ver_x_list)
    ver_x = np.array([v[:min_len] for v in ver_x_list]).T
    if ver_x.ndim < 2 or ver_x.shape[0] < 2:
        return 0, 0

    # Horizontal lines
    col = np.sum(temp, axis=1)
    bcol = col > np.mean(col)
    all_hor_x = findcenter(bcol, 0)
    hor_x = all_hor_x.copy()

    hor_y_list = []
    for n in range(len(hor_x)):
        temp1 = findcenter(Iedge[int(hor_x[n]), :], 0)
        hor_y_list.append(temp1)

    if len(hor_y_list) < 2:
        return 0, 0

    min_len = min(len(h) for h in hor_y_list)
    hor_y = np.array([h[:min_len] for h in hor_y_list])
    if hor_y.ndim < 2 or hor_y.shape[1] < 2:
        return 0, 0

    # Center
    xc = 0.5 * xm
    yc = 0.5 * ym
    ver_x_c = ver_x - xc
    ver_y_c = ver_y - yc
    hor_x_c = hor_x - xc
    hor_y_c = hor_y - yc

    # Edge weighting
    for n in range(len(ver_y)):
        weight = 1.0
        mid = ver_x_c.shape[0] // 2
        for m in range(mid):
            ver_x_c[m, n] *= weight
            if ver_x_c.shape[0] - m - 1 >= 0:
                ver_x_c[-m - 1, n] *= weight
            weight *= 0.8

    for n in range(len(hor_x)):
        weight = 1.0
        mid = hor_y_c.shape[1] // 2
        for m in range(mid):
            hor_y_c[n, m] *= weight
            if hor_y_c.shape[1] - m - 1 >= 0:
                hor_y_c[n, -m - 1] *= weight
            weight *= 0.8

    from scipy.optimize import least_squares
    from .e_all import e_all

    obj = lambda k1_val: np.array([e_all(float(k1_val[0]), ver_x_c, ver_y_c,
                                        hor_x_c, hor_y_c, xm, ym)])
    result = least_squares(obj, [0.1], method='lm')
    k1 = float(result.x[0])

    # TV calculation using numpy roots (replaced sympy for compatibility)
    # Eq1: k1*x^3 + x = 0.5^0.5
    coeffs1 = [float(k1), 0, 1, -(0.5 ** 0.5)]
    r1_all = np.roots(coeffs1)
    y1 = None
    for r in r1_all:
        if abs(np.imag(r)) < 1e-6:
            rv = float(np.real(r))
            if y1 is None or (rv > 0 and abs(rv) < abs(y1)):
                y1 = rv * (0.5 ** 0.5)

    # Eq2: k1*x^3 + x = 1
    coeffs2 = [float(k1), 0, 1, -1]
    r2_all = np.roots(coeffs2)
    y2 = None
    for r in r2_all:
        if abs(np.imag(r)) < 1e-6:
            rv = float(np.real(r))
            if y2 is None or (rv > 0 and abs(rv) < abs(y2)):
                y2 = rv

    if y1 is not None and y2 is not None and abs(y2) > 1e-9:
        TV = (y1 - y2) / y2
    else:
        TV = 0.0

    print(f'k1 = {k1:.6f}')
    print(f'SMIA TV distortion = {100 * TV:.2f}%')
    return k1, TV


def _residual(k1, ver_x, ver_y, hor_x, hor_y, xm, ym):
    """Residual function for least_squares optimization."""
    k1_val = float(k1[0]) if hasattr(k1, '__len__') else float(k1)
    e = e_all(k1_val, ver_x, ver_y, hor_x, hor_y, xm, ym)
    return np.array([e])
