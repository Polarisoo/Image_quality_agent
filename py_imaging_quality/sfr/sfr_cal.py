"""SFR/MTF calculation, equivalent to MATLAB sfr_cal.m."""
import numpy as np
from .ahamming import ahamming
from .centroid import centroid
from .cent import cent
from .deriv1 import deriv1
from .findedge import findedge
from .fir2fix import fir2fix
from .project import project
from .rotatev2 import rotatev2
from .mtf_eff import mtf_eff
from .inbox1 import inbox1
from ..imageread import imageread
from ..getmyroi import getmyroi


def sfr_cal(atemp=None):
    """Main SFR (Spatial Frequency Response) / MTF calculation.

    Args:
        atemp: input image array. If None, user selects an image.

    Returns:
        rst_freq: list of frequency arrays per ROI
        rst_mtf: list of MTF arrays per ROI
        rst_esf: list of ESF arrays per ROI
    """
    if atemp is None:
        status, atemp, ftype, pathname, f = imageread()
        if f is None or f == 0:
            print('No image selected')
            return None, None, None

    sfr_num, GAMMA, weight = inbox1()

    nbin = 4
    del_ = 1

    if atemp.dtype == np.uint16:
        smax = 2 ** 16 - 1
    elif atemp.dtype == np.uint8:
        smax = 255
    else:
        smax = 1e10

    roi_temp, roi = getmyroi(atemp, sfr_num)

    if roi is None or len(roi) == 0:
        print('No ROI selected')
        return None, None, None

    rst_freq = []
    rst_mtf = []
    rst_esf = []

    for n_sfr in range(sfr_num):
        print(f'ROI: {roi[3] - roi[1]} X {roi[2] - roi[0]}')
        a = roi_temp[n_sfr].astype(float)

        # Gamma correction
        a = a / smax
        from skimage.exposure import adjust_gamma
        a = adjust_gamma(a, gamma=GAMMA)
        a = a * smax

        nlin, npix = a.shape[0], a.shape[1]
        ncol = a.shape[2] if a.ndim == 3 else 1

        if ncol == 3:
            lum = weight[0] * a[:, :, 0] + weight[1] * a[:, :, 1] + weight[2] * a[:, :, 2]
            cc = np.zeros((nlin, npix, 4))
            cc[:, :, 0] = a[:, :, 0]
            cc[:, :, 1] = a[:, :, 1]
            cc[:, :, 2] = a[:, :, 2]
            cc[:, :, 3] = lum
            a = cc
            ncol = 4
        elif ncol == 1:
            a = a[:, :, np.newaxis]
            ncol = 1

        # Rotate edge to vertical
        a, nlin, npix, _ = rotatev2(a)

        loc = np.zeros((ncol, nlin))
        fil1 = np.array([0.5, -0.5])
        fil2 = np.array([0.5, 0, -0.5])

        # Determine edge direction
        tleft = np.sum(a[:, :5, 0])
        tright = np.sum(a[:, -5:, 0])
        if tleft > tright:
            fil1 = np.array([-0.5, 0.5])
            fil2 = np.array([-0.5, 0, 0.5])

        test = abs((tleft - tright) / (tleft + tright))
        if test < 0.2:
            print(' ** WARNING: Edge contrast is less than 20%')

        win1 = ahamming(npix, (npix + 1) / 2)

        fitme = np.zeros((ncol, 3))
        for color in range(ncol):
            c = deriv1(a[:, :, color], nlin, npix, fil1)
            for n in range(nlin):
                loc[color, n] = centroid(c[n, :npix] * win1) - 0.5

            fitme[color, :2] = findedge(loc[color, :], nlin)

            place = np.zeros(nlin)
            for n in range(nlin):
                place[n] = fitme[color, 1] + fitme[color, 0] * n
                win2 = ahamming(npix, place[n])
                loc[color, n] = centroid(c[n, :npix] * win2) - 0.5

            fitme[color, :2] = findedge(loc[color, :], nlin)

        nlin1 = int(round(np.floor(nlin * abs(fitme[0, 0])) / abs(fitme[0, 0])))
        a = a[:nlin1, :, :]

        vslope = fitme[0, 0]
        slope_deg = 180 * np.arctan(abs(vslope)) / np.pi
        print(f'Edge angle: {slope_deg:.3f} degrees')
        if slope_deg < 3.5:
            print(f'High slope warning {slope_deg:.3f} degrees')

        delfac = np.cos(np.arctan(vslope))
        del_ = del_ * delfac

        nn = int(np.floor(npix * nbin))
        mtf = np.zeros((nn, ncol))
        nn2 = int(np.floor(nn / 2) + 1)

        print('Derivative correction')
        dcorr = fir2fix(nn2, 3)
        freq = np.zeros(nn)
        for n in range(nn):
            freq[n] = nbin * n / (del_ * nn)

        freqlim = 1
        nn2out = int(round(nn2 * freqlim / 2))

        win = ahamming(nbin * npix, (nbin * npix + 1) / 2)

        esf = np.zeros((nn, ncol))

        for color in range(ncol):
            point, status = project(a[:, :, color], loc[color, :], fitme[color, 0], nbin)
            esf[:, color] = point

            c = deriv1(point[np.newaxis, :], 1, nn, fil2)
            c = c.flatten()

            mid = centroid(c)
            temp_c = cent(c, round(mid))
            c = temp_c

            c = win * c

            temp = abs(np.fft.fft(c, nn))
            dc = temp[0] if temp[0] != 0 else 1.0
            mtf[:nn2, color] = temp[:nn2] / dc
            mtf[:nn2, color] = mtf[:nn2, color] * dcorr

        freq = freq[:nn2out]
        mtf = mtf[:nn2out, :]
        esf = esf[:nn2out, :]

        rst_freq.append(freq)
        rst_mtf.append(mtf)
        rst_esf.append(esf)

    return rst_freq, rst_mtf, rst_esf


def _sfr_cal_from_roi(roi_list, gamma=0.5, weight=None):
    """Compute SFR from pre-selected ROI images (non-interactive).

    Args:
        roi_list: list of ROI image arrays
        gamma: gamma correction value
        weight: [R_w, G_w, B_w] luminance weights

    Returns:
        rst_freq, rst_mtf, rst_esf
    """
    if weight is None:
        weight = [0.213, 0.715, 0.072]

    nbin = 4
    del_ = 1
    sfr_num = len(roi_list)

    rst_freq = []
    rst_mtf = []
    rst_esf = []

    for n_sfr in range(sfr_num):
        raw = roi_list[n_sfr]
        src_dtype = raw.dtype if isinstance(raw, np.ndarray) else np.array(raw).dtype

        if src_dtype == np.uint16:
            smax = 2 ** 16 - 1
        elif src_dtype == np.uint8:
            smax = 255
        else:
            smax = 1e10

        a = raw.astype(float)

        a = a / smax
        a = np.power(a, gamma)
        a = a * smax

        nlin, npix = a.shape[0], a.shape[1]
        ncol = a.shape[2] if a.ndim == 3 else 1

        if ncol >= 3:
            lum = weight[0] * a[:, :, 0] + weight[1] * a[:, :, 1] + weight[2] * a[:, :, 2]
            cc = np.zeros((nlin, npix, 4))
            cc[:, :, 0] = a[:, :, 0]
            cc[:, :, 1] = a[:, :, 1]
            cc[:, :, 2] = a[:, :, 2]
            cc[:, :, 3] = lum
            a = cc
            ncol = 4
        else:
            a = a[:, :, np.newaxis]
            ncol = 1

        a, nlin, npix, _ = rotatev2(a)

        loc = np.zeros((ncol, nlin))
        fil1 = np.array([0.5, -0.5])
        fil2 = np.array([0.5, 0, -0.5])

        tleft = np.sum(a[:, :5, 0])
        tright = np.sum(a[:, -5:, 0])
        if tleft > tright:
            fil1 = np.array([-0.5, 0.5])
            fil2 = np.array([-0.5, 0, 0.5])

        test = abs((tleft - tright) / (tleft + tright + 1e-10))
        if test < 0.2:
            print(' ** WARNING: Edge contrast is less than 20%')

        win1 = ahamming(npix, (npix + 1) / 2)
        fitme = np.zeros((ncol, 3))

        for color in range(ncol):
            c = deriv1(a[:, :, color], nlin, npix, fil1)
            for n in range(nlin):
                loc[color, n] = centroid(c[n, :npix] * win1) - 0.5
            fitme[color, :2] = findedge(loc[color, :], nlin)
            for n in range(nlin):
                place = fitme[color, 1] + fitme[color, 0] * n
                win2 = ahamming(npix, place)
                loc[color, n] = centroid(c[n, :npix] * win2) - 0.5
            fitme[color, :2] = findedge(loc[color, :], nlin)

        nlin1 = int(round(np.floor(nlin * abs(fitme[0, 0])) / abs(fitme[0, 0])))
        a = a[:nlin1, :, :]

        vslope = fitme[0, 0]
        slope_deg = 180 * np.arctan(abs(vslope)) / np.pi
        print(f'Edge angle: {slope_deg:.3f} degrees')

        delfac = np.cos(np.arctan(vslope))
        del_ = del_ * delfac

        nn = int(np.floor(npix * nbin))
        mtf = np.zeros((nn, ncol))
        nn2 = int(np.floor(nn / 2) + 1)

        dcorr = fir2fix(nn2, 3)
        freq = np.zeros(nn)
        for n in range(nn):
            freq[n] = nbin * n / (del_ * nn)

        freqlim = 1
        nn2out = int(round(nn2 * freqlim / 2))
        win = ahamming(nbin * npix, (nbin * npix + 1) / 2)
        esf = np.zeros((nn, ncol))

        for color in range(ncol):
            point, status = project(a[:, :, color], loc[color, :], fitme[color, 0], nbin)
            esf[:, color] = point
            c = deriv1(point[np.newaxis, :], 1, nn, fil2)
            c = c.flatten()
            # NOTE: Do NOT center the LSF before windowing.
            # The symmetric Hamming window naturally centers around the array midpoint.
            # Using cent() (non-wrap linear shift) truncates the LSF, causing MTF > 1.
            # Circular shift (np.roll) creates wrap artifacts.
            # MATLAB's cent() behavior cannot be faithfully replicated in Python
            # for large shifts (|del| > 500), so we skip centering entirely.
            # This is the correct approach: window at array center, no explicit centering.
            c = win * c
            temp = abs(np.fft.fft(c, nn))
            dc = temp[0] if temp[0] != 0 else 1.0
            mtf[:nn2, color] = temp[:nn2] / dc
            mtf[:nn2, color] = mtf[:nn2, color] * dcorr

        freq = freq[:nn2out]
        mtf = mtf[:nn2out, :]
        esf = esf[:nn2out, :]

        rst_freq.append(freq)
        rst_mtf.append(mtf)
        rst_esf.append(esf)

    return rst_freq, rst_mtf, rst_esf
