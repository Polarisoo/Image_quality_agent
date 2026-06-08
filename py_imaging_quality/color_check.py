"""ColorChecker 24-patch color accuracy test, equivalent to MATLAB ColorCheck.m."""
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
from .getmyroi import getmyroi


def color_check(array1):
    """Analyze ColorChecker 24-patch chart.

    Compares measured colors against ideal Lab values for X-Rite ColorChecker.

    Args:
        array1: input image array (RGB)

    Returns:
        delta_L, delta_a, delta_b, delta_E, delta_C: color difference metrics
        whiteB: white balance errors for 6 gray patches
    """
    # X-Rite ColorChecker ideal Lab values (D65)
    Lab_ideal = np.array([
        [37.986, 13.555, 14.059],
        [65.711, 18.13, 17.81],
        [49.927, -4.88, -21.925],
        [43.139, -13.095, 21.905],
        [55.112, 8.844, -25.339],
        [70.719, -33.397, -0.199],
        [62.661, 36.067, 57.096],
        [40.02, 10.41, -45.964],
        [51.124, 48.239, 16.248],
        [30.325, 22.976, -21.587],
        [72.532, -23.709, 57.255],
        [71.941, 19.363, 67.857],
        [28.778, 14.179, -50.297],
        [55.261, -38.342, 31.37],
        [42.101, 53.378, 28.19],
        [81.733, 4.039, 79.819],
        [51.935, 49.986, -14.574],
        [51.038, -28.631, -28.638],
        [96.539, -0.425, 1.186],
        [81.257, -0.638, -0.335],
        [66.766, -0.734, -0.504],
        [50.867, -0.153, -0.27],
        [35.656, -0.421, -1.231],
        [20.461, -0.079, -0.093]
    ])

    select, _ = getmyroi(array1, 24, dialtext='Select 24 ColorChecker patches (row by row)')
    if select is None:
        return None, None, None, None, None, None

    # Extract average RGB per patch
    RGB = np.zeros((24, 3))
    for i in range(24):
        patch = select[i]
        h, w = patch.shape[:2]
        n_pixels = h * w
        R = np.sum(patch[:, :, 0]) / n_pixels
        G = np.sum(patch[:, :, 1]) / n_pixels
        B = np.sum(patch[:, :, 2]) / n_pixels
        RGB[i, :] = [R, G, B]

    RGB_norm = RGB / 255.0
    Lab = rgb2lab(RGB_norm.reshape(1, 24, 3)).reshape(24, 3)

    # Indices: 0-23 = per-patch, 24 = mean, 25 = max
    delta_L = np.zeros(26)
    delta_a = np.zeros(26)
    delta_b = np.zeros(26)
    delta_E = np.zeros(26)
    delta_C = np.zeros(26)
    Saturation = np.zeros(25)

    for i in range(24):
        delta_L[i] = abs(Lab[i, 0] - Lab_ideal[i, 0])
        delta_a[i] = abs(Lab[i, 1] - Lab_ideal[i, 1])
        delta_b[i] = abs(Lab[i, 2] - Lab_ideal[i, 2])
        delta_E[i] = np.sqrt(delta_L[i] ** 2 + delta_a[i] ** 2 + delta_b[i] ** 2)
        delta_C[i] = np.sqrt(delta_a[i] ** 2 + delta_b[i] ** 2)
        sat_meas = np.sqrt(Lab[i, 1] ** 2 + Lab[i, 2] ** 2)
        sat_ideal = np.sqrt(Lab_ideal[i, 1] ** 2 + Lab_ideal[i, 2] ** 2)
        Saturation[i] = sat_meas / (sat_ideal + 1e-10)

        if i < 18:
            Saturation[24] += Saturation[i]

    delta_L[24] = np.mean(delta_L[:24])
    delta_a[24] = np.mean(delta_a[:24])
    delta_b[24] = np.mean(delta_b[:24])
    delta_E[24] = np.mean(delta_E[:24])
    delta_C[24] = np.mean(delta_C[:24])
    Saturation[24] = np.mean(Saturation[:18])

    delta_L[25] = np.max(delta_L[:24])
    delta_a[25] = np.max(delta_a[:24])
    delta_b[25] = np.max(delta_b[:24])
    delta_E[25] = np.max(delta_E[:24])
    delta_C[25] = np.max(delta_C[:24])

    # White balance from last 6 patches (grays)
    whiteB = np.zeros(6)
    piancha = np.zeros(6)
    for i in range(6):
        rgb_patch = RGB[i + 18, :]
        whiteB[i] = max(rgb_patch) - min(rgb_patch)
        piancha[i] = np.argmax(rgb_patch) + 1

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(Lab[:, 1], Lab[:, 2], 'go', markersize=10, markerfacecolor='g',
            markeredgecolor='k', linewidth=2, label='Measured')
    ax.plot(Lab_ideal[:, 1], Lab_ideal[:, 2], 'cs', markersize=10,
            markerfacecolor='c', markeredgecolor='k', linewidth=2, label='Ideal')

    for i in range(24):
        ax.plot([Lab[i, 1], Lab_ideal[i, 1]], [Lab[i, 2], Lab_ideal[i, 2]], 'k')

    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.grid(True)
    ax.legend()
    ax.text(-95, -70, f'mean camera chroma(saturation) = {Saturation[24] * 100:.1f}%')
    ax.text(-95, -80, f'delta_C chroma errors: mean={delta_C[24]:.2f}  max={delta_C[25]:.2f}')
    ax.text(-95, -90, f'delta_E: mean={delta_E[24]:.2f}  max={delta_E[25]:.2f}')
    ax.set_xlabel('a*')
    ax.set_ylabel('b*')
    ax.set_title('ColorChecker Color Accuracy')

    plt.show()

    return delta_L, delta_a, delta_b, delta_E, delta_C, whiteB


def _color_check_from_rois(roi_list):
    """Compute color check from pre-selected ROI patches (non-interactive).

    Args:
        roi_list: list of 24 patch image arrays (one per ColorChecker patch)

    Returns:
        delta_E, delta_C, Saturation arrays (each has 26 elements: 0-23 per patch, 24=mean, 25=max)
    """
    Lab_ideal = np.array([
        [37.986, 13.555, 14.059], [65.711, 18.13, 17.81],
        [49.927, -4.88, -21.925], [43.139, -13.095, 21.905],
        [55.112, 8.844, -25.339], [70.719, -33.397, -0.199],
        [62.661, 36.067, 57.096], [40.02, 10.41, -45.964],
        [51.124, 48.239, 16.248], [30.325, 22.976, -21.587],
        [72.532, -23.709, 57.255], [71.941, 19.363, 67.857],
        [28.778, 14.179, -50.297], [55.261, -38.342, 31.37],
        [42.101, 53.378, 28.19], [81.733, 4.039, 79.819],
        [51.935, 49.986, -14.574], [51.038, -28.631, -28.638],
        [96.539, -0.425, 1.186], [81.257, -0.638, -0.335],
        [66.766, -0.734, -0.504], [50.867, -0.153, -0.27],
        [35.656, -0.421, -1.231], [20.461, -0.079, -0.093]
    ])

    n_patches = min(len(roi_list), 24)
    RGB = np.zeros((n_patches, 3))
    for i in range(n_patches):
        patch = roi_list[i]
        h, w = patch.shape[:2]
        RGB[i, 0] = np.sum(patch[:, :, 0]) / (h * w)
        RGB[i, 1] = np.sum(patch[:, :, 1]) / (h * w)
        RGB[i, 2] = np.sum(patch[:, :, 2]) / (h * w)

    Lab = rgb2lab((RGB / 255.0).reshape(1, n_patches, 3)).reshape(n_patches, 3)

    delta_E = np.zeros(26)
    delta_C = np.zeros(26)
    Saturation = np.zeros(25)

    for i in range(n_patches):
        dL = abs(Lab[i, 0] - Lab_ideal[i, 0])
        da = abs(Lab[i, 1] - Lab_ideal[i, 1])
        db = abs(Lab[i, 2] - Lab_ideal[i, 2])
        delta_E[i] = np.sqrt(dL**2 + da**2 + db**2)
        delta_C[i] = np.sqrt(da**2 + db**2)
        s_m = np.sqrt(Lab[i, 1]**2 + Lab[i, 2]**2)
        s_i = np.sqrt(Lab_ideal[i, 1]**2 + Lab_ideal[i, 2]**2)
        Saturation[i] = s_m / (s_i + 1e-10)
        if i < 18:
            Saturation[24] += Saturation[i]

    delta_E[24] = np.mean(delta_E[:n_patches])
    delta_C[24] = np.mean(delta_C[:n_patches])
    Saturation[24] = np.mean(Saturation[:18]) if n_patches >= 18 else np.mean(Saturation[:n_patches])
    delta_E[25] = np.max(delta_E[:n_patches])
    delta_C[25] = np.max(delta_C[:n_patches])

    # Plot
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(Lab[:, 1], Lab[:, 2], 'go', ms=10, mfc='g', mec='k', lw=2, label='Measured')
    ax.plot(Lab_ideal[:n_patches, 1], Lab_ideal[:n_patches, 2], 'cs', ms=10, mfc='c', mec='k', lw=2, label='Ideal')
    for i in range(n_patches):
        ax.plot([Lab[i, 1], Lab_ideal[i, 1]], [Lab[i, 2], Lab_ideal[i, 2]], 'k', lw=0.5)
    ax.set_xlim(-100, 100); ax.set_ylim(-100, 100)
    ax.grid(True); ax.legend(); ax.set_xlabel('a*'); ax.set_ylabel('b*')
    ax.set_title('ColorChecker Accuracy')
    ax.text(-95, -70, f'chroma saturation = {Saturation[24]*100:.1f}%')
    ax.text(-95, -80, f'delta_C: mean={delta_C[24]:.2f} max={delta_C[25]:.2f}')
    ax.text(-95, -90, f'delta_E: mean={delta_E[24]:.2f} max={delta_E[25]:.2f}')
    plt.tight_layout(); plt.show()

    return delta_E, delta_C, Saturation
