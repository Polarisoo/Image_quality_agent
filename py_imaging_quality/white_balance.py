"""White balance analysis, equivalent to MATLAB white_balance.m and whitebalance.m."""
import numpy as np
import matplotlib.pyplot as plt


def white_balance(array1):
    """Analyze white balance by selecting 6 gray patches.

    Args:
        array1: input image array

    Returns:
        whiteB: RGB max-min differences for 6 patches
        piancha: which channel is max (1=R, 2=G, 3=B)
    """
    from ..getmyroi import getmyroi

    select, _ = getmyroi(array1, 6, dialtext='Select 6 gray patches (darkest to lightest)')
    if select is None:
        return None, None

    n_patches = len(select)
    AR = np.zeros(n_patches)
    AG = np.zeros(n_patches)
    AB = np.zeros(n_patches)
    whiteB = np.zeros(n_patches)
    piancha = np.zeros(n_patches)

    for i in range(n_patches):
        patch = select[i]
        h, w = patch.shape[:2]
        n_pixels = h * w
        R = np.sum(patch[:, :, 0])
        G = np.sum(patch[:, :, 1])
        B = np.sum(patch[:, :, 2])
        AR[i] = R / n_pixels
        AG[i] = G / n_pixels
        AB[i] = B / n_pixels

        rgb = [AR[i], AG[i], AB[i]]
        whiteB[i] = max(rgb) - min(rgb)
        piancha[i] = rgb.index(max(rgb)) + 1

    return whiteB, piancha


def whitebalance_display(text1, text2, img):
    """Display white balance result figure.

    Args:
        text1: title text
        text2: result text
        img: image to display
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    ax1.imshow(img)
    ax1.set_title(text1)
    ax2.text(0.5, 0.5, text2, fontsize=12, ha='center', va='center',
             transform=ax2.transAxes)
    ax2.axis('off')
    plt.tight_layout()
    plt.show()
    return fig


def _wb_from_rois(roi_list):
    """Compute white balance from pre-selected gray patches (non-interactive).

    Args:
        roi_list: list of 6 gray patch images (darkest to lightest)

    Returns:
        whiteB: RGB max-min differences for 6 patches
        piancha: which channel is max (1=R, 2=G, 3=B)
    """
    n = len(roi_list)
    whiteB = np.zeros(n)
    piancha = np.zeros(n)

    for i in range(n):
        patch = roi_list[i]
        h, w = patch.shape[:2]
        npx = h * w
        AR = np.sum(patch[:, :, 0]) / npx
        AG = np.sum(patch[:, :, 1]) / npx
        AB = np.sum(patch[:, :, 2]) / npx
        rgb = [AR, AG, AB]
        whiteB[i] = max(rgb) - min(rgb)
        piancha[i] = rgb.index(max(rgb)) + 1

    return whiteB, piancha
