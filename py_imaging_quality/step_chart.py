"""Step chart analysis, equivalent to MATLAB stepdart.m."""
import numpy as np
import matplotlib.pyplot as plt
from .getmyroi import getmyroi


def step_chart(array1, m=20):
    """Analyze a step chart image for contrast and number of detectable steps.

    Args:
        array1: input image array (RGB or grayscale)
        m: number of patches to select (default 20)

    Returns:
        n: number of detected steps
        LOG1: sorted log10 pixel values
    """
    # Convert to grayscale if needed
    if array1.ndim == 3:
        I_gray = np.dot(array1[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        I_gray = array1.copy()

    select, _ = getmyroi(I_gray, m, dialtext=f'Select {m} step chart patches (darkest to lightest)')
    if select is None:
        return 0, None

    AG = np.zeros(m)
    for i in range(m):
        patch = select[i]
        h, w = patch.shape[:2]
        AG[i] = np.sum(patch) / (h * w)

    Pixel = AG / 255.0
    LOG = np.log10(np.maximum(Pixel, 1e-10))

    # Count steps
    n = 0
    for i in range(m - 1):
        deltaG = AG[i + 1] - AG[i]
        if abs(deltaG) > 1:
            n += 1

    duibidu = (np.max(AG) - np.min(AG)) / 255.0
    LOG1 = np.sort(LOG)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, m + 1), LOG1, 's-')
    ax.set_xlabel('Patch Number')
    ax.set_ylabel('Log(Pixel level / 255)')
    DELTA_AG = np.max(LOG) - np.min(LOG)
    ax.text(3 * m / 4, np.min(LOG) + 1 * DELTA_AG / 4,
            f'{n} step detected')
    ax.text(3 * m / 4, np.min(LOG) + 1 * DELTA_AG / 5,
            f'contrast ratio is {duibidu:.2f}')
    ax.set_title('Step Chart Analysis')
    plt.show()

    return n, LOG1


def _step_chart_from_rois(roi_list):
    """Compute step chart from pre-selected ROI patches (non-interactive).

    Args:
        roi_list: list of patch image arrays (from darkest to lightest)

    Returns:
        n: number of detected steps
        duibidu: contrast ratio
    """
    m = len(roi_list)
    AG = np.zeros(m)
    for i in range(m):
        patch = roi_list[i]
        if patch.ndim == 3:
            patch = np.dot(patch[..., :3], [0.2989, 0.5870, 0.1140])
        h, w = patch.shape[:2]
        AG[i] = np.sum(patch) / (h * w)

    Pixel = AG / 255.0
    LOG = np.log10(np.maximum(Pixel, 1e-10))
    duibidu = (np.max(AG) - np.min(AG)) / 255.0

    n = 0
    for i in range(m - 1):
        if abs(AG[i + 1] - AG[i]) > 1:
            n += 1

    LOG1 = np.sort(LOG)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, m + 1), LOG1, 's-')
    ax.set_xlabel('Patch Number')
    ax.set_ylabel('Log(Pixel level / 255)')
    d = np.max(LOG) - np.min(LOG)
    ax.text(3 * m / 4, np.min(LOG) + d / 4, f'{n} step detected')
    ax.text(3 * m / 4, np.min(LOG) + d / 5, f'contrast ratio is {duibidu:.2f}')
    ax.set_title('Step Chart Analysis')
    plt.tight_layout(); plt.show()

    return n, duibidu
