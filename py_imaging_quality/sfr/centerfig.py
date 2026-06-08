"""Center figure on screen, equivalent to MATLAB centerfig.m."""
import numpy as np


def centerfig(width, height, scale=0.65):
    """Calculate position vector to center a figure on the screen.

    Args:
        width, height: figure dimensions in pixels
        scale: max screen fraction (default 0.65)

    Returns:
        pos: [left, bottom, width, height] position vector
    """
    # Use default screen size since we can't query in headless
    ms = np.array([0, 0, 1920, 1080])
    center = np.array([ms[0] + ms[2] / 2, ms[1] + ms[3] / 2])
    rat = width / height
    maxw = scale * ms[2] - ms[0]
    maxh = scale * ms[3] - ms[1]
    fw = width / maxw
    fh = height / maxh

    if fw > fh:
        width = maxw
        height = width / rat
    else:
        height = maxh
        width = height * rat

    pos = [center[0] - width / 2, center[1] - height / 2, width, height]
    return pos
