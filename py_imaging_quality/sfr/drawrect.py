"""Draw rectangle on image, equivalent to MATLAB drawrect.m."""
import numpy as np


def drawrect(src, ul, lr, line_size=1):
    """Draw a white rectangle on an image.

    Args:
        src: input image (nlin, npix, ncol)
        ul: upper-left corner (x, y)
        lr: lower-right corner (x, y)
        line_size: line width

    Returns:
        dest: image with rectangle drawn
    """
    src = np.asarray(src)
    dest = src.copy()
    if src.ndim == 2:
        dest = dest[:, :, np.newaxis]
        src_was_2d = True
    else:
        src_was_2d = False

    nlin, npix, ncol = dest.shape
    color = 255

    ul, lr = np.asarray(ul), np.asarray(lr)
    if ul[0] > lr[0]:
        ul[0], lr[0] = lr[0], ul[0]
    if ul[1] > lr[1]:
        ul[1], lr[1] = lr[1], ul[1]

    # Clamp to boundaries
    ul = np.clip(ul, 0, [npix, nlin])
    lr = np.clip(lr, 0, [npix, nlin])

    temp = [line_size, ul[0], ul[1], npix - lr[0] + 1, nlin - lr[1] + 1]
    line_size = min(temp)
    if line_size < 1:
        line_size = 1

    for dl in range(line_size):
        d = dl
        y1, y2 = max(0, ul[1] - d), min(nlin - 1, lr[1] + d)
        x1, x2 = max(0, ul[0] - d), min(npix - 1, lr[0] + d)
        dest[y1, x1:x2 + 1, :] = color  # top
        dest[y2, x1:x2 + 1, :] = color  # bottom
        dest[y1:y2 + 1, x1, :] = color  # left
        dest[y1:y2 + 1, x2, :] = color  # right

    if src_was_2d:
        dest = dest[:, :, 0]
    return dest
