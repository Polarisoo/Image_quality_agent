"""Interactive ROI selection, equivalent to MATLAB getmyroi.m."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle


def getmyroi(src, num=1, tol=1, dialtext='Select ROI', ax=None):
    """Select and return regions of interest via interactive GUI.

    Args:
        src: input image array
        num: number of ROIs to select (default 1)
        tol: tolerance in pixels (default 1)
        dialtext: dialog title
        ax: optional existing matplotlib axes to draw on

    Returns:
        select: list of ROI image arrays
        coord: (4, num) array, columns = [ul_x, ul_y, lr_x, lr_y]
    """
    src = np.asarray(src)
    select = []
    coord = np.zeros((num, 4), dtype=int)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.canvas.manager.set_window_title(dialtext)
        external_fig = True
    else:
        external_fig = False

    ax.clear()
    if src.ndim == 2 or src.shape[2] == 1:
        ax.imshow(src.squeeze(), cmap='gray')
    else:
        ax.imshow(src)

    ax.set_title(f'{dialtext}  (Draw {num} ROI(s), close window when done)')

    current_roi = [0]
    patches = []
    # Make axes visible to onselect via closure
    selector_ax = ax

    def onselect(eclick, erelease):
        if current_roi[0] >= num:
            return
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        ul_x, lr_x = min(x1, x2), max(x1, x2)
        ul_y, lr_y = min(y1, y2), max(y1, y2)

        h, w = src.shape[:2]
        ul_x = max(0, ul_x)
        ul_y = max(0, ul_y)
        lr_x = min(w - 1, lr_x)
        lr_y = min(h - 1, lr_y)

        roi_w, roi_h = lr_x - ul_x, lr_y - ul_y
        if tol != 0:
            if roi_w < tol:
                ul_x, lr_x = 0, w - 1
            if roi_h < tol:
                ul_y, lr_y = 0, h - 1

        idx = current_roi[0]
        coord[idx, :] = [ul_x, ul_y, lr_x, lr_y]
        roi_data = src[ul_y:lr_y + 1, ul_x:lr_x + 1].copy()
        select.append(roi_data)

        rect = Rectangle((ul_x, ul_y), lr_x - ul_x, lr_y - ul_y,
                        linewidth=2, edgecolor='red', facecolor='none')
        selector_ax.add_patch(rect)
        patches.append(rect)
        selector_ax.figure.canvas.draw()

        current_roi[0] += 1
        if current_roi[0] >= num:
            selector_ax.set_title(f'{dialtext}  -  All {num} ROI(s) selected. Close window.')
        else:
            selector_ax.set_title(f'{dialtext}  -  ROI {current_roi[0]}/{num} selected')

    rs = RectangleSelector(ax, onselect, useblit=True,
                           button=[1], minspanx=5, minspany=5,
                           spancoords='pixels', interactive=True)

    if external_fig:
        plt.tight_layout()
        plt.show()

    if len(select) < num:
        return None, None

    return select, coord.T
