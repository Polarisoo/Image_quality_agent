"""Image file reader, equivalent to MATLAB imageread.m."""
import os
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog


def imageread(filename=None, nlin=0, npix=0):
    """Read image files with optional file browser.

    Supports: tif, jpeg, bmp, gif, png.

    Args:
        filename: optional file name or path
        nlin, npix: optional dimensions (for raw files, not used)

    Returns:
        status: 0 = OK, 1 = not OK
        dat1: image data array (numpy)
        ftype: file extension
        fpath: file path
        f: file handle/name
    """
    status = 0

    if filename is None or filename == '':
        file_path = filedialog.askopenfilename(
            title='Select input image file',
            filetypes=[
                ('Supported images',
                 '*.tif;*.TIF;*.tiff;*.TIFF;*.jpg;*.jpeg;*.JPG;*.JPEG;*.bmp;*.BMP;*.png;*.PNG;*.gif;*.GIF'),
                ('All files', '*.*')
            ]
        )
        if not file_path:
            status = 1
            return status, 0, 0, '', None
        filename = file_path
    else:
        file_path = filename

    fname = os.path.basename(file_path)
    fpath = os.path.dirname(file_path)
    ftype = os.path.splitext(fname)[1].lower().replace('.', '')

    try:
        img = Image.open(file_path)
        dat1 = np.array(img)
    except Exception as e:
        print(f'Error reading image: {e}')
        status = 1
        return status, 0, 0, '', None

    return status, dat1, ftype, fpath, fname
