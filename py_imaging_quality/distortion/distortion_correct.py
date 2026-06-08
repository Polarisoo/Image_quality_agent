"""Distortion correction, equivalent to MATLAB distortion_correct.m."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .inbox_dis import inbox_dis
from .invert_distortion_poly import invert_distortion_poly
from ..imageread import imageread


def distortion_correct(k1=0, img_d=None):
    """Apply radial distortion correction to an image.

    Uses polynomial inversion to find the mapping from corrected to
    distorted coordinates, then interpolates.

    Args:
        k1: radial distortion coefficient (ru = rd + k1*rd^3)
        img_d: distorted image. If None, user selects one.

    Returns:
        img_u: corrected image (uint8)
    """
    if img_d is None:
        status, img_d, ftype, pathname, f = imageread()
        if f is None:
            print('No image selected')
            return None

    k1 = inbox_dis(k1)
    print('Applying distortion correction...')

    inverseCoeffs = np.array([k1, 0, 1, 0])
    forwardCoeffs = invert_distortion_poly(inverseCoeffs)

    img_d = img_d.astype(float)
    height, width = img_d.shape[:2]
    channels = img_d.shape[2] if img_d.ndim == 3 else 1

    ys = np.arange(height)
    xs = np.arange(width)
    X, Y = np.meshgrid(xs, ys)
    Xc = X - (width + 1) / 2
    Yc = Y - (height + 1) / 2

    # Convert to polar
    THETA = np.arctan2(Yc, Xc)
    RHO_d = np.sqrt(Xc ** 2 + Yc ** 2)
    normFactor = RHO_d[0, 0] if RHO_d[0, 0] != 0 else 1.0
    scaleFactor = np.polyval(inverseCoeffs, 1)
    RHO_d_norm = RHO_d / normFactor * scaleFactor

    RHO_u = np.polyval(forwardCoeffs, RHO_d_norm)

    X_d = RHO_u * normFactor * np.cos(THETA) + (width + 1) / 2
    Y_d = RHO_u * normFactor * np.sin(THETA) + (height + 1) / 2

    # Interpolation
    img_u = np.zeros((height, width, channels), dtype=np.uint8)

    if channels == 1:
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator((ys, xs), img_d.reshape(height, width),
                                        bounds_error=False, fill_value=0)
        pts = np.stack([Y_d.ravel(), X_d.ravel()], axis=-1)
        img_u = interp(pts).reshape(height, width)
        img_u = np.clip(img_u, 0, 255).astype(np.uint8)
    else:
        for c in range(channels):
            from scipy.interpolate import RegularGridInterpolator
            interp = RegularGridInterpolator((ys, xs), img_d[:, :, c],
                                            bounds_error=False, fill_value=0)
            pts = np.stack([Y_d.ravel(), X_d.ravel()], axis=-1)
            img_u[:, :, c] = np.clip(interp(pts).reshape(height, width), 0, 255)

        img_u = img_u.astype(np.uint8)

    return img_u


def _distortion_correct_from_img(k1, img_d):
    """Non-interactive distortion correction for a pre-loaded image.

    Args:
        k1: radial distortion coefficient
        img_d: distorted image (numpy array)

    Returns:
        img_u: corrected image (uint8)
    """
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator
    from .invert_distortion_poly import invert_distortion_poly

    print('Applying distortion correction...')

    inverseCoeffs = np.array([k1, 0, 1, 0])
    forwardCoeffs = invert_distortion_poly(inverseCoeffs)

    img_d = img_d.astype(float)
    height, width = img_d.shape[:2]
    channels = img_d.shape[2] if img_d.ndim == 3 else 1

    ys = np.arange(height)
    xs = np.arange(width)
    X, Y = np.meshgrid(xs, ys)
    Xc = X - (width + 1) / 2
    Yc = Y - (height + 1) / 2

    THETA = np.arctan2(Yc, Xc)
    RHO_d = np.sqrt(Xc ** 2 + Yc ** 2)
    normFactor = RHO_d[0, 0] if RHO_d[0, 0] != 0 else 1.0
    scaleFactor = np.polyval(inverseCoeffs, 1)
    RHO_d_norm = RHO_d / normFactor * scaleFactor

    RHO_u = np.polyval(forwardCoeffs, RHO_d_norm)

    X_d = RHO_u * normFactor * np.cos(THETA) + (width + 1) / 2
    Y_d = RHO_u * normFactor * np.sin(THETA) + (height + 1) / 2

    if channels == 1:
        interp = RegularGridInterpolator((ys, xs), img_d.reshape(height, width),
                                        bounds_error=False, fill_value=0)
        pts = np.stack([Y_d.ravel(), X_d.ravel()], axis=-1)
        img_u = np.clip(interp(pts).reshape(height, width), 0, 255).astype(np.uint8)
    else:
        img_u = np.zeros((height, width, channels), dtype=np.uint8)
        for c in range(channels):
            interp = RegularGridInterpolator((ys, xs), img_d[:, :, c],
                                            bounds_error=False, fill_value=0)
            pts = np.stack([Y_d.ravel(), X_d.ravel()], axis=-1)
            img_u[:, :, c] = np.clip(interp(pts).reshape(height, width), 0, 255)
        img_u = img_u.astype(np.uint8)

    return img_u
