import numpy as np
from scipy.interpolate import CubicSpline


def get_interp(x_bins, y_bins):
    """
    Get the interpolated profiles.

    Parameters:
    ----------
    x_bins: array
        Array of the x-param.
    y_bins: array
        Array of the y-param.

    Returns:
    -------
    interp: CubicSpline
        Interpolated profile.
    """

    x_order = np.argsort(x_bins)
    x_increasing_mask = np.append([True], np.diff(x_bins[x_order]) > 0)

    x_bins = x_bins[x_order][x_increasing_mask]
    y_bins = y_bins[x_order][x_increasing_mask]

    finite_mask = np.logical_and(np.isfinite(x_bins), np.isfinite(y_bins))

    return CubicSpline(x_bins[finite_mask], y_bins[finite_mask])
