from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike
from scipy import ndimage


def filter_long_wavelength(
    unwrapped_phase: ArrayLike,
    correlation: ArrayLike,
    mask_cutoff: float = 0.5,
    wavelength_cutoff: float = 50 * 1e3,
    pixel_spacing: float = 30,
) -> np.ndarray:
    """Filter out signals with spatial wavelength longer than a threshold.

    Parameters
    ----------
    unwrapped_phase : np.ndarray, 2D complex array
        Unwrapped interferogram phase to filter.
    correlation : Arraylike, 2D
        Array of interferometric correlation from 0 to 1.
    mask_cutoff: float
        Threshold to use on `correlation` so that pixels where
        `correlation[i, j] > mask_cutoff` are used and the rest are ignored.
        The default is 0.5.
    wavelength_cutoff: float
        Spatial wavelength threshold to filter the unwrapped phase.
        Signals with wavelength longer than 'wavelength_cutoff' are filtered out.
        The default is 50*1e3 (m).
    pixel_spacing : float
        Pixel spatial spacing. Assume same spacing for x, y axes.
        The default is 30 (m).

    Returns
    -------
    filtered_ifg : 2D complex array
        filtered interferogram that does not contain signals with spatial wavelength
        longer than a threshold.

    """
    nrow, ncol = correlation.shape
    mask = (correlation > mask_cutoff).astype(bool)
    mask_boundary = ~(correlation == 0).astype(bool)

    plane = fit_ramp_plane(unwrapped_phase, mask)
    unw_ifg_interp = np.copy(unwrapped_phase)
    unw_ifg_interp[~mask * mask_boundary] = plane[~mask * mask_boundary]

    reflect_fill = _fill_boundary_area(unw_ifg_interp, mask)

    sigma = _get_filter_cutoff(wavelength_cutoff, pixel_spacing)
    lowpass_filtered = ndimage.gaussian_filter(reflect_fill, sigma)
    filtered_ifg = unwrapped_phase - lowpass_filtered * mask_boundary

    return filtered_ifg


def _get_filter_cutoff(wavelength_cutoff: float, pixel_spacing: float) -> np.ndarray:
    """Find the Gaussian filter to remove long wavelength signals."""
    cutoff_value = 0.5
    sigma_f = 1 / wavelength_cutoff / np.sqrt(np.log(1 / cutoff_value))
    sigma_x = 1 / np.pi / 2 / sigma_f
    return sigma_x / pixel_spacing


def fit_ramp_plane(unw_ifg: ArrayLike, mask: ArrayLike) -> np.ndarray:
    """Fit a ramp plane to the given data.

    Parameters
    ----------
    unw_ifg : ArrayLike
        2D array where the unwrapped interferogram data is stored.
    mask : ArrayLike
        2D boolean array indicating the valid (non-NaN) pixels.

    Returns
    -------
    np.ndarray
        2D array of the fitted ramp plane.

    """
    # Extract data for non-NaN & masked pixels
    Y = unw_ifg[mask]
    Xdata = np.argwhere(mask)  # Get indices of non-NaN & masked pixels

    # Include the intercept term (bias) in the model
    X = np.c_[np.ones((len(Xdata))), Xdata]

    # Compute the parameter vector theta using the least squares solution
    theta = np.linalg.pinv(X.T @ X) @ X.T @ Y

    # Prepare grid for the entire image
    nrow, ncol = unw_ifg.shape
    X1_, X2_ = np.mgrid[:nrow, :ncol]
    X_ = np.hstack(
        (np.reshape(X1_, (nrow * ncol, 1)), np.reshape(X2_, (nrow * ncol, 1)))
    )
    X_ = np.hstack((np.ones((nrow * ncol, 1)), X_))

    # Compute the fitted plane
    plane = np.reshape(X_ @ theta, (nrow, ncol))

    return plane


def _fill_boundary_area(unw_ifg_interp: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill the boundary area by reflecting pixel values."""
    edge_filled = _fill_edge_values(unw_ifg_interp, mask)
    reflect_filled = _reflect_boundary_values(edge_filled, unw_ifg_interp, mask)
    return reflect_filled


def _fill_edge_values(unw_ifg_interp: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill edge values of the boundary area."""
    edge_filled = np.copy(unw_ifg_interp)
    corner_indices = _get_corner_indices(mask)

    for direction in ["north", "south", "west", "east"]:
        edge_filled = _fill_direction(
            edge_filled, unw_ifg_interp, corner_indices, direction
        )

    return edge_filled


def _reflect_boundary_values(
    edge_filled: np.ndarray, unw_ifg_interp: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Reflect boundary values to fill the boundary area."""
    reflect_filled = np.copy(edge_filled)
    corner_indices = _get_corner_indices(mask)

    for direction in ["north", "south", "west", "east"]:
        reflect_filled = _reflect_direction(
            reflect_filled, edge_filled, unw_ifg_interp, corner_indices, direction
        )

    reflect_filled = _fill_corner_areas(reflect_filled, corner_indices)

    return reflect_filled


def _get_corner_indices(mask: np.ndarray) -> dict[str, np.ndarray]:
    """Get indices of corner pixels."""
    valid_pixels = np.argwhere(mask)
    return {
        "nw": valid_pixels[np.argmin(valid_pixels[:, 0])],
        "se": valid_pixels[np.argmax(valid_pixels[:, 0])],
        "sw": valid_pixels[np.argmin(valid_pixels[:, 1])],
        "ne": valid_pixels[np.argmax(valid_pixels[:, 1])],
    }


def _fill_direction(
    arr: np.ndarray,
    original: np.ndarray,
    corners: Mapping[str, np.ndarray],
    direction: str,
) -> np.ndarray:
    """Fill edge values for a specific direction."""
    if direction == "north":
        start, end = corners["nw"][1], corners["ne"][1] + 1
        for col in range(start, end):
            zero_count = np.count_nonzero(original[: corners["ne"][0] + 1, col] == 0)
            if zero_count > 0:
                arr[:zero_count, col] = arr[zero_count, col]
    elif direction == "south":
        start, end = corners["sw"][1], corners["se"][1] + 1
        for col in range(start, end):
            zero_count = np.count_nonzero(original[corners["sw"][0] + 1 :, col] == 0)
            if zero_count > 0:
                arr[-zero_count:, col] = arr[-zero_count - 1, col]
    elif direction == "west":
        start, end = corners["nw"][0], corners["sw"][0] + 1
        for row in range(start, end):
            zero_count = np.count_nonzero(original[row, : corners["nw"][1] + 1] == 0)
            if zero_count > 0:
                arr[row, :zero_count] = arr[row, zero_count]
    elif direction == "east":
        start, end = corners["ne"][0], corners["se"][0] + 1
        for row in range(start, end):
            zero_count = np.count_nonzero(original[row, corners["se"][1] + 1 :] == 0)
            if zero_count > 0:
                arr[row, -zero_count:] = arr[row, -zero_count - 1]
    return arr


def _reflect_direction(
    reflect: np.ndarray,
    edge: np.ndarray,
    original: np.ndarray,
    corners: Mapping[str, np.ndarray],
    direction: str,
) -> np.ndarray:
    """Reflect boundary values for a specific direction."""
    if direction == "north":
        start, end = corners["nw"][1], corners["ne"][1] + 1
        for col in range(start, end):
            zero_count = np.count_nonzero(original[: corners["ne"][0] + 1, col] == 0)
            if zero_count > 0:
                reflect[:zero_count, col] = np.flipud(
                    edge[zero_count : zero_count * 2, col]
                )
    elif direction == "south":
        start, end = corners["sw"][1], corners["se"][1] + 1
        for col in range(start, end):
            zero_count = np.count_nonzero(original[corners["sw"][0] + 1 :, col] == 0)
            if zero_count > 0:
                reflect[-zero_count:, col] = np.flipud(
                    edge[-zero_count * 2 : -zero_count, col]
                )
    elif direction == "west":
        start, end = corners["nw"][0], corners["sw"][0] + 1
        for row in range(start, end):
            zero_count = np.count_nonzero(original[row, : corners["nw"][1] + 1] == 0)
            if zero_count > 0:
                reflect[row, :zero_count] = np.flipud(
                    edge[row, zero_count : zero_count * 2]
                )
    elif direction == "east":
        start, end = corners["ne"][0], corners["se"][0] + 1
        for row in range(start, end):
            zero_count = np.count_nonzero(original[row, corners["se"][1] + 1 :] == 0)
            if zero_count > 0:
                reflect[row, -zero_count:] = np.flipud(
                    edge[row, -zero_count * 2 : -zero_count]
                )
    return reflect


def _fill_corner_areas(
    reflect: np.ndarray, corners: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Fill corner areas of the boundary."""
    nrow, ncol = reflect.shape

    # Upper left corner
    reflect[: corners["nw"][0], : corners["nw"][1]] = np.flipud(
        reflect[corners["nw"][0] : corners["nw"][0] * 2, : corners["nw"][1]]
    )

    # Upper right corner
    reflect[: corners["ne"][0], corners["ne"][1] + 1 :] = np.fliplr(
        reflect[
            : corners["ne"][0],
            corners["ne"][1] + 1 - (ncol - corners["ne"][1] - 1) : corners["ne"][1] + 1,
        ]
    )

    # Lower left corner
    reflect[corners["sw"][0] + 1 :, : corners["sw"][1]] = np.fliplr(
        reflect[corners["sw"][0] + 1 :, corners["sw"][1] : corners["sw"][1] * 2]
    )

    # Lower right corner
    reflect[corners["se"][0] + 1 :, corners["se"][1] + 1 :] = np.flipud(
        reflect[
            corners["se"][0] + 1 - (nrow - corners["se"][0] - 1) : corners["se"][0] + 1,
            corners["se"][1] + 1 :,
        ]
    )

    return reflect
