import logging
from typing import Optional, Tuple

import numpy as np

from dolphin.utils import Pathlike

from . import covariance, metrics
from .mle import mle_stack

logger = logging.getLogger(__name__)


def run_cpu(
    slc_stack: np.ndarray,
    half_window: Tuple[int, int],
    strides: Tuple[int, int] = (1, 1),
    beta: float = 0.0,
    reference_idx: int = 0,
    output_cov_file: Optional[Pathlike] = None,
    n_workers: int = 1,
    **kwargs,
):
    """Run the CPU version of the stack covariance estimator and MLE solver.

    Parameters
    ----------
    slc_stack : np.ndarray
        The SLC stack, with shape (n_slc, n_rows, n_cols)
    half_window : Tuple[int, int]
        The half window size as [half_x, half_y] in pixels.
        The full window size is 2 * half_window + 1 for x, y.
    strides : Tuple[int, int], optional
        The (row, col) strides (in pixels) to use for the sliding window.
        By default (1, 1)
    beta : float, optional
        The regularization parameter, by default 0.0.
    reference_idx : int, optional
        The index of the (non compressed) reference SLC, by default 0
    output_cov_file : str, optional
        HDF5 filename to save the estimated covariance at each pixel.
    n_workers : int, optional
        The number of workers to use for (CPU version) multiprocessing.
        If 1 (default), no multiprocessing is used.


    Returns
    -------
    mle_est : np.ndarray[np.complex64]
        The estimated linked phase, with shape (n_slc, n_rows, n_cols)
    temp_coh : np.ndarray[np.float32]
        The temporal coherence at each pixel, shape (n_rows, n_cols)
    """
    C_arrays = covariance.estimate_stack_covariance_cpu(
        slc_stack,
        half_window,
        strides,
        n_workers=n_workers,
    )
    if output_cov_file:
        covariance._save_covariance(output_cov_file, C_arrays)

    output_phase = mle_stack(C_arrays, beta, reference_idx)
    cpx_phase = np.exp(1j * output_phase)
    # Get the temporal coherence
    temp_coh = metrics.estimate_temp_coh(cpx_phase, C_arrays)
    # use the amplitude from the original SLCs
    mle_est = np.abs(slc_stack) * np.exp(1j * output_phase)
    return mle_est, temp_coh