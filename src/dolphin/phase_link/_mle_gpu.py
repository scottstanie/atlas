from math import ceil
from typing import Optional, Tuple

import numpy as np
from numba import cuda

from dolphin.utils import Pathlike

from . import covariance, metrics
from .mle import mle_stack


def run_gpu(
    slc_stack: np.ndarray,
    half_window: Tuple[int, int],
    strides: Tuple[int, int] = (1, 1),
    beta: float = 0.0,
    reference_idx: int = 0,
    output_cov_file: Optional[Pathlike] = None,
    threads_per_block: Tuple[int, int] = (16, 16),
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the GPU version of the stack covariance estimator and MLE solver.

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
    threads_per_block : Tuple[int, int], optional
        The number of threads per block to use for the GPU kernel.
        By default (16, 16)

    Returns
    -------
    mle_est : np.ndarray[np.complex64]
        The estimated linked phase, with shape (n_slc, n_rows, n_cols)
    temp_coh : np.ndarray[np.float32]
        The temporal coherence at each pixel, shape (n_rows, n_cols)
    """
    import cupy as cp

    num_slc, rows, cols = slc_stack.shape

    # Copy the read-only data to the device
    d_slc_stack = cuda.to_device(slc_stack)

    # Make a buffer for each pixel's coherence matrix
    # d_ means "device_", i.e. on the GPU
    out_rows, out_cols = covariance.compute_out_shape((rows, cols), strides)
    d_C_arrays = cp.zeros((out_rows, out_cols, num_slc, num_slc), dtype=np.complex64)

    # Divide up the output shape using a 2D grid
    blocks_x = ceil(out_cols / threads_per_block[0])
    blocks_y = ceil(out_rows / threads_per_block[1])
    blocks = (blocks_x, blocks_y)

    covariance.estimate_stack_covariance_gpu[blocks, threads_per_block](
        d_slc_stack, half_window, strides, d_C_arrays
    )

    if output_cov_file:
        covariance._save_covariance(output_cov_file, d_C_arrays.get())

    d_output_phase = mle_stack(d_C_arrays, beta=beta, reference_idx=reference_idx)
    d_cpx_phase = cp.exp(1j * d_output_phase)

    # Get the temporal coherence
    temp_coh = metrics.estimate_temp_coh(d_cpx_phase, d_C_arrays).get()

    # # https://docs.cupy.dev/en/stable/user_guide/memory.html
    # may just be cached a lot of the huge memory available on aurora
    # But if we need to free GPU memory:
    # cp.get_default_memory_pool().free_all_blocks()

    # use the amplitude from the original SLCs
    mle_est = np.abs(slc_stack) * d_cpx_phase.get()
    return mle_est, temp_coh