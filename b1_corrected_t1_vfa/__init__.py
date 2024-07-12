import numpy as np
from scipy.optimize import curve_fit


def flash_equation(fas, tr, s0, t1):
    E1 = np.exp(-tr / t1)
    return s0 * (1 - E1) * np.sin(np.deg2rad(fas)) / (1 - np.cos(np.deg2rad(fas)) * E1)


def fit_t1_vfa(data_stack, fas, tr, b1_map):
    """Fit T1 map into VFA data using a B1 correction.

    Parameters
    ----------
    data_stack : np.ndarray
        Multiple GRE acquisitions corresponding to the different flip angles,
        stacked along the last axis. Shape: `(*grid_shape, nfa)`
    fa : np.ndarray
        1D vector of nominal flip angles in degrees. Shape: `(nfa,)`
    tr : float
        Repetition time. Its units will define the units of the output T1 map.
    b1_map : np.ndarray
        Map of the B1 field in the fractions of the unit. Shape must match
        the shape of the leading dimensions of the data (i.e. `grid_shape`).

    Returns
    -------
    s0, t1 : np.ndarray
        Arrays of shape `grid_shape` containing the results of the voxel-wise fitting.
        Voxels where fitting failed contain NaNs.
    """

    assert (
        data_stack.ndim == 4
    ), f"data_stack is expected to be 4D, got shape: {data_stack.shape}"

    grid_shape, nfa = data_stack.shape[:-1], data_stack.shape[-1]
    assert fas.shape == (
        nfa,
    ), f"{fas.shape=} does not agree with the rightmost axis of the data ({nfa})."
    assert (
        b1_map.shape == grid_shape
    ), f"{b1_map.shape=} does not agree with the leading axes of the data ({grid_shape})"

    s0_map = np.full(grid_shape, np.nan)
    t1_map = np.full(grid_shape, np.nan)

    for iread in range(grid_shape[0]):
        for jline in range(grid_shape[1]):
            for kslice in range(grid_shape[2]):
                signal = data_stack[iread, jline, kslice, :]
                corrected_fas = fas * b1_map[iread, jline, kslice]
                try:
                    popt, _ = curve_fit(
                        lambda fas, s0, t1: flash_equation(fas, tr, s0, t1),
                        corrected_fas,
                        signal,
                        bounds=(0, np.inf),
                    )
                    s0_map[iread, jline, kslice] = popt[0]
                    t1_map[iread, jline, kslice] = popt[1]
                except RuntimeError:
                    pass

    return s0_map, t1_map
