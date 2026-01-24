from collections.abc import Callable
import pathlib
import sys

import numpy as np
import scipy
import opinf


def generate_training_data(
    n_samples: int,
    n_timesteps: int,
    q_0: Callable[[np.ndarray], np.ndarray],
    u: Callable[[int], np.ndarray] | None = None,
):
    """Generate sample data to be used for Operator Inference.

    Args:
    n_samples: Number of spatial samples.
    n_timesteps: Number of time steps.
    q_0: Initial condition function. Accepts an array of spatial locations and
        returns an array of initial condition values.
    u: External input function. Accepts a time value and returns the input
        values for that time step. If None, u defaults is the zero function
        (for a model that does not use external inputs)

    Returns:
    t: Array of time points.
    Q: Array of "observed" snapshots, shape (n_samples, n_timesteps).
    """
    external_inputs = True
    if u is None:
        external_inputs = False

        def u(t):
            return 0

    # construct the spatial and temporal domains
    x = np.linspace(0, 1, n_samples + 2)[1:-1]
    dx = x[1] - x[0]
    t = np.linspace(0, 1, n_timesteps)

    # construct the matrix of linear operators
    diags = np.array([1, -2, 1]) / dx**2
    A = scipy.sparse.diags(diags, [-1, 0, 1], (n_samples, n_samples))

    # construct the matrix of external input operators
    B = np.zeros_like(x)
    B[0], B[-1] = 1 / dx**2, 1 / dx**2

    fom = opinf.models.ContinuousModel(
        operators=[
            opinf.operators.LinearOperator(A),
            opinf.operators.InputOperator(B),
        ]
    )

    initial_values = q_0(x)
    initial_values = (
        initial_values if not external_inputs else initial_values * u(0)
    )

    return t, fom.predict(initial_values, t, input_func=u, method="BDF")


def save_data_to_file(
    t: np.ndarray, Q: np.ndarray, filepath: str, overwrite=True
):
    with opinf.utils.hdf5_savehandle(filepath, overwrite=overwrite) as h5file:
        h5file.create_dataset("t", data=t)
        h5file.create_dataset("Q", data=Q)

    print(f"Training data saved to {filepath}")


if __name__ == "__main__":
    BASE_DIR = pathlib.Path(__file__).resolve().parent
    if len(sys.argv) < 2:
        # filepath
        path = BASE_DIR / "basic_training_data.h5"
    else:
        path = BASE_DIR / sys.argv[1]

    n_samples = 512
    n_timesteps = 401

    def q_0(x):
        return x * (1 - x)

    t, Q = generate_training_data(n_samples, n_timesteps, q_0)

    save_data_to_file(t, Q, str(path), overwrite=True)
