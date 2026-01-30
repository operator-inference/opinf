from collections.abc import Callable
import pathlib
import sys
import h5py

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


def generate_basics_data(filepath: str = "basics_data.h5"):
    # set up basic parameters
    n_samples = 512
    n_timesteps = 401

    # define the various initial conditions used in the tutorial
    def q_0_default(x):
        return x * (1 - x)

    initial_condition_funcs = [
        lambda x: 10 * x * (1 - x),
        lambda x: 5 * x**2 * (1 - x) ** 2,
        lambda x: 50 * x**4 * (1 - x) ** 4,
        lambda x: 0.5 * np.sqrt(x * (1 - x)),
        lambda x: 0.25 * np.sqrt(np.sqrt(x * (1 - x))),
        lambda x: np.sin(np.pi * x) / 3 + np.sin(5 * np.pi * x) / 5,
    ]

    # initialize the file we will write the data to
    f = h5py.File(filepath, "w")

    # generate and save data for the default initial condition
    # also save the data for the time dimension
    t, Q_default = generate_training_data(n_samples, n_timesteps, q_0_default)
    f.create_dataset("t", data=t)
    f.create_dataset("default", data=Q_default)

    for idx, func in enumerate(initial_condition_funcs):
        # for each initial condition,
        # generate the data for that condition
        # and save it as a new dataset
        _, Q = generate_training_data(n_samples, n_timesteps, func)
        f.create_dataset(str(idx), data=Q)

    f.close()

    print(f"Training data saved to {filepath}")


def generate_external_inputs_data(filepath: str = "inputs_data.h5"):
    n_samples = 512
    n_timesteps = 1000

    alpha = 100

    # the part of the initial condition independent of u(t)
    def q_0(x):
        return np.exp(alpha * (x - 1)) + np.exp(-alpha * x) - np.exp(-alpha)

    # the define the external inputs functions
    def u(t):
        return np.ones_like(t) + np.sin(4 * np.pi * t) / 4

    train_inputs = [
        lambda t: np.exp(-t),
        lambda t: 1 + t**2 / 2,
        lambda t: 1 - np.sin(np.pi * t) / 2,
    ]
    test_inputs = [
        lambda t: 1 - np.sin(3 * np.pi * t) / 3,
        lambda t: 1 + 25 * (t * (t - 1)) ** 3,
        lambda t: 1 + np.exp(-2 * t) * np.sin(np.pi * t),
    ]

    # initialize the h5 file to write to
    f = h5py.File(filepath, "w")

    # generate the default training data (for the first part of the tutorial)
    t, Q = generate_training_data(n_samples, n_timesteps, q_0, u)
    U = u(t)

    f.create_dataset("t", data=t)
    f.create_dataset("Q", data=Q)
    f.create_dataset("U", data=U)

    train_grp = f.create_group("train")
    test_grp = f.create_group("test")
    for idx, [train_input, test_input] in enumerate(
        zip(train_inputs, test_inputs)
    ):
        _, Q_train = generate_training_data(
            n_samples, n_timesteps, q_0, train_input
        )
        U_train = train_input(t)
        _, Q_test = generate_training_data(
            n_samples, n_timesteps, q_0, test_input
        )
        U_test = test_input(t)

        train_grp.create_dataset(f"Q_{idx}", data=Q_train)
        train_grp.create_dataset(f"U_{idx}", data=U_train)
        test_grp.create_dataset(f"Q_{idx}", data=Q_test)
        test_grp.create_dataset(f"U_{idx}", data=U_test)

    f.close()
    print(f"Training data saved to {filepath}")


if __name__ == "__main__":
    BASE_DIR = pathlib.Path(__file__).resolve().parent
    data_to_generate = None

    # TODO: write --help flag for this script
    if len(sys.argv) < 2:
        data_to_generate = "all"
    else:
        if sys.argv[1] in ["basics", "inputs", "parametric", "all"]:
            data_to_generate = sys.argv[1]
        else:
            raise ValueError(
                "Data to generate must be one of the following: "
                "'basics', 'inputs', 'parametric', or 'all'."
            )

    if data_to_generate == "basics" or data_to_generate == "all":
        generate_basics_data(
            str(BASE_DIR / "source" / "tutorials" / "basics_data.h5")
        )
    elif data_to_generate == "inputs" or data_to_generate == "all":
        generate_external_inputs_data(
            str(BASE_DIR / "source" / "tutorials" / "inputs_data.h5")
        )
    elif data_to_generate == "parametric" or data_to_generate == "all":
        raise NotImplementedError(
            "Parametric data generation has not yet been implemented!"
        )
