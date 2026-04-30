---
kernelspec:
  display_name: opinf
  language: python
  name: python3
file_format: mystnb
---

# `opinf.pre`

+++

```{eval-rst}
.. automodule:: opinf.pre

.. currentmodule:: opinf.pre

**Classes**

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   TransformerTemplate
   NullTransformer
   ShiftTransformer
   ScaleTransformer
   ShiftScaleTransformer
   TransformerPipeline
   TransformerMulti

**Functions**

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   shift
   scale
```

+++

:::{admonition} Overview
:class: note

- Operator Inference performance often improves when the training data are standardized. Multivariable data in particular benefits from preprocessing.
- `opinf.pre` classes define invertible transformations for data standardization.
:::

+++

::::{admonition} Example Data
:class: tip

The examples on this page use data downsampled from the combustion problem described in {cite}`swischuk2020combustion`.

:::{dropdown} State Variables

The data consists of nine variables recorded at 100 points in time.

- Pressure $p$
- $x$-velocity $v_{x}$
- $y$-velocity $v_{y}$
- Temperature $T$
- Specific volume (inverse density) $\xi = 1/\rho$
- Chemical species molar concentrations for CH$_{4}$, O$_{2}$, CO$_{2}$, and H$_{2}$O.

The dimension of the spatial discretization in the full example in {cite}`swischuk2020combustion` is $n_x = 38{,}523$ for each of the $n_q = 9$ variables, so the total state dimension is $n_q n_x = 9 \times 38{,}523 = 346{,}707$.
For demonstration purposes, we have downsampled the state dimension to $n_x' = 535$, hence $n = n_q n_x' = 9 \times 535 = 4{,}815$ is the total state dimension of the example data.
:::

You can [download the data here](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/raw/data/pre_example.npy) to repeat the experiments.
The full dataset is available [here](https://doi.org/10.7302/nj7w-j319).
::::

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

import opinf

opinf.utils.mpl_config()
```

## Preprocessing Data

+++

Raw dynamical systems data often need to be lightly preprocessed before use in Operator Inference.
This module includes tools for centering/shifting and scaling/nondimensionalization of snapshot data after lifting (when applicable) and prior to dimensionality reduction.

+++

:::{admonition} Notation
:class: note

On this page,
- $\q \in \RR^n$ denotes the unprocessed state variable for which we have $k$ snapshots $\q_0,\ldots,\q_{k-1}\in\RR^n$,
- $\q'\in\RR^n$ denotes state variable after being shifted (centered), and
- $\q''\in\RR^n$ denotes the state variable after being shifted _and_ scaled (non-dimensionalized).

The tools demonstrated here define a mapping $\mathcal{T}:\RR^n\to\RR^n$ with $\q'' = \mathcal{T}(\q)$.
:::

+++

:::{admonition} Lifting and Preprocessing
:class: note

A [lifting map](opinf.lift) can be viewed as a type of preprocessing map, $\mathcal{L}:\RR^{n_1}\to\RR^{n_2}$.
However, the preprocessing transformations defined in this module map from a vector space back to itself ($n_1 = n_2$) while lifting maps may augment the state with additional variables ($n_2 \ge n_1$).
:::

+++

::::{admonition} Fit-and-Transform versus Transform
:class: important

Pre-processing transformation classes are calibrated through user-provided hyperparameters in the constructor and/or training snapshots passed to ``fit()`` or ``fit_transform()``.
The ``transform()`` method applies but *does not alter* the transformation.
Some transformations are designed so that the transformed training data has certain properties, but those properties are not guaranteed to hold for transformed data that was not used for training.

:::{dropdown} Example

Consider a set of training snapshots $\{\q_{j}\}_{j=0}^{k-1}\subset\RR^n$.
The {class}`ShiftScaleTransformer` can shift data by the mean training snapshot, meaning it can represent the transformation $\mathcal{T}:\RR^{n}\to\RR^{n}$ given by

$$
\begin{aligned}
    \mathcal{T}(\q) = \q - \bar{\q},
    \qquad
    \bar{\q} = \frac{1}{k}\sum_{j=0}^{k-1}\q_{j}.
\end{aligned}
$$

The key property of this transformation is that the transformed training snapshots have zero mean.
That is,

$$
\begin{aligned}
    \frac{1}{k}\sum_{j=0}^{k-1}\mathcal{T}(\q_j)
    = \frac{1}{k}\sum_{j=0}^{k-1}(\q_j - \bar{\q})
    = \frac{1}{k}\sum_{j=0}^{k-1}\q_j - \frac{1}{k}\sum_{j=0}^{k-1}\bar{\q}
    = \bar{\q} - \frac{k}{k}\bar{\q}
    = \0.
\end{aligned}
$$

However, for any other collection $\{\mathbf{x}_j\}_{j=0}^{k'-1}\subset\RR^{n}$ of snapshots, the set of transformed snapshots $\{\mathcal{T}(\mathbf{x}_j)\}_{j=0}^{k'-1}$ is not guaranteed to have zero mean because $\mathcal{T}$ shifts by the mean of the $\q_j$'s, not the mean of the $\mathbf{x}_j$'s.
That is,

$$
\begin{aligned}
    \frac{1}{k'}\sum_{j=0}^{k'-1}\mathcal{T}(\mathbf{x}_j)
    = \frac{1}{k'}\sum_{j=0}^{k'-1}(\mathbf{x}_j - \bar{\q})
    \neq \0.
\end{aligned}
$$
:::
::::

+++

## Shifting / Centering

+++

A common first preprocessing step is to shift the training snapshots by some reference snapshot $\bar{\q}\in\RR^n$, i.e.,

$$
    \q' = \q - \bar{\q}.
$$

The {class}`ShiftTransformer` receives a reference snapshot $\bar{\q}$ and applies this transformation.
This is useful for scenarios where a specific $\bar{\q}$ can result in desirable properties in the shifted data, such as homogeneous boundary conditions.

```{code-cell} ipython3
# Load the example snapshot data.
snapshots = np.load("pre_example.npy")

snapshots.shape
```

```{code-cell} ipython3
# Extract the pressure variable from the snapshot data.
pressure = np.split(snapshots, 9, axis=0)[0]

# Initialize a ShiftTransformer for shifting the pressure so that
# each row has a minimum of 0.
pressure_shifter = opinf.pre.ShiftTransformer(
    pressure.min(axis=1),
    name="pressure",
)
print(pressure_shifter)
```

```{code-cell} ipython3
pressure_shifted = pressure_shifter.fit_transform(pressure)
pressure_shifted.shape
```

```{code-cell} ipython3
print(f"minimum pressure before shift: {pressure.min():.2e}")
print(f"minimum pressure after shift:  {pressure_shifted.min():.2e}")
```

One strategy that is often effective for Operator Inference is to set the reference snapshot to be the average of the training snapshots:

$$
    \bar{\q}
    := \frac{1}{k}\sum_{j=0}^{k-1}\q_{j}.
$$

In this case, the transformed snapshots $\q_j' = \q_j - \bar{\q}$ are centered around $\0$.
This type of transformation can be accomplished using a {class}`ShiftScaleTransformer` with `centering=True`.

```{code-cell} ipython3
# Initialize a ShiftScaleTransformer for centering the pressure.
pressure_transformer = opinf.pre.ShiftScaleTransformer(
    centering=True,
    name="pressure",
    verbose=True,
)
print(pressure_transformer)
```

```{code-cell} ipython3
# Shift the pressure snapshots by the average pressure snapshot.
pressure_shifted = pressure_transformer.fit_transform(pressure)
```

```{code-cell} ipython3
# Plot the distribution of the entries of the raw and processed states.
fig, axes = plt.subplots(1, 2, sharey=True)
axes[0].hist(pressure.flatten(), bins=40)
axes[1].hist(pressure_shifted.flatten(), bins=40)

axes[0].set_ylabel("Frequency")
axes[0].set_xlabel("Pressure")
axes[1].set_xlabel("Shifted pressure")

fig.tight_layout()
plt.show()
```

::::{admonition} Shifting Affects Model Form
:class: important

Introducing a shift can cause a structural change in the governing dynamics.
When shifting state variables, the structure of a reduced-order model should be determined based on the dynamics of the shifted variable, not the original variable.

:::{dropdown} Example 1: Linear System

Consider the linear system

$$
\begin{align*}
    \ddt\q(t) = \A\q(t).
\end{align*}
$$

The dynamics of the shifted variable $\q'(t) = \q(t) - \bar{\q}$ are given by

$$
\begin{align*}
    \ddt\q'(t)
    = \ddt[\q(t) - \bar{\q}]
    = \ddt\q(t)
    = \A\q(t)
    = \A[\bar{\q} + \q'(t)]
    = \A\bar{\q} + \A\q'(t),
\end{align*}
$$

which has a new constant term $\A\bar{\q}$ in addition to a linear term $\A\q'(t)$.
If the variable $\q$ is used for Operator Inference, the reduced-order model should take on the linear form $\ddt\qhat(t) = \Ahat\qhat(t)$, while if $\q'$ is the state variable, the reduced-order model should be $\ddt\qhat(t) = \chat + \Ahat\qhat(t)$.
:::

:::{dropdown} Example 2: Quadratic System

Consider the purely quadratic system

$$
\begin{align*}
    \ddt\q(t) = \H[\q(t)\otimes\q(t)],
\end{align*}
$$

where $\otimes$ denotes the [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product).
An appropriate reduced-order model for this system is also quadratic, $\ddt\qhat(t) = \Hhat[\qhat(t)\otimes\qhat(t)]$.
However, the dynamics of the shifted variable $\q'(t) = \q(t) - \bar{\q}$ includes lower-order terms:

$$
\begin{align*}
    \ddt\q'(t)
    &= \ddt[\q(t) - \bar{\q}]
    \\
    &= \H[\q(t)\otimes\q(t)]
    \\
    &= \H[(\bar{\q} + \q'(t))\otimes(\bar{\q} + \q'(t))]
    \\
    &= \H[\bar{\q}\otimes\bar{\q}]
    + \H[\bar{\q}\otimes\q'(t)] + \H[\q'(t)\otimes\bar{\q}]
    + \H[\q'(t)\otimes\q'(t)].
\end{align*}
$$

The terms $\H[\bar{\q}\otimes\q'(t)] + \H[\q'(t)\otimes\bar{\q}]$ can be interpreted as a linear transformation of $\q'(t)$, hence an appropriate reduced-order model for $\q'(t)$ has the fully quadratic form $\ddt\qhat(t) = \chat + \Ahat\qhat(t) + \Hhat[\qhat(t)\otimes\qhat(t)]$.
:::
::::

+++

## Scaling / Non-dimensionalization

+++

Many engineering problems feature multiple variables with ranges across different scales.
For such cases, it is often beneficial to scale the variables to similar ranges so that one variable does not overwhelm the other during operator learning.
In other words, training data should be nondimensionalized when possible.

A scaling operation for a single variable is given by

$$
    \q'' = \alpha\q',
$$

where $\alpha \neq 0$ and $\q'$ is a training snapshot after shifting (when desired).
The {class}`ScaleTransformer` class receives a scaler $\alpha$ and implements this transformation.

```{code-cell} ipython3
# Initialize a ScaleTransformer for scaling the pressure to [0, 1].
pressure_scaler = opinf.pre.ScaleTransformer(
    1 / pressure.max(), name="pressure"
)

print(pressure_scaler)
```

```{code-cell} ipython3
# Apply the scaling.
pressure_scaled = pressure_scaler.fit_transform(pressure)
pressure_scaled.shape
```

```{code-cell} ipython3
print(f"min pressure before scaling: {pressure.min():.2e}")
print(f"max pressure before scaling: {pressure.max():.2e}")
print(f"min pressure after scaling:  {pressure_scaled.min():.2e}")
print(f"max pressure after scaling:  {pressure_scaled.max():.2e}")
```

The entries of the state can be scaled individually by passing a vector to {class}`ScaleTransformer`.

```{code-cell} ipython3
# Scale the pressure so the maximum of each row is 1.
pressure_scaler = opinf.pre.ScaleTransformer(
    1 / pressure.max(axis=1), name="pressure"
)

print(pressure_scaler)
```

```{code-cell} ipython3
# Apply the scaling.
pressure_scaled2 = pressure_scaler.fit_transform(pressure)
pressure_scaled2.shape
```

```{code-cell} ipython3
print(
    "number of rows whose maximum is 1 (whole scaling): "
    f"{np.count_nonzero(np.isclose(pressure_scaled.max(axis=1), 1))}"
)
print(
    "number of rows whose maximum is 1 (row scaling): "
    f"{np.count_nonzero(np.isclose(pressure_scaled2.max(axis=1), 1))}"
)
```

The {class}`ShiftScaleTransformer` class implements several types of scalings that are calibrated from data.
For example, setting `scaling="maxabs"` scales the training data by the inverse of its absolute maximum entry so that the resulting data lies in the interval $[-1, 1]$.

```{code-cell} ipython3
# Extract the velocity in the x direction.
xvelocity = np.split(snapshots, 9, axis=0)[1]

# Initialize a ShiftScaleTransformer for scaling the velocity to [-1, 1].
xvelocity_scaler = opinf.pre.ShiftScaleTransformer(
    centering=False,
    scaling="maxabs",
    name="x velocity",
)

print(xvelocity_scaler)
```

```{code-cell} ipython3
# Apply the scaling.
xvelocity_scaled = xvelocity_scaler.fit_transform(xvelocity)
xvelocity_scaled.shape
```

```{code-cell} ipython3
print(f"min x-velocity before scaling: {xvelocity.min():.2e}")
print(f"max x-velocity before scaling: {xvelocity.max():.2e}")
print(f"min x-velocity after scaling:  {xvelocity_scaled.min():.2e}")
print(f"max x-velocity after scaling:  {xvelocity_scaled.max():.2e}")
```

The {class}`ShiftScaleTransformer` class can perform a mean-centering shift, followed by a data-driven scaling.
To link a custom shift with a custom scaling, instantiate a {class}`ShiftTransformer` and a {class}`ScaleTransformer` and pass them to a {class}`TransformerPipeline`.

```{code-cell} ipython3
# Combine the shift to zero from before with a custom scaling.
pressure_scaler = opinf.pre.ScaleTransformer(1e-6, "pressure")
pressure_transformer2 = opinf.pre.TransformerPipeline(
    [pressure_shifter, pressure_scaler], name="pressure"
)

print(pressure_transformer2)
```

```{code-cell} ipython3
# Apply the scaling.
pressure_transformed = pressure_transformer2.fit_transform(pressure)
pressure_transformed.shape
```

```{code-cell} ipython3
print(f"min pressure before shifting/scaling: {pressure.min():.2e}")
print(f"max pressure before shifting/scaling: {pressure.max():.2e}")
print(f"min pressure after shifting/scaling: {pressure_transformed.min():.2e}")
print(f"max pressure after shifting/scaling: {pressure_transformed.max():.2e}")
```

:::{admonition} No Free Lunch
:class: note

Choosing an advantageous preprocessing strategy is highly problem dependent, and the tools in this module are not the only ways to preprocess snapshot data.
See, for example, {cite}`issan2023shifted` for a compelling application of Operator Inference to solar wind streams in which preprocessing plays a vital role.
:::

+++

## Multivariable Data

+++

For systems where the full state consists of several variables (pressure, velocity, temperature, etc.), it may not be appropriate to apply the same scaling to each variable.
The {class}`TransformerMulti` class joins individual transformers together to handle multi-state data.

+++

Below, we construct the following transformation for the nine state variables.
- Pressure: center, then scale to $[-1, 1]$.
- $x$-velocity: Scale to $[-1, 1]$.
- $y$-velocity: Scale to $[-1, 1]$.
- Temperature: center, then scale to $[-1, 1]$.
- Specific volume: scale to $[0, 1]$.
- Chemical species: scale to $[0, 1]$.

```{code-cell} ipython3
combustion_transformer = opinf.pre.TransformerMulti(
    transformers=[
        opinf.pre.ShiftScaleTransformer(
            name="pressure", centering=True, scaling="maxabs", verbose=True
        ),
        opinf.pre.ShiftScaleTransformer(
            name="x-velocity", scaling="maxabs", verbose=True
        ),
        opinf.pre.ShiftScaleTransformer(
            name="y-velocity", scaling="maxabs", verbose=True
        ),
        opinf.pre.ShiftScaleTransformer(
            name="temperature", centering=True, scaling="maxabs", verbose=True
        ),
        opinf.pre.ShiftScaleTransformer(
            name="specific volume", scaling="minmax", verbose=True
        ),
        opinf.pre.ShiftScaleTransformer(
            name="methane", scaling="minmax", verbose=True
        ),
        opinf.pre.ShiftScaleTransformer(
            name="oxygen", scaling="minmax", verbose=True
        ),
        opinf.pre.ShiftScaleTransformer(
            name="carbon dioxide", scaling="minmax", verbose=True
        ),
        opinf.pre.ShiftScaleTransformer(
            name="water", scaling="minmax", verbose=True
        ),
    ]
)

snapshots_preprocessed = combustion_transformer.fit_transform(snapshots)
```

```{code-cell} ipython3
print(combustion_transformer)
```

```{code-cell} ipython3
# Extract a single variable from the processed snapshots.
oxygen_processed = combustion_transformer.get_var(
    "oxygen",
    snapshots_preprocessed,
)

oxygen_processed.shape
```

## Custom Transformers

+++

New transformers can be defined by inheriting from the {class}`TransformerTemplate`.
Once implemented, the [`verify()`](TransformerTemplate.verify) method may be used to test for consistency between the required methods.

```{code-cell} ipython3
class MyTransformer(opinf.pre.TransformerTemplate):
    """Custom pre-processing transformation."""

    def __init__(self, hyperparameters, name=None):
        """Set any transformation hyperparameters.
        If there are no hyperparameters, __init__() may be omitted.
        """
        super().__init__(name)
        # Process/store 'hyperparameters' here.

    # Required methods --------------------------------------------------------
    def fit_transform(self, states, inplace=False):
        """Learn and apply the transformation."""
        # Set self.state_dimension in this method, e.g.,
        self.state_dimension = len(states)
        raise NotImplementedError

    def transform(self, states, inplace=False):
        """Apply the learned transformation."""
        raise NotImplementedError

    def inverse_transform(self, states_transformed, inplace=False, locs=None):
        """Apply the inverse of the learned transformation."""
        raise NotImplementedError

    # Optional methods --------------------------------------------------------
    # These may be deleted if not implemented.
    def transform_ddts(self, ddts, inplace=False):
        """Apply the learned transformation to snapshot time derivatives."""
        return NotImplemented

    def save(self, savefile, overwrite=False):
        """Save the transformer to an HDF5 file."""
        return NotImplemented

    @classmethod
    def load(cls, loadfile):
        """Load a transformer from an HDF5 file."""
        return NotImplemented
```

See the {class}`TransformerTemplate` page for details on the arguments for each method.

+++

### Example: Hadamard Scaling

+++

The following class implements the transformation $\mathcal{T}(\q) = \q \ast \w$ where $\ast$ is the Hadamard (elementwise) product and $\s\in\RR^{n}$ is a given vector with all nonzero entries.
The inverse of this transform is $\mathcal{T}^{-1}(\q) = \q \ast \w'$ where the entries of $\w'\in\RR^{n}$ are the inverse of the entries of $\w$.
This transformation is equivalent to {class}`ScaleTransformer` with `scaler` set to $\w$ and can be interpreted as applying a diagonal weighting matrix to the state snapshots.

```{code-cell} ipython3
class HadamardTransformer(opinf.pre.TransformerTemplate):
    """Hadamard product transformer (weighting)."""

    def __init__(self, w, name=None):
        """Set the product vector."""
        super().__init__(name)
        self.w = w
        self.winv = 1 / w

    # Required methods --------------------------------------------------------
    def fit_transform(self, states, inplace=False):
        """Learn and apply the transformation."""
        self.state_dimension = self.w.size
        return self.transform(states, inplace=inplace)

    def transform(self, states, inplace=False):
        """Apply the learned transformation."""
        out = states if inplace else np.empty_like(states)
        w = self.w
        if states.ndim == 2:
            w = w.reshape((self.state_dimension, 1))
        out[:] = states * w
        return out

    def inverse_transform(self, states_transformed, inplace=False, locs=None):
        """Apply the inverse of the learned transformation."""
        winv = self.winv
        if locs is not None:
            winv = winv[locs]
        if states_transformed.ndim == 2:
            winv = winv.reshape((-1, 1))
        states = (
            states_transformed
            if inplace
            else np.empty_like(states_transformed)
        )
        states[:] = states_transformed * winv
        return states

    def transform_ddts(self, ddts, inplace=False):
        """Apply the learned transformation to snapshot time derivatives."""
        return self.transform(ddts, inplace=inplace)

    def save(self, savefile, overwrite=False):
        """Save the transformer to an HDF5 file."""
        with opinf.utils.hdf5_savehandle(savefile, overwrite) as hf:
            hf.create_dataset("w", data=self.w)
            if self.name is not None:
                meta = hf.create_dataset("meta", shape=(0,))
                meta.attrs["name"] = self.name

    @classmethod
    def load(cls, loadfile):
        """Load a transformer from an HDF5 file."""
        name = None
        with opinf.utils.hdf5_loadhandle(loadfile) as hf:
            w = hf["w"][:]
            if "meta" in hf:
                name = str(hf["meta"].attrs["name"])
        return cls(w, name=name)
```

```{code-cell} ipython3
w = np.random.uniform(size=pressure.shape[0])
ht = HadamardTransformer(w, name="Pressure weighter")
pressure_weighted = ht.fit_transform(pressure)
```

```{code-cell} ipython3
ht.verify()
```

:::{admonition} Developer Notes
:class: note

- In this example, the `state_dimension` could be set in the constructor because the `w` argument is a vector of length $n$. However, the `state_dimension` is not required to be set until [`fit_transform()`](TransformerTemplate.fit_transform).
- Because the transformation is dictated by the choice of `w` and not calibrated from data, [`fit_transform()`](TransformerTemplate.fit_transform) simply calls [`transform()`](TransformerTemplate.transform).
- When `locs` is provided in [`inverse_transform()`](TransformerTemplate.inverse_transform), it is assumed that the `states_transformed` are the elements of the state vector at the given locations. That is,`inverse_transform(transform(states)[locs], locs) == states[locs]`.
:::