---
kernelspec:
  display_name: opinf
  language: python
  name: python3
file_format: mystnb
---

# `opinf.basis`

+++

```{eval-rst}
.. automodule:: opinf.basis

.. currentmodule:: opinf.basis

**Classes**

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   BasisTemplate
   BasisMulti
   LinearBasis
   PODBasis

**Functions**

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   pod_basis
   method_of_snapshots
   cumulative_energy
   residual_energy
   svdval_decay
```

+++

:::{admonition} Overview
:class: note

- `opinf.basis` classes define mappings between high-dimensional states and low-dimensional latent coordinates.
- The most common low-dimensional state approximation for Operator Inference is [Proper Orthogonal Decomposition](sec:api-basis-pod).
- [Multivariable data](sec:api-basis-multivar) can be treated jointly (monolithic) or variable by variable (multilithic).
:::

+++

:::{admonition} Example Data
:class: tip

Some examples on this page use data from a compressible flow problem for an ideal gas as described in {cite}`qian2020liftandlearn,guo2022bayesopinf`.

:::{dropdown} State Variables

The data consists of three variables recorded at $k = 200$ points in time:

- Velocity $v$,
- Pressure $p$, and
- Specific volume (inverse density) $\xi = 1/\rho$.

The spatial discretization results in $400$ degrees of freedom per variable, i.e., $n = 3 \times 400 = 1{,}200$.
:::

You can [download the data here](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/raw/data/basis_example.npy) to repeat the demonstration.
:::

:::
```{code-cell} ipython3
import warnings
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

import opinf

# Suppress some warning printouts.
warnings.showwarning = lambda msg, cat, *args: print(f"{cat.__name__}: {msg}")
opinf.utils.mpl_config()
```
:::

## Low-dimensional Approximations

+++

Reduced-order models enact a computational speedup by approximating the $n$-dimensional system state with $r \ll n$ degrees of freedom and working in the reduced state space.
The {mod}`opinf.basis` module provides tools for constructing low-dimensional state approximations.

+++

::::{admonition} Notation
:class: note

On this page,
- $\q \in \RR^n$ denotes the high-dimensional state vector to be reduced (the _full state_),
- $\qhat\in\RR^r$ is the low-dimensional, compressed version of $\q$ (the _reduced state_),
- $\breve{\q}\in\RR^n$ is the high-dimensional, decompressed version of $\qhat$ (the _reconstructed state_),
- $\boldsymbol{\Gamma}:\RR^r\to\RR^n$ is the decompression function defining the approximation of $\q$ with $\qhat$ (sometimes called the _decoder_), and
- $\boldsymbol{\Gamma}^{*}:\RR^n\to\RR^r$ is the compression function corresponding to $\boldsymbol{\Gamma}$ (sometimes called the _encoder_).

That is, $\qhat = \boldsymbol{\Gamma}^{*}(\q)$ and $\q \approx \breve{\q} = \boldsymbol{\Gamma}(\qhat)$.

:::{image} ../../_static/dimension-reduction.svg
:align: center
:width: 60 %
:::

We assume that the full state $\q$ has already been preprocessed, for instance with the tools described in {mod}`opinf.pre`.
::::

+++

For a given $\boldsymbol{\Gamma}$ and norm $\|\cdot\|$, the natural compression function $\boldsymbol{\Gamma}^{*}$ is the one that minimizes the approximation error induced by $\boldsymbol{\Gamma}$:

$$
\begin{aligned}
    \boldsymbol{\Gamma}^{*}(\q)
    = \underset{\qhat\in\RR^{r}}{\operatorname{argmin}}\left\|
        \q - \boldsymbol{\boldsymbol{\Gamma}}(\qhat)
    \right\|.
\end{aligned}
$$

::::{margin}
:::{admonition} Compression vs Projection
:class: warning

In the literature, the term _project_ is sometimes used to refer to applying the compression function $\boldsymbol{\Gamma}^{*}$.
In this package, _project_ always refers to compression followed by decompression.
:::
::::

In this case, $P = \boldsymbol{\Gamma} \circ \boldsymbol{\Gamma}^{*}:\RR^n\to\RR^n$ satisfies $P(P(\q)) = P(\q)$ for all $\q\in\RR^{n}$ and $P$ is called a _projection_.
The error between $\breve{\q} = P(\q)$ and $\q$ is called the _projection error_ at $\q$, and $\breve{\q}$ is, by definition, the closest point to $\q$ in $\operatorname{range}(\boldsymbol{\Gamma})$.

:::{dropdown} Proof
For any $\tilde{\q}\in\RR^r$, we have

$$
\begin{aligned}
    \boldsymbol{\Gamma}^{*}(\boldsymbol{\Gamma}(\tilde{\q}))
    = \underset{\qhat\in\RR^{r}}{\operatorname{argmin}}\left\|
        \boldsymbol{\Gamma}(\tilde{\q}) - \boldsymbol{\boldsymbol{\Gamma}}(\qhat)
    \right\|.
\end{aligned}
$$

By inspection, setting $\qhat = \tilde{\q}$ achieves the minimal value of zero.
Hence, $\boldsymbol{\Gamma}^{*}(\boldsymbol{\Gamma}(\tilde{\q})) = \tilde{\q}$, and we have

$$
\begin{aligned}
    \boldsymbol{\Gamma}(\boldsymbol{\Gamma}^{*}(\boldsymbol{\Gamma}(\boldsymbol{\Gamma}^{*}(\q))))
    = \boldsymbol{\Gamma}(\boldsymbol{\Gamma}^{*}(\q)).
\end{aligned}
$$
:::

+++

:::{admonition} Basis API
:class: important

`opinf.basis` classes are equipped with the following methods.
- [**`fit()`**](opinf.basis.BasisTemplate.fit) uses state data to learn appropriate decompression / compression functions.
- [**`compress()`**](opinf.basis.BasisTemplate.compress) applies the compression function $\boldsymbol{\Gamma}^{*}$.
- [**`decompress()`**](opinf.basis.BasisTemplate.decompress) applies the decompression function $\boldsymbol{\Gamma}$.
- [**`project()`**](opinf.basis.BasisTemplate.project) applies the composition $\boldsymbol{\Gamma}\circ\boldsymbol{\Gamma}^{*}$.
- [**`projection_error()`**](opinf.basis.BasisTemplate.projection_error) computes the relative or absolute error of the projection at a given state.
- [**`verify()`**](opinf.basis.BasisTemplate.verify) checks that composing `decompress()` with `compress()` defines a projection.
:::

+++

## Linear Bases

+++

A linear basis approximates approximates the full state $\q\in\RR^n$ as a linear combination of $r$ _basis vectors_ $\v_1,\ldots,\v_r\in\RR^n$.
Mathematically, this means the reconstructed state $\breve{\q}\in\RR^n$ is contained in the $r$-dimensional linear subspace $\operatorname{span}(\{\v_1,\ldots,\v_r\})\subset\RR^n$.
The reduced state vector $\qhat = [~\hat{q}_1~~\cdots~~\hat{q}_r~]\trp\in\RR^r$ consists of latent coordinate coefficients for each basis vector.

Gathering the basis vectors into the _basis matrix_ $\Vr = [~\v_1~~\cdots~~\v_r~]\in\RR^{n\times r}$, a linear basis approximation is given by

$$
\begin{aligned}
    \q
    \approx \Vr\qhat
    = \sum_{i=1}^{r}\hat{q}_i\v_i.
\end{aligned}
$$

If the basis vectors are orthonormal, i.e., $\Vr\trp\Vr = \I\in\RR^{r\times r}$, then the compression function is defined by the application of $\Vr\trp$.

::::{grid}
:gutter: 3
:margin: 2 2 0 0

:::{grid-item-card}
Compression
^^^
$\boldsymbol{\Gamma}^{*} : \q \mapsto \Vr\trp\q$
:::

:::{grid-item-card}
Decompression
^^^
$\boldsymbol{\Gamma} : \qhat \mapsto \Vr\qhat$
:::

:::{grid-item-card}
Projection
^^^
$\boldsymbol{\Gamma}\circ\boldsymbol{\Gamma}^{*} : \q \mapsto \Vr\Vr\trp\q$.
:::
::::

% This is a test.

:::{dropdown} Derivation of $~\boldsymbol{\Gamma}^{*}$
As stated above, the optimal compression function is defined by

$$
\begin{aligned}
    \boldsymbol{\Gamma}^{*}(\q)
    = \underset{\qhat\in\RR^{r}}{\operatorname{argmin}}\left\|
        \q - \boldsymbol{\boldsymbol{\Gamma}}(\qhat)
    \right\|_2
    = \underset{\qhat\in\RR^{r}}{\operatorname{arg min}}\left\|
        \q - \Vr\qhat
    \right\|_2^2.
\end{aligned}
$$

This is a linear least-squares problem; the solution is given by the normal equations

$$
\begin{aligned}
    \Vr\trp\Vr\qhat
    = \Vr\trp\q,
\end{aligned}
$$

which simplifies to $\qhat = \Vr\trp\q$ since $\Vr\trp\Vr = \I$.

In general, $\qhat = (\Vr\trp\Vr)^{-1}\Vr\q = \Vr^{\dagger}\q$, where $\Vr^{\dagger}$ is the [Moore--Penrose pseudoinverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) of $\Vr$.
:::

+++

### Known Basis Matrix

+++


The {class}`LinearBasis` class creates a linear basis from a given, orthonormal basis matrix.

Unlike the other classes in this module, {meth}`LinearBasis.compress` and {meth}`LinearBasis.decompress` can be used immediately after instantiation because the basis matrix is provided up front instead of learned from state data in {meth}`LinearBasis.fit`.
The basis matrix is accessible via the `entries` attribute or by indexing with `[:]`.

+++

:::{admonition} Non-orthonormal Bases
:class: tip

{class}`LinearBasis` raises a warning if initialized with a basis matrix $\Vr$ whose columns are not orthonormal.
To avoid this issue, a basis matrix can be orthonormalized (without changing the column span) through the [Gram--Schmidt process](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process) via, e.g., `scpiy.linalg.qr()`.

```python
>>> import scipy.linalg as la

>>> Vr_orth = la.qr(Vr, mode="economic")[0]
```
:::

+++

The following code uses a {class}`LinearBasis` for a trigonometric basis representing the functions $\sin(2\pi x)$, $\sin(4\pi x)$, ..., $\sin(2r\pi x)$ evaluated over a one-dimensional spatial domain.

```{code-cell} ipython3
# Choose full and reduced state dimensions.
n = 500
r = 8

# Define a spatial domain and evaluate the basis functions over that domain.
x = np.linspace(0, 1, n)
Vr = np.column_stack([np.sin(2 * k * np.pi * x) for k in range(1, r + 1)])

# Check to see if Vr has orthonormal columns.
np.round(Vr.T @ Vr, 4)
```

Because $\Vr\trp\Vr$ is diagonal but not the identify matrix, these basis vectors are orthogonal but do not have unit length.
Using this basis matrix to initialize a {class}`LinearBasis` results in a warning; additionally, calling {meth}`LinearBasis.verify` raises an error ot indicate that composing the decompression and compression functions does not define a projection.

```{code-cell} ipython3
trig_basis = opinf.basis.LinearBasis(Vr)

try:
    trig_basis.verify()
except Exception as ex:
    print(f"{type(ex).__name__}: {ex}")
```

To normalize the basis matrix, we divide each column vector by its norm.

```{code-cell} ipython3
# Normalize the columns of the basis matrix.
Vr_orth = Vr / la.norm(Vr, axis=0)
trig_basis_orth = opinf.basis.LinearBasis(Vr_orth)

# Print dimensions and check that the basis matrix is orthonormal.
print(trig_basis_orth)
trig_basis_orth.verify()

# Plot the first few basis vectors as functions of x.
trig_basis_orth.plot1D(x, num_vectors=4)
plt.show()
```

The next code block checks how well the function $f(x) = x(1 - x)e^x$ can be approximated by this basis.

```{code-cell} ipython3
# Represent a new function in the span of the basis.
f = x * (1 - x) * np.exp(x)
f_projected = trig_basis_orth.project(f)

# Compute the projection error for the new function.
error = trig_basis_orth.projection_error(f)
print(f"Relative projection error: {error:%}")

# Plot the original function and its projection with this basis.
plt.plot(x, f, label="$f(x)$")
plt.plot(x, f_projected, label="$f(x)$ projected")
plt.legend(loc="upper left")
plt.show()
```

The projection is a poor approximation for the original function because $f(x)$ lies well outside of the span of the basis functions $\sin(2\pi x),\ldots,\sin(2r\pi x)$.
A better option is a Fourier basis consisting of cosines as well as sines.

```{code-cell} ipython3
# Construct basis vectors with cosines and sines.
Vr_raw = np.column_stack(
    [np.cos(2 * k * np.pi * x) for k in range(r // 2)]
    + [np.sin(2 * k * np.pi * x) for k in range(1, (r // 2) + 1)]
)

# Orthonormalize the basis with the Gram-Schmidt process.
Vr_orth = la.qr(Vr_raw, mode="economic")[0]
fourier_basis = opinf.basis.LinearBasis(Vr_orth)

# Print dimensions and check that the basis matrix is orthonormal.
print(fourier_basis)
fourier_basis.verify()

# Calculate the projection of f(x) and its relative projection error.
f_projected = fourier_basis.project(f)
error = fourier_basis.projection_error(f)
print(f"Relative projection error: {error:%}")

# Plot the original function and its projection with this new basis.
plt.plot(x, f, label="$f(x)$")
plt.plot(x, f_projected, label="$f(x)$ projected")
plt.legend(loc="upper left")
plt.show()
```

The approximation with the Fourier basis is much better than with the sine basis, even when they have same number of basis vectors.
In general, using more basis vectors improves the approximation power of the basis and decreases projection error.

+++

(sec:api-basis-pod)=
### Proper Orthogonal Decomposition

+++

There is no one basis that works well for every problem.
The {class}`PODBasis` class uses [Proper Orthogonal Decomposition](https://en.wikipedia.org/wiki/Proper_orthogonal_decomposition) (POD), a close relative of [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) (PCA), to construct a linear basis that is tailored to a provided set of state data.
This is the most common basis used in Operator Inference applications.

+++

#### Definition

+++

In finite dimensions, POD is computed by taking the [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) (SVD) of a snapshot matrix $\Q = [~\q_0~~\cdots~~\q_{k-1}~]\in\RR^{n\times k}$.
If $\boldsymbol{\Phi\Sigma\Psi}\trp = \Q$ is the SVD, then the rank-$r$ POD basis is $\Vr = \boldsymbol{\Phi}_{:,:r}$, the matrix comprising the first $r$ left singular vectors.
Since the singular vectors are orthogonal, we always have $\Vr\trp\Vr = \I$.

```{code-cell} ipython3
# Load the example snapshot data.
snapshots = np.load("basis_example.npy")
snapshots.shape
```

```{code-cell} ipython3
# Extract the pressure variable from the snapshot data.
velocity, pressure, sp_volume = np.split(snapshots, 3, axis=0)

pressure.shape
```

```{code-cell} ipython3
# Construct a POD basis with a specified number of basis vectors.
r = 6
pressure_basis = opinf.basis.PODBasis(num_vectors=r, name="pressure")
pressure_basis.fit(pressure)
print(pressure_basis)

# Plot the first few vectors.
pressure_basis.plot1D()
plt.show()
```

(sec:api-basis-dimselect)=
#### Selecting the Reduced Dimension

+++


By default, {meth}`PODBasis.fit` stores all computed basis vectors, and the reduced state dimension $r$ can be changed on the fly with {meth}`PODBasis.set_dimension`.
In this case, we can increase the reduced state dimension up to $k = 200$.
However, we often don't need (or want) all $k$ basis vectors---as $r$ increases so does the computational cost of solving the eventual reduced-order model, and at some point adding additional vectors does not significantly increase the approximation power of the basis.

The singular values of the snapshot matrix $\Q$ can provide guidance for choosing an appropriate reduced state dimension.
There are four related criteria stemming from the singular values $\sigma_1 > \sigma_2 > \cdots > \sigma_k$.
{class}`PODBasis` has methods for visualizing each criteria, and {meth}`PODBasis.set_dimension` has arguments related to each.

```{eval-rst}
.. currentmodule:: opinf.basis.PODBasis
```

| Criterion            | Definition | {meth}`set_dimension` arg | Visualization |
| -------------------: | :--------- |:------------------------- | :------------ |
| Singular value decay | $\sigma_i \ge \xi$ for $i\le r$ | `svdval_threshold` $\xi$ | {meth}`plot_svdval_decay` |
| Cumulative energy    | $\frac{\sum_{i=1}^r\sigma_i^2}{\sum_{j=1}^k\sigma_j^2} \ge \kappa$ | `cumulative_energy` $\kappa$ | {meth}`plot_cumulative_energy` |
| Residual energy      | $\frac{\sum_{i=r+1}^k\sigma_i^2}{\sum_{j=1}^k\sigma_j^2} \le \epsilon$ | `residual_energy` $\epsilon$ | {meth}`plot_residual_energy` | 
| Projection error     | $\frac{\\|\Q - \Vr\Vr\trp\Q\\|_F}{\\|\Q\\|_F} \le \rho$ | `projection_error` $\rho$ | {meth}`plot_projection_error` |

```{eval-rst}
.. currentmodule:: opinf.basis
```

These singular value-based selection criteria are all related.
In particular, the squared projection error of the snapshot matrix is equal to the residual energy:

$$
\begin{aligned}
    \frac{\|\Q - \Vr\Vr\trp\Q\|_{F}^{2}}{\|\Q\|_{F}^{2}}
    = \displaystyle\frac{\sum_{j = r + 1}^{\ell}\sigma_{j}^{2}}{\sum_{j=1}^{\ell}\sigma_{j}^{2}}.
\end{aligned}
$$

:::{dropdown} Proof
Let $\Q = \boldsymbol{\Phi}\boldsymbol{\Sigma}\boldsymbol{\Psi}\trp$ be the thin singular value decomposition of $\Q\in\RR^{n \times k}$,
meaning $\boldsymbol{\Phi}\in\RR^{n\times \ell}$, $\boldsymbol{\Sigma}\in\RR^{\ell \times \ell}$, and $\boldsymbol{\Psi}\in\RR^{k\times \ell}$,
with $\boldsymbol{\Phi}\trp\boldsymbol{\Phi} = \boldsymbol{\Psi}\trp\boldsymbol{\Psi} = \I$.
Splitting the decomposition into "first $r$ modes" and "remaining modes" gives

$$
    \Q
    = \left[\begin{array}{cc}
        \boldsymbol{\Phi}_{r} & \boldsymbol{\Phi}_{\perp}
    \end{array}\right]
    \left[\begin{array}{cc}
        \boldsymbol{\Sigma}_{r} & \\
        & \boldsymbol{\Sigma}_{\perp}
    \end{array}\right]
    \left[\begin{array}{c}
        \boldsymbol{\Psi}_{r}\trp \\
        \boldsymbol{\Psi}_{\perp}\trp
    \end{array}\right]
    = \underbrace{\boldsymbol{\Phi}_{r}\boldsymbol{\Sigma}_{r}\boldsymbol{\Psi}_{r}\trp}_{\Q_{r}} + \underbrace{\boldsymbol{\Phi}_{\perp}\boldsymbol{\Sigma}_{\perp}\boldsymbol{\Psi}_{\perp}\trp}_{\Q_{\perp}},
$$

where

\begin{align*}
    &\boldsymbol{\Phi}_{r}\in\RR^{n\times r},
    &
    &\boldsymbol{\Phi}_{\perp}\in\RR^{n\times (\ell - r)},
    &
    &\boldsymbol{\Phi}_{r}\trp\boldsymbol{\Phi}_{r}
    = \boldsymbol{\Psi}_{r}\trp\boldsymbol{\Psi}_{r}
    = \I,
    \\
    &\boldsymbol{\Sigma}_{r}\in\RR^{r\times r},
    &
    &\boldsymbol{\Sigma}_{\perp}\in\RR^{(\ell - r)\times (\ell - r)},
    &
    &\boldsymbol{\Phi}_{\perp}\trp\boldsymbol{\Phi}_{\perp}
    = \boldsymbol{\Psi}_{\perp}\trp\boldsymbol{\Psi}_{\perp}
    = \I,
    \\
    &\boldsymbol{\Psi}_{r}\in\RR^{k\times r},
    &
    &\boldsymbol{\Psi}_{\perp}\in\RR^{k\times (\ell - r)},
    &
    &\boldsymbol{\Phi}_{r}\trp\boldsymbol{\Phi}_{\perp}
    = \boldsymbol{\Psi}_{r}\trp\boldsymbol{\Psi}_{\perp}
    = \mathbf{0}.
\end{align*}

We have defined $\Vr = \boldsymbol{\Phi}_{r}$.
Using $\Vr\trp\boldsymbol{\Phi}_{r} = \Vr\trp\Vr = \I$
and $\Vr\trp\boldsymbol{\Phi}_{\perp} = \boldsymbol{\Phi}_{r}\boldsymbol{\Phi}_{\perp} = \mathbf{0}$,

\begin{align*}
    \Vr\Vr\trp\Q_{r}
    &= \Vr\Vr\trp\boldsymbol{\Phi}_{r}\boldsymbol{\Sigma}_{r}\boldsymbol{\Psi}_{r}\trp
    = \Vr\boldsymbol{\Sigma}_{r}\boldsymbol{\Psi}_{r}\trp
    = \Q_{r},
    \\
    \Vr\Vr\trp\Q_{\perp}
    &= \Vr\Vr\trp\boldsymbol{\Phi}_{\perp}\boldsymbol{\Sigma}_{\perp}\boldsymbol{\Psi}_{\perp}\trp
    = \mathbf{0}.
\end{align*}

That is, $\Vr\Vr\trp\Q = \Vr\Vr\trp(\Q_{r} + \Q_{\perp}) = \Q_{r}$.
It follows that

$$
    \Q - \Vr\Vr\trp\Q
    = \Q_{r} + \Q_{\perp} - \Q_{r}
    = \Q_{\perp}
    = \boldsymbol{\Phi}_{\perp}\boldsymbol{\Sigma}_{\perp}\boldsymbol{\Psi}_{\perp}\trp.
$$

Since $\boldsymbol{\Phi}_{\perp}$ and $\boldsymbol{\Psi}_{\perp}$ have orthonormal columns,

$$
    \left\|\boldsymbol{\Phi}_{\perp}\boldsymbol{\Sigma}_{\perp}\boldsymbol{\Psi}_{\perp}\trp\right\|_{F}^{2}
    = \left\|\boldsymbol{\Sigma}_{\perp}\right\|_{F}^{2}
    = \sum_{j=r + 1}^{\ell}\sigma_{j}^{2}.
$$

Putting it all together,

$$
    \frac{\|\Q - \Vr\Vr\trp\Q\|_{F}^{2}}{\|\Q\|_{F}^{2}}
    = \frac{\|\boldsymbol{\Sigma}_{\perp}\|_{F}^{2}}{\|\boldsymbol{\Sigma}\|_{F}^{2}}
    = \frac{\sum_{j=r + 1}^{\ell}\sigma_{j}^{2}}{\sum_{j=1}^{\ell}\sigma_{j}^{2}}.
$$

:::

{meth}`PODBasis.plot_energy` visualizes several criteria together.

```{code-cell} ipython3
# Show the number of normalized singular values above a given threshold.
pressure_basis.plot_svdval_decay(threshold=1e-5)
plt.show()
```

```{code-cell} ipython3
# Set the reduced state dimension based on a singular value threshold.
pressure_basis.set_dimension(svdval_threshold=1e-5)
print(pressure_basis)

# Plot the singular value decay and the singular value energies.
pressure_basis.plot_energy(right=100)
plt.show()
```

The {func}`pod_basis` function computes a POD basis matrix and the corresponding singular values.
The functions {func}`svdval_decay`, {func}`cumulative_energy`, and {func}`residual_energy` visualize the dimension selection criteria.

+++

#### Centered POD

+++

{class}`PODBasis` uses a linear approximation $\q\approx\Vr\qhat$ with $\Vr$ derived from the SVD of a snapshot matrix $\Q = [~\q_0~~\cdots~~\q_{k-1}~]\in\RR^{n \times k}$.
It is often advantageous to first center the training data columnwise, in which case POD coincides with PCA.
This results in an _affine_ state approximation,

$$
\begin{aligned}
    \q \approx \Vr\qhat + \bar{\q},
    \qquad\qquad
    \bar{\q} = \frac{1}{k}\sum_{j=0}^{k-1}\q_j.
\end{aligned}
$$

To get this kind of approximation, center the snapshot matrix first with {class}`opinf.pre.ShiftScaleTransformer`, and remember to "uncenter" the results of `decompress()`.

```{code-cell} ipython3
shifter = opinf.pre.ShiftScaleTransformer(centering=True)
pressure_centered = shifter.fit_transform(pressure)

# Train a centered POD basis and plot the first few basis vectors.
pressure_pca = opinf.basis.PODBasis(num_vectors=r).fit(pressure_centered)

# To project the (uncentered) pressure, transform, project, then untransform.
pressure_projected = shifter.inverse_transform(
    pressure_pca.project(shifter.transform(pressure))
)

# Compute the projection error.
la.norm(pressure - pressure_projected) / la.norm(pressure)
```

```{code-cell} ipython3
# Plot the first few basis vectors.
pressure_pca.plot1D()
plt.show()
```

The principal basis vector of the previous, non-centered basis is almost constant when compared to the other basis vectors.
By contrast, the new, centered basis does not have any relatively constant basis vectors.

```{code-cell} ipython3
# Check the variance of the first basis vector in the non-centered case.
np.var(pressure_basis.entries[:, 0])
```

```{code-cell} ipython3
# Check the variance of each basis vector in the centered case.
np.var(pressure_pca.entries, axis=0).min()
```

Applying a centering can be interpreted more or less as removing the principal component of non-centered data.
In other words, [the principal component of non-centered data is often comparable to the mean of the data](https://en.wikipedia.org/wiki/Principal_component_analysis#Further_considerations).

+++

(sec:api-basis-multivar)=
## Multivariable Data

+++

For systems where the full state consists of several variables (pressure, velocity, temperature, etc.), there are two ways to approach a dimensionality reduction.
In this section we use $\q^{(i)}\in\RR^{n_i}$ to denote the $i$-th of $n_q$ state variables, $i = 0, \ldots, n_q - 1$.
For instance, $\q^{(0)}\in\RR^{100}$ could represent the velocity measured at $n_0 = 100$ spatial points and $\q^{(1)}\in\RR^{200}$ could be the pressure at $n_1 = 200$ spatial points.
In many application the state variables are all defined on the same spatial grid with $n_x$ nodes, in which case $n_0 = \cdots = n_{n_q-1} = n_x$.

+++

### Monolithic Reduction

+++

One option is to reduce all state variables jointly by considering the concatenated state

$$
\begin{aligned}
    \q = \left[\begin{array}{c}
        \q^{(0)} \\ \vdots \\ \q^{(n_q - 1)}
    \end{array}\right]
    \in \RR^{n},
\end{aligned}
$$

where $n = \sum_{i=0}^{n_q - 1} n_i$.
A low-dimensional approximation $\q \approx \boldsymbol{\Gamma}(\qhat)$ and a corresponding compression function $\boldsymbol{\Gamma}^{*}$ can be used to define a dimensionality reduction without reference to the sub-dimensions $n_i$.
With this approach, it is often very important to [preprocess](opinf.pre) the state variables so that their entries are of similar magnitude.
For instance, the following code creates a single {class}`PODBasis` for velocity, pressure, and specific volume jointly _without_ and preprocessing.

```{code-cell} ipython3
# Define a basis using the raw snapshot data for all three variables.
joint_pod = opinf.basis.PODBasis(
    residual_energy=1e-8,
    name="velocity-pressure-specificvolume",
)
joint_pod.fit(snapshots)

print(joint_pod)

# Plot the basis vectors.
joint_pod.plot1D()
plt.show()
```

This basis is highly suspicious: the left third of the plot above shows the portion of the basis vectors corresponding to the velocity, the middle third is for the pressure, and the final third is for the specific volume.
The pressure basis vector entries have a much stronger magnitude than the entries for the other two variables because the pressure data is of much larger magnitude than the other variables.
Furthermore, the relative projection error for the snapshot set as a whole is low, but it varies significantly across the variables.

```{code-cell} ipython3
def projection_error_report(snapshots_projected):
    """Compute the projection error for each state variable separately."""
    velocity_proj, pressure_proj, spvol_proj = np.split(snapshots_projected, 3)

    # Compute projection errors for the physical variables individually.
    v_error = la.norm(velocity_proj - velocity) / la.norm(velocity)
    p_error = la.norm(pressure_proj - pressure) / la.norm(pressure)
    s_error = la.norm(spvol_proj - sp_volume) / la.norm(sp_volume)
    joint_error = la.norm(snapshots_projected - snapshots) / la.norm(snapshots)

    # Report the errors.
    print(f"Relative projection error of velocity:\t\t{v_error:.4%}")
    print(f"Relative projection error of pressure:\t\t{p_error:.4%}")
    print(f"Relative projection error of specific volume:\t{s_error:.4%}")
    print(f"Relative projection error of joint variables:\t{joint_error:.4%}")
```

```{code-cell} ipython3
projection_error_report(joint_pod.project(snapshots))
```

To improve the basis, one option is to scale each variables to the range $[0, 1]$ using {class}`opinf.pre.TransformerMulti`.

```{code-cell} ipython3
# Initialize a different scaling for each physical variable.
euler_transformer = opinf.pre.TransformerMulti(
    [
        opinf.pre.ShiftScaleTransformer(
            scaling="minmax",
            name="velocity",
            verbose=True,
        ),
        opinf.pre.ShiftScaleTransformer(
            scaling="minmax",
            name="pressure",
            verbose=True,
        ),
        opinf.pre.ShiftScaleTransformer(
            scaling="minmax",
            name="specific volume",
            verbose=True,
        ),
    ]
)

# Learn and apply the transformation.
snapshots_scaled = euler_transformer.fit_transform(snapshots)
```

```{code-cell} ipython3
# Learn a new POD basis from the scaled data.
joint_pod.fit(snapshots_scaled)

print(joint_pod)

# Plot the basis vectors.
joint_pod.plot1D(num_vectors=7)
plt.show()
```

The transition zones between the three variables are still visible, but now the basis vector entries are all of the same order of magnitude.
The projection error across the variables is also much more consistent than before (remember to unscale the projection!).

```{code-cell} ipython3
projection_error_report(
    euler_transformer.inverse_transform(joint_pod.project(snapshots_scaled))
)
```

### Multilithic Reduction

+++

Instead of reducing multiple variables jointly, it is sometimes advantageous to define different reductions for each variable separately.
Mathematically, we define approximations $\q^{(i)} \approx \boldsymbol{\Gamma}_{i}(\qhat^{(i)})$ for $\qhat^{(i)}\in\RR^{r_i}$ for each $i = 0, \ldots, n_q - 1$, each with corresponding compression functions $\boldsymbol{\Gamma}_{i}^{*}:\RR^{r_i}\to\RR^{n_i}$.
The state approximation for the joint state $\q$ is then

$$
\begin{aligned}
    \left[\begin{array}{c}
        \q^{(0)} \\ \vdots \\ \q^{(n_q - 1)}
    \end{array}\right]
    = \q
    \approx 
    \boldsymbol{\Gamma}(\qhat)
    = \left[\begin{array}{c}
        \boldsymbol{\Gamma}_{0}(\qhat^{(0)})
        \\ \vdots \\ 
        \boldsymbol{\Gamma}_{n_q-1}(\qhat^{(n_q - 1)})
    \end{array}\right],
    \qquad
    \qhat
    = \left[\begin{array}{c}
        \qhat^{(0)} \\ \vdots \\ \qhat^{(n_q - 1)}
    \end{array}\right]
    \in \RR^{r},
\end{aligned}
$$

where $r = \sum_{i=0}^{n_q - 1}r_i$.
This type of reduction can be helpful for structure preservation because the vector $\qhat^{(i)}$ is the latent representation for $\q^{(i)}$, as opposed to the monolithic case in which each of the entries of $\qhat$ affect every $\q^{(0)},\ldots,\q^{(n_q-1)}$.

The {class}`BasisMulti` class joins individual bases to reduce multiple variables independently.
The bases need not have the same dimensions or even be of the same class.

```{code-cell} ipython3
euler_pod = opinf.basis.BasisMulti(
    [
        opinf.basis.PODBasis(residual_energy=1e-8, name="velocity"),
        opinf.basis.PODBasis(projection_error=1e-2, name="pressure"),
        opinf.basis.PODBasis(num_vectors=10, name="specific volume"),
    ],
)

euler_pod.fit(snapshots)
print(euler_pod)
```

```{code-cell} ipython3
projection_error_report(euler_pod.project(snapshots))
```

Note that we are working with the non-scaled data again.
Because the variables are treated separately, we can easily adjust the accuracy of the approximation.

```{code-cell} ipython3
# Use more basis vectors for the pressure.
euler_pod[1].set_dimension(projection_error=1e-4)
print(euler_pod[1], end="\n\n")

projection_error_report(euler_pod.project(snapshots))
```

:::{admonition} Block-diagonal Linear Basis
:class: tip

In multilithic reduction, if all of the state approximations are linear, then the overall state approximation is linear with a block-diagonal matrix $\Vr$.
Specifically, if $\q^{(i)} \approx \Vr^{(i)}\qhat^{(i)}$, then the approximation for the joint state is $\q \approx \Vr\qhat$ where

$$
\begin{aligned}
    \Vr
    = \operatorname{blockdiag}(\Vr^{(0)},\ldots,\Vr^{(n_q - 1)})
    = \left[\begin{array}{ccc}
        \Vr^{(0)} & & \\
        & \ddots & \\
        & & \Vr^{(n_q - 1)}
    \end{array}\right]
    \in \RR^{n \times r}.
\end{aligned}
$$
:::

+++

## Custom Bases

+++

New bases can be defined by inheriting from {class}`BasisTemplate`.
Once implemented, the [`verify()`](opinf.basis.BasisTemplate.verify) method may be used to test for consistency between the required methods.

```{code-cell} ipython3
class MyBasis(opinf.basis.BasisTemplate):
    """Custom basis for dimension reduction."""

    # Constructor -------------------------------------------------------------
    def __init__(self, hyperparameters, name=None):
        """Set any basis hyperparameters.
        If there are no hyperparameters, __init__() may be omitted.
        """
        super().__init__(name)
        # Process/store 'hyperparameters' here.

    # Required methods --------------------------------------------------------
    def fit(self, states):
        """Construct the basis."""
        # Set the state dimensions.
        self.full_state_dimension = NotImplemented
        self.reduced_state_dimension = NotImplemented

        # Learn the basis from `states` to enable compress() and decompress().
        raise NotImplementedError

    def compress(self, states):
        """Map high-dimensional states to low-dimensional latent
        coordinates.
        """
        raise NotImplementedError

    def decompress(self, states_compressed, locs=None):
        """Map low-dimensional latent coordinates to high-dimensional
        states.
        """
        raise NotImplementedError

    # Optional methods --------------------------------------------------------
    # These may be deleted if not implemented.
    def save(self, savefile, overwrite=False):
        """Save the basis to an HDF5 file."""
        return NotImplemented

    @classmethod
    def load(cls, loadfile):
        """Load a basis from an HDF5 file."""
        return NotImplemented
```

See {class}`BasisTemplate` for details on the arguments for each method.

+++

### Example: Canonical Projection

+++

The following custom basis reduces the full state vector $\q$ by selecting a limited number of entries.
This strategy is useful for impementing the discrete empirical interpolation method (DEIM).

```{code-cell} ipython3
class CanonicalBasis(opinf.basis.BasisTemplate):
    """Canonical projection basis."""

    def __init__(self, num_elements, name=None):
        """Set the number of locations to keep when the full state is
        compressed.
        """
        super().__init__(name)
        self.reduced_state_dimension = num_elements
        self.indices = None

    def fit(self, states):
        """Randomly select locations to keep when the full state is
        compressed.
        """
        self.full_state_dimension = states.shape[0]
        self.indices = np.sort(
            np.random.choice(
                self.full_state_dimension,
                size=self.reduced_state_dimension,
                replace=False,
            )
        )
        return self

    def compress(self, states):
        """Extract the selected elements from the full state."""
        if self.indices is None:
            raise AttributeError("indices not set, call fit()")
        return states[self.indices]

    def decompress(self, states_compressed, locs=None):
        """Populate a full state with the extracted elements."""
        if self.indices is None:
            raise AttributeError("indices not set, call fit()")

        shape = self.full_state_dimension
        if states_compressed.ndim == 2:
            shape = (self.full_state_dimension, states_compressed.shape[1])

        states = np.zeros(shape, dtype=states_compressed.dtype)
        states[self.indices] = states_compressed
        return states if locs is None else states[locs]

    def save(self, savefile, overwrite=False):
        """Save the basis to an HDF5 file."""
        with opinf.utils.hdf5_savehandle(savefile, overwrite) as hf:
            hf.create_dataset("r", data=[self.reduced_state_dimension])
            if self.indices is not None:
                hf.create_dataset("n", data=[self.full_state_dimension])
                hf.create_dataset("indices", data=self.indices)

    @classmethod
    def load(cls, loadfile):
        """Load a basis from an HDF5 file."""
        with opinf.utils.hdf5_loadhandle(loadfile) as hf:
            basis = cls(hf["r"][0])
            if "n" in hf:
                basis.full_state_dimension = hf["n"][0]
                basis.indices = hf["indices"][:]
            return basis
```

```{code-cell} ipython3
canon = CanonicalBasis(4, name="pressure").fit(pressure)

print(canon)
canon.verify()
```