---
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
file_format: mystnb
---

# `opinf.ddt`

+++

```{eval-rst}
.. automodule:: opinf.ddt

.. currentmodule:: opinf.ddt

**Classes**

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   DerivativeEstimatorTemplate
   UniformFiniteDifferencer
   NonuniformFiniteDifferencer
   InterpDerivativeEstimator

**Finite Difference Schemes for Uniformly Spaced Data**

*Forward Differences*

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   fwd1
   fwd2
   fwd3
   fwd4
   fwd5
   fwd6

*Backward Differences*

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   bwd1
   bwd2
   bwd3
   bwd4
   bwd5
   bwd6

*Central Differences*

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   ctr2
   ctr4
   ctr6

*Mixed Differences*

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   ord2
   ord4
   ord6
   ddt_uniform

**Finite Difference Schemes for Nonuniformly Spaced Data**

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   ddt_nonuniform
   ddt
```

+++

:::{admonition} Overview
:class: note

- Operator Inference for [continuous models](opinf.models.ContinuousModel) requires time derivatives of the state snapshots.
- `opinf.ddt` provides tools for estimating the time derivatives of the state snapshots.
- Finite difference approximations are available through {class}`UniformFiniteDifferencer` and {class}`NonuniformFiniteDifferencer`.
:::

```{code-cell} ipython3
import opinf
import numpy as np
import scipy.interpolate as spinterpolate

opinf.utils.mpl_config()
```

## Time Derivative Estimation

+++

To calibrate time-continuous models, Operator Inference requires the time derivative of the state snapshots.
For example, consider the LTI system

$$
\begin{aligned}
    \ddt\qhat(t)
    = \Ahat\qhat(t) + \Bhat\u(t).
\end{aligned}
$$ (eq:ddt:lti-reduced)

Here, $\qhat(t)\in\RR^{r}$ is the time-dependent ([reduced-order](./basis.md)) state and $\u(t)\in\RR^{m}$ is the time-dependent input.
In order to learn $\Ahat$ and $\Bhat$, Operator Inference solves a regression problem of the form

$$
\begin{aligned}
    \min_{\Ahat,\Bhat}\sum_{j=0}^{k-1}\left\|
    \Ahat\qhat_j + \Bhat\u_j
    - \dot{\qhat}_j
    \right\|_2^2
\end{aligned}
$$

[or similar](opinf.lstsq), where each triplet $(\qhat_j, \dot{\qhat}_j, \u_j)$ should correspond to the solution of {eq}`eq:ddt:lti-reduced` at some time $t_j$, $j = 0, \ldots, k - 1$.
In particular, we want

$$
\begin{aligned}
    \dot{\qhat}_j
    \approx \ddt\qhat(t)\big|_{t = t_j}
    = \Ahat\qhat_j + \Bhat\u_j.
\end{aligned}
$$

This module provides tools for estimating the snapshot time derivatives $\dot{\qhat}_0,\ldots,\dot{\qhat}_{k-1}\in\RR^{r}$ from the reduced snapshots $\qhat_0,\ldots,\qhat_{k-1}\in\RR^{r}$.

+++

:::{admonition} Preprocessing and Time Derivatives
:class: warning

In some cases, a full-order model may provide snapshot time derivatives $\dot{\q}_0,\ldots,\dot{\q}_{k-1}\in\RR^{n}$ in addition to state snapshots $\q_0,\ldots,\q_{k-1}\in\RR^{n}$.
If any lifting or preprocessing steps are used on the state snapshots, be careful to use the appropriate transformation for snapshot time derivatives, which may be different than the transformation used on the snapshots themselves.

For example, consider the affine state approximation $\q(t) \approx \Vr\qhat(t) + \bar{\q}$ with an orthonormal basis matrix $\Vr\in\RR^{n\times r}$ and a fixed vector $\bar{\q}\in\RR^{n}$.
In this case,

$$
\begin{aligned}
    \ddt\q(t)
    \approx \ddt\left[\Vr\qhat(t) + \bar{\q}\right]
    = \Vr\ddt\left[\qhat(t)\right].
\end{aligned}
$$

Hence, while the compressed state snapshots are given by $\qhat_j = \Vr\trp(\q_j - \bar{\q})$, the correct compressed snapshot time derivatives are $\dot{\qhat}_j = \Vr\trp\dot{\q}_j$ (without the $\bar{\q}$ shift).

See {meth}`opinf.lift.LifterTemplate.lift_ddts()` and {meth}`opinf.pre.TransformerTemplate.transform_ddts()`.
:::

+++

## Finite Difference Estimators

+++

### Partial Estimation

+++

Every finite difference scheme has limitations on where the derivative can be estimated.
For example, a [first-order backward scheme](opinf.ddt.bwd1) requires $\qhat(t_{j-1})$ and $\qhat(t_j)$ to estimate $\dot{\qhat}(t_j)$, hence the derivative cannot be estimated at $t = t_0$.

The forward, backward, and central difference functions ({func}`fwd1`, {func}`bwd3`, {func}`ctr6`, etc.) take in a snapshot matrix $\Qhat\in\RR^{r\times k}$, a time step, and (optionally) the corresponding input matrix $\U\in\RR^{m\times k}$ and return a subset of the snapshots $\Qhat'\in\mathbb{R}^{r\times k'}$, the corresponding derivatives $\dot{\Qhat}\in\RR^{r\times k'}$, and (optionally) the corresponding inputs $\U'\in\RR^{m \times k'}$.

```{code-cell} ipython3
# Set a state dimension, input dimension, and number of snapshots.
r = 20
m = 3
k = 400

# Make test data.
t = np.linspace(0, 1, k)
Q = np.random.random((r, k))
U = np.random.random((m, k))

# Extract the time step.
dt = t[1] - t[0]
```

```{code-cell} ipython3
Qnew, Qdot = opinf.ddt.bwd2(states=Q, dt=dt)
print(f"{Qnew.shape=}, {Qdot.shape=}")
```

```{code-cell} ipython3
Qnew, Qdot, Unew = opinf.ddt.ctr6(states=Q, dt=dt, inputs=U)
print(f"{Qnew.shape=}, {Qdot.shape=}, {Unew.shape=}")
```

### Complete Estimation

+++

The finite difference functions {func}`ord2`, {func}`ord4`, and {func}`ord6` mix forward, central, and backward differences to provide derivative estimates for all provided state snapshots.
These schemes are used by {func}`ddt_uniform`, and {func}`ddt_nonuniform`, which only return the estimated derivatives.

```{code-cell} ipython3
Qnew, Qdot = opinf.ddt.ord4(states=Q, dt=dt)
print(f"{Qnew.shape=}, {Qdot.shape=}")
print(f"{(Qnew is Q)=}")
```

```{code-cell} ipython3
Qdot = opinf.ddt.ddt_uniform(states=Q, dt=dt, order=4)
print(f"{Qdot.shape=}")
```

```{code-cell} ipython3
Qdot = opinf.ddt.ddt_nonuniform(states=Q, t=t)
print(f"{Qdot.shape=}")
```

### Convenience Classes

+++

The classes {class}`UniformFiniteDifferencer` and {class}`NonuniformFiniteDifferencer` wrap the finite difference methods listed above for use with {mod}`opinf.roms` classes.
They also have a [`verify()`](opinf.ddt.UniformFiniteDifferencer.verify) method for checking the estimation scheme against true derivatives for a limited set of test cases.

```{code-cell} ipython3
differ = opinf.ddt.UniformFiniteDifferencer(t, scheme="fwd1")
print(differ)
```

```{code-cell} ipython3
differ.verify(plot=True)
```

```{code-cell} ipython3
Qnew, Qdot, Unew = differ.estimate(states=Q, inputs=U)
print(f"{Qnew.shape=}, {Qdot.shape=}, {Unew.shape=}")
```

## Interpolatory Estimators

+++

The {class}`InterpDerivativeEstimator` interpolates the state data using classes from {mod}`scipy.interpolate` and evaluates the derivative of the interpolant.

```{code-cell} ipython3
estimator = opinf.ddt.InterpDerivativeEstimator(t, "pchip")
print(estimator)
```

```{code-cell} ipython3
estimator.verify(plot=True)
```

## Custom Estimators

+++

New time derivative estimators can be defined by inheriting from {class}`DerivativeEstimatorTemplate`.
Once implemented, the [`verify()`](opinf.ddt.DerivativeEstimatorTemplate.verify) method may be used to compare the results of [`estimate()`](opinf.ddt.DerivativeEstimatorTemplate.estimate) with true derivatives for a limited number of test cases.

```{code-cell} ipython3
class MyEstimator(opinf.ddt.DerivativeEstimatorTemplate):
    """Inheritance template for custom derivative estimators."""

    # Constructor -------------------------------------------------------------
    def __init__(self, time_domain, hyperparameters):
        """Set any hyperparameters.
        If there are no hyperparameters, __init__() may be omitted.
        """
        super().__init__(time_domain)
        # Process hyperparameters here.

    # Required methods --------------------------------------------------------
    def estimate(self, states, inputs=None):
        """Estimate the first time derivatives of the states.

        Parameters
        ----------
        states : (r, k) ndarray
            State snapshots, either full or (preferably) reduced.
        inputs : (m, k) ndarray or None
            Inputs corresponding to the state snapshots, if applicable.

        Returns
        -------
        _states : (r, k') ndarray
            Subset of the state snapshots.
        ddts : (r, k') ndarray
            First time derivatives corresponding to ``_states``.
        _inputs : (m, k') ndarray or None
            Inputs corresponding to ``_states``, if applicable.
            **Only returned** if ``inputs`` is provided.
        """
        raise NotImplementedError
```

### Example: Cubic Spline Interpolation

+++

The following class wraps {class}`scipy.interpolate.CubicSpline` and uses its `derivative()` method to compute the derivative.
This is a simplified version of {class}`DerivativeEstimatorTemplate`.

```{code-cell} ipython3
class CubicSplineEstimator(opinf.ddt.DerivativeEstimatorTemplate):
    """Derivative estimator using cubic spline interpolation."""

    # Constructor -------------------------------------------------------------
    def __init__(self, time_domain, bc_type="not-a-knot"):
        """Save arguments for scipy.interpolate.CubicSpline."""
        super().__init__(time_domain)
        self.options = dict(bc_type=bc_type, axis=1, extrapolate=None)

    # Required methods --------------------------------------------------------
    def estimate(self, states, inputs=None):
        """Estimate the first time derivatives of the states."""
        spline = spinterpolate.CubicSpline(
            x=self.time_domain,
            y=states,
            **self.options,
        )
        ddts = spline(self.time_domain, 1)
        if inputs is None:
            return states, ddts
        return states, ddts, inputs
```

```{code-cell} ipython3
spline_estimator = CubicSplineEstimator(t)
print(spline_estimator)
```

```{code-cell} ipython3
spline_estimator.verify()
```