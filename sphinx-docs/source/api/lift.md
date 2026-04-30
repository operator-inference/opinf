---
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
file_format: mystnb
---

# `opinf.lift`

+++

```{eval-rst}
.. automodule:: opinf.lift

.. currentmodule:: opinf.lift

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   LifterTemplate
   QuadraticLifter
   PolynomialLifter
```

+++

:::{admonition} Overview
:class: note

- Operator Inference models have polynomial structure. To calibrate a model with data, it can be advantageous to transform and/or augment the state variables to induce a desired structure.
- {class}`LifterTemplate` provides an API for variable transformation / augmentation operations.
:::

```{code-cell} ipython3
import numpy as np

import opinf
```

## Lifting Maps

+++

Operator Inference learns models with [polynomial terms](opinf.operators), for example,

$$
    \ddt\qhat(t)
    = \chat
    + \Ahat\qhat(t)
    + \Hhat[\qhat(t)\otimes\qhat(t)]
    + \Bhat\u(t).
$$

If training data do not exhibit this kind of polynomial structure, an Operator Inference model is not likely to perform well.
In some systems with nonpolynomial nonlinearities, a change of variables can induce a polynomial structure, which can greatly improve the effectiveness of Operator Inference.
Such variable transformations are often called _lifting maps_, especially if the transformation augments the state by introducing additional variables.

This module defines a template class for implementing lifting maps that can interface with {mod}`opinf.roms` classes and provides a few examples of lifting maps.

+++

:::{admonition} Example
:class: tip

This example originates from {cite}`qian2021thesis`.
Consider a nonlinear diffusion-reaction equation with a cubic reaction term:

$$
\begin{align*}
    \frac{\partial}{\partial t}q
    = \Delta q - q^3.
\end{align*}
$$

By introducing an auxiliary variable $w = q^{2}$, we have $\frac{\partial}{\partial t}w = 2q\frac{\partial q}{\partial t} = 2q\Delta q - 2q^4$, hence the previous equation can be expressed as the system

$$
\begin{align*}
    \frac{\partial}{\partial t}q
    &= \Delta q - qw,
    &
    \frac{\partial}{\partial t}w
    &= 2q\Delta q - 2w^2.
\end{align*}
$$

This system is quadratic in the lifted variables $(q, w)$, motivating a quadratic model structure instead of a cubic one.
The {class}`QuadraticLifter` class implements this transformation.
:::

+++

## Custom Lifting Maps

+++

New transformers can be defined by inheriting from the {class}`LifterTemplate`.
Once implemented, the [`verify()`](LifterTemplate.verify) method may be used to test the consistency of these three methods.

```{code-cell} ipython3
class MyLifter(opinf.lift.LifterTemplate):
    """Custom lifting map."""

    # Required methods --------------------------------------------------------
    @staticmethod
    def lift(state):
        """Lift the native state variables to the learning variables."""
        raise NotImplementedError

    @staticmethod
    def unlift(lifted_state):
        """Recover the native state variables from the learning variables."""
        raise NotImplementedError

    # Optional methods --------------------------------------------------------
    @staticmethod
    def lift_ddts(state, ddts):
        """Lift the native state time derivatives to the time derivatives
        of the learning variables.
        """
        raise NotImplementedError
```

A more detailed version of this class is included in the package as {class}`opinf.lift.QuadraticLifter`.

+++

### Example: Specific Volume Variables

+++

This example was used in {cite}`qian2019transform,qian2020liftandlearn,qian2021thesis,guo2022bayesopinf`.
The compressible Euler equations for an ideal gas can be written in conservative form as

$$
\begin{align*}
    \frac{\partial}{\partial t}\left[\rho\right]
    &= -\frac{\partial}{\partial x}\left[\rho u\right],
    &
    \frac{\partial}{\partial t}\left[\rho u\right]
    &= -\frac{\partial}{\partial x}\left[\rho u^2 + p\right],
    &
    \frac{\partial}{\partial t}\left[\rho e\right]
    &= -\frac{\partial}{\partial x}\left[(\rho e + p)u\right].
\end{align*}
$$

These equations are nonpolynomially nonlinear in the conservative variables $\vec{q}_{c} = (\rho, \rho u, \rho e)$.
However, by changing to the specific-volume variables $\vec{q} = (u, p, \zeta)$ and using the ideal gas law

$$
\begin{align*}
    \rho e = \frac{p}{\gamma - 1} + \frac{\rho u^2}{2},
\end{align*}
$$

we arrive at a _quadratic_ system

$$
\begin{align*}
    \frac{\partial u}{\partial t}
    &= -u \frac{\partial u}{\partial x} - \zeta\frac{\partial p}{\partial x},
    &
    \frac{\partial p}{\partial t}
    &= -\gamma p \frac{\partial u}{\partial x} - u\frac{\partial p}{\partial x},
    &
    \frac{\partial \zeta}{\partial t}
    &= -u \frac{\partial\zeta}{\partial x} + \zeta\frac{\partial u}{\partial x}.
\end{align*}
$$

Hence, a quadratic reduced-order model of the form

$$
    \frac{\text{d}}{\text{d}t}\qhat(t)
    = \Hhat[\qhat(t)\otimes\qhat(t)]
$$

can be learned for this system using data in the variables $\vec{q}$.
See {cite}`qian2020liftandlearn` for details.

The following class defines this the variable transformation.

```{code-cell} ipython3
class EulerLifter(opinf.lift.LifterTemplate):
    """Lifting map for the Euler equations transforming conservative
    variables to specific volume variables.
    """

    def __init__(self, gamma=1.4):
        """Store the heat capacity ratio, gamma."""
        self.gamma = gamma

    def lift(self, state):
        """Map the conservative variables to the learning variables,
        [rho, rho*u, rho*e] -> [u, p, 1/rho].
        """
        rho, rho_u, rho_e = np.split(state, 3)

        u = rho_u / rho
        p = (self.gamma - 1) * (rho_e - 0.5 * rho * u**2)
        zeta = 1 / rho

        return np.concatenate((u, p, zeta))

    def unlift(self, upzeta):
        """Map the learning variables to the conservative variables,
        [u, p, 1/rho] -> [rho, rho*u, rho*e].
        """
        u, p, zeta = np.split(upzeta, 3)

        rho = 1 / zeta
        rho_u = rho * u
        rho_e = p / (self.gamma - 1) + 0.5 * rho * u**2

        return np.concatenate((rho, rho_u, rho_e))
```

```{code-cell} ipython3
# Get test state data.
n = 100
Q = np.random.random((3 * n, 200))

# Verify the implementation.
EulerLifter().verify(Q)
```

:::{admonition} Developer Notes

- In this example, [`lift()`](LifterTemplate.lift) and [`unlift()`](LifterTemplate.unlift) are *not* static methods because they rely on the `gamma` attribute.
- Since [`lift_ddts()`](LifterTemplate.lift_ddts) is not implemented, [`verify()`](LifterTemplate.verify) only checks the consistency between [`lift()`](LifterTemplate.lift) and [`unlift()`](LifterTemplate.unlift).
:::