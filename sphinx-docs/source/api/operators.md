---
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
file_format: mystnb
---

# `opinf.operators`

+++

```{eval-rst}
.. automodule:: opinf.operators

.. currentmodule:: opinf.operators

**Nonparametric Operators**

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   OperatorTemplate
   InputMixin
   OpInfOperator
   ConstantOperator
   LinearOperator
   QuadraticOperator
   CubicOperator
   InputOperator
   StateInputOperator

**Parametric Operators**

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   ParametricOperatorTemplate
   ParametricOpInfOperator
   AffineConstantOperator
   AffineLinearOperator
   AffineQuadraticOperator
   AffineCubicOperator
   AffineInputOperator
   AffineStateInputOperator
   InterpConstantOperator
   InterpLinearOperator
   InterpQuadraticOperator
   InterpCubicOperator
   InterpInputOperator
   InterpStateInputOperator
```

+++

:::{admonition} Overview
:class: note

- `opinf.operators` classes represent the individual terms in a [model](opinf.models). [Nonparametric operators](sec-operators-nonparametric) are functions of the state and input, while [parametric operators](sec-operators-parametric) are also dependent on one or more external parameters.
- Operators that can be written as the product of a matrix and a known vector-valued function can be [calibrated](sec-operators-calibration) through solving a regression problem.
- {mod}`opinf.models` objects are  constructed with a list of operator objects. A model's [`fit()`](opinf.models.ContinuousModel.fit) method constructs and solves a regression problem to learn the operator matrices.
:::

<!-- - Monolithic operators are designed for dense systems; multilithic operators are designed for systems with sparse block structure. -->

```{code-cell} ipython3
import numpy as np
import scipy.linalg as la

import opinf
```

## Operators

+++

Models based on Operator Inference are systems of [ordinary differential equations](opinf.models.ContinuousModel) (or [discrete-time difference equations](opinf.models.DiscreteModel)) that can be written as a sum of terms,

$$
\begin{aligned}
   \ddt\qhat(t)
   = \sum_{\ell=1}^{n_\textrm{terms}}\Ophat_{\ell}(\qhat(t),\u(t)),
\end{aligned}
$$ (eq:operators:model)

where each $\Ophat_{\ell}:\RR^{r}\times\RR^{m}\to\RR^{r}$ is a vector-valued function of the reduced state $\qhat\in\RR^{r}$ and the input $\u\in\RR^{m}$.
We call these functions *operators* on this page.

Operator Inference calibrates operators that can be written as the product of a matrix $\Ohat_{\ell}\in\RR^{r \times d_\ell}$ and a known (possibly nonlinear) vector-valued function $\d_{\ell}:\RR^{r}\times\RR^{m}\to\RR^{d_ell}$,

$$
\begin{aligned}
    \Ophat_{\ell}(\qhat,\u)
    = \Ohat_{\ell}\d_{\ell}(\qhat,\u).
\end{aligned}
$$

Operators with this structure are called *OpInf operators* on this page.
The goal of Operator Inference is to learn the *operator matrix* $\Ohat_\ell$ for each OpInf operator in the model.

This module defines classes representing various operators.

+++

:::{admonition} Example
:class: tip

To represent a linear time-invariant (LTI) system

$$
\begin{align}
    \ddt\qhat(t)
    = \Ahat\qhat(t) + \Bhat\u(t),
    \qquad
    \Ahat\in\RR^{r \times r},
    ~
    \Bhat\in\RR^{r \times m},
\end{align}
$$ (eq:operators:ltiexample)

we use the following operator classes.

| Class | Definition | Operator matrix | data vector |
| :---- | :--------- | :-------------- | :---------- |
| {class}`LinearOperator` | $\Ophat_{1}(\qhat,\u) = \Ahat\qhat$ | $\Ohat_{1} = \Ahat \in \RR^{r\times r}$ | $\d_{1}(\qhat,\u) = \qhat\in\RR^{r}$ |
| {class}`InputOperator` | $\Ophat_{2}(\qhat,\u) = \Bhat\u$ | $\Ohat_{2} = \Bhat \in \RR^{r\times m}$ | $\d_{2}(\qhat,\u) = \u\in\RR^{m}$ |

An {class}`opinf.models.ContinuousModel` object can be instantiated with a list of operators objects to represent {eq}`eq:operators:ltiexample`:

```python
LTI_model = opinf.models.ContinuousModel(
    operators=[
        opinf.operators.LinearOperator(),
        opinf.operators.InputOperator(),
    ]
)
```

The operator matrices $\Ahat$ and $\Bhat$ are calibrated by calling [`LTI_model.fit()`](opinf.models.ContinuousModel.fit).
:::

+++

(sec-operators-nonparametric)=
## Nonparametric Operators

+++

A _nonparametric_ operator is a function of the state and input only.
For OpInf operators, this means the operator matrix $\Ohat_\ell$ is constant.
See [Parametric Operators](sec-operators-parametric) for operators that also depend on one or more external parameters.

Available nonparametric OpInf operators are listed below.

+++

```{eval-rst}
.. autosummary::
   :nosignatures:

   ConstantOperator
   LinearOperator
   QuadraticOperator
   CubicOperator
   InputOperator
   StateInputOperator
```

+++

Nonparametric OpInf operators can be instantiated without arguments.
If the operator matrix is known, it can be passed into the constructor or set later with [`set_entries()`](OpInfOperator.set_entries).
The operator matrix is stored as the [`entries`](OpInfOperator.entries) attribute and can be accessed with slicing operations `[:]`.

```{code-cell} ipython3
N = opinf.operators.StateInputOperator()
print(N)
```

```{code-cell} ipython3
r = 4
m = 2
Ohat = np.arange(r * r * m).reshape((r, r * m))

N.set_entries(Ohat)
print(N)
```

```{code-cell} ipython3
N.entries  # or N[:]
```

There are two ways to determine operator matrices in the context of model reduction:

- [**Non-intrusive Operator Inference**](sec-operators-calibration): Learn operator matrices from data.
- [**Intrusive (Petrov--)Galerkin Projection**](sec-operators-projection): Compress an existing high-dimensional operator.

Once the `entries` are set, the following methods are used to compute the action
of the operator or its derivatives.

- [`apply()`](OperatorTemplate.apply): compute the operator action $\Ophat_\ell(\qhat, \u)$.
- [`jacobian()`](OperatorTemplate.jacobian): construct the state Jacobian $\ddqhat\Ophat_\ell(\qhat, \u)$.

```{code-cell} ipython3
qhat = np.zeros(r)
u = np.ones(m)

N.apply(qhat, u)
```

```{code-cell} ipython3
N.jacobian(qhat, u)
```

(sec-operators-calibration)=
### Learning Operators from Data

+++

:::{attention}
This section describes the crux of Operator Inference.
:::

+++

Operator Inference requires state, input, and "left-hand side" data $\{(\qhat_j,\u_j,\z_j)\}_{j=0}^{k-1}$ that approximately satisfy the desired model dynamics.
For [time-continuous models](opinf.models.ContinuousModel), $\z_j$ is the [time derivative](opinf.ddt) of the state data; for [fully discrete models](opinf.models.DiscreteModel), $\z_j$ is the ``next state,'' usually $\qhat_{j+1}$.
For {eq}`eq:operators:model`, and assuming each operator is an OpInf operator, the data should approximately satisfy

$$
\begin{aligned}
    \z_j
    \approx \Ophat(\qhat_j, \u_j)
    = \sum_{\ell=1}^{n_\textrm{terms}} \Ophat_{\ell}(\qhat_j, \u_j)
    = \sum_{\ell=1}^{n_\textrm{terms}} \Ohat_{\ell}\d_{\ell}(\qhat_j, \u_j),
    \qquad
    j = 0, \ldots, k-1.
\end{aligned}
$$ (eq:operators:approx)

Operator Inference determines the operator matrices $\Ohat_1,\ldots,\Ohat_{n_\textrm{terms}}$ through a regression problem, written generally as

$$
\begin{aligned}
    \text{find}\quad\Ohat\quad\text{such that}\quad
    \Z \approx \Ohat\D\trp
    \quad\Longleftrightarrow\quad
    \D\Ohat\trp \approx \Z\trp,
\end{aligned}
$$

where $\D$ is the $k \times d$ *data matrix* formed from state and input snapshots and $\Z$ is the $r \times k$ matrix of left-hand side data.
To arrive at this problem, we write

$$
\begin{aligned}
    \z_j
    \approx \sum_{\ell=1}^{n_\textrm{terms}} \Ohat_{\ell}\d_{\ell}(\qhat_j, \u_j)
    = [~\Ohat_{1}~~\cdots~~\Ohat_{n_\textrm{terms}}~]
    \left[\begin{array}{c}
        \d_{1}(\qhat_j, \u_j)
        \\ \vdots \\
        \d_{n_\textrm{terms}}(\qhat_j, \u_j)
    \end{array}\right].
\end{aligned}
$$

Collecting this matrix-vector product for each $j=0,\ldots,k-1$ results in the matrix-matrix product system

$$
\begin{aligned}
    \left[\begin{array}{c|c|c}
        & & \\
        \z_0 & \cdots & \z_{k-1}
        \\ & &
    \end{array}\right]
    \approx
    [~\Ohat_{1}~~\cdots~~\Ohat_{n_\textrm{terms}}~]
    \left[\begin{array}{ccc}
        \d_{1}(\qhat_0, \u_0) & \cdots & \d_{1}(\qhat_{k-1}, \u_{k-1})
        \\ \vdots & & \vdots \\
        \d_{n_\textrm{terms}}(\qhat_0, \u_0) & \cdots & \d_{n_\textrm{terms}}(\qhat_{k-1}, \u_{k-1})
    \end{array}\right],
\end{aligned}
$$

which is $\Z \approx \Ohat\D\trp$ with

$$
\begin{aligned}
    \Z
    &= [~\z_0~~\cdots~~\z_{k-1}~] \in \RR^{r\times k},
    \\ & \\
    \Ohat
    &= [~\Ohat_{1}~~\cdots~~\Ohat_{n_\textrm{terms}}~] \in \RR^{r \times d},
    \\ & \\
    \D\trp
    &= \left[\begin{array}{ccc}
        \d_{1}(\qhat_0, \u_0) & \cdots & \d_{1}(\qhat_{k-1}, \u_{k-1})
        \\ \vdots & & \vdots \\
        \d_{n_\textrm{terms}}(\qhat_0, \u_0) & \cdots & \d_{n_\textrm{terms}}(\qhat_{k-1}, \u_{k-1})
    \end{array}\right]
    \in \RR^{d \times k},
\end{aligned}
$$

where $d = \sum_{i=1}^{n_\textrm{terms}}d_\ell$.

+++

Nonparametric OpInf operator classes have two static methods that facilitate constructing the operator regression problem.

- [`operator_dimension()`](OpInfOperator.operator_dimension): given the state dimension $r$ and the input dimension $r$, return the data dimension $d_\ell$.
- [`datablock()`](OpInfOperator.datablock): given the state-input data pairs $\{(\qhat_j,\u_j)\}_{j=0}^{k-1}$, form the matrix

$$
\begin{aligned}
    \D_{\ell}\trp = \left[\begin{array}{c|c|c|c}
        & & & \\
        \d_{\ell}(\qhat_0,\u_0) & \d_{\ell}(\qhat_1,\u_1) & \cdots & \d_{\ell}(\qhat_{k-1},\u_{k-1})
        \\ & & &
    \end{array}\right]
    \in \RR^{d_{\ell} \times k}.
\end{aligned}
$$

The complete data matrix $\D$ is the concatenation of the data matrices from each operator:

$$
\begin{aligned}
    \D = \left[\begin{array}{ccc}
        & & \\
        \D_1 & \cdots & \D_{n_\textrm{terms}}
        \\ & &
    \end{array}\right].
\end{aligned}
$$

+++

:::{admonition} Model Classes Do the Work
:class: important

Model classes from {mod}`opinf.models` are instantiated with a list of operators.
The model's `fit()` method calls the [`datablock()`](OpInfOperator.datablock) method of each OpInf operator to assemble the full data matrix $\D$, solves the regression problem for the full data matrix $\Ohat$ (see {mod}`opinf.lstsq`), and extracts from $\Ohat$ the individual operator matrix $\Ohat_{\ell}$ for each $\ell = 1, \ldots, n_{\textrm{terms}}$ using the [`operator_dimension()`](OpInfOperator.operator_dimension).

+++

:::{admonition} Example
:class: tip

For the LTI system {eq}`eq:operators:ltiexample`, the Operator Inference regression $\Z \approx \Ohat\D\trp$ has the following matrices.

$$
\begin{aligned}
    \Z
    &= [~\z_0~~\cdots~~\z_{k-1}~] \in \RR^{r\times k},
    \\ \\
    \Ohat
    &= [~\Ahat~~\Bhat~] \in \RR^{r \times (r + m)},
    \\ \\
    \D\trp
    &= \left[\begin{array}{ccc}
        \qhat_0 & \cdots & \qhat_{k-1}
        \\
        \u_0 & \cdots & \u_{k-1}
    \end{array}\right]
    \in \RR^{(r + m) \times k}.
\end{aligned}
$$

In this setting, $\z_j = \dot{\qhat}_j$, the time derivative of $\qhat_j$.
Collecting the state snapshots in the matrix $\Qhat = [~\qhat_0~~\cdots~~\qhat_{k-1}~]\in\RR^{r\times k}$ and the inputs in the matrix $\U = [~\u_0~~\cdots~~\u_{k-1}~]$, the full data matrix can be abbreviated as $\D = [~\Qhat\trp~~\U\trp~]$.

If the regression $\Z \approx \Ohat\D\trp$ is treated as an [ordinary least-squares problem](opinf.lstsq.PlainSolver), the optimization problem to solve is given by

$$
\begin{aligned}
    \min_{\Ohat}\left\|
        \D\Ohat\trp - \Z\trp
    \right\|_F^2
    = \min_{\Ahat,\Bhat}\sum_{j=0}^{k-1}\left\|
        \Ahat\qhat_j + \Bhat\u_j - \z_j
    \right\|_2^2.
\end{aligned}
$$

That is, the ordinary least-squares Operator Inference regression minimizes a sum of residuals of the model equation {eq}`eq:operators:ltiexample` with respect to available data.
:::

+++

:::{admonition} Operators with Entries are _Not_ Recalibrated
:class: important

Only operators whose entries are _not initialized_ (set to `None`) when a model is constructed are learned with Operator Inference when [`fit()`](opinf.models.ContinuousModel.fit) is called.
For example, suppose for the LTI system {eq}`eq:operators:ltiexample` an appropriate input matrix $\Bhat$ is known and stored as the variable `B_`.

```python
LTI_model = opinf.models.ContinuousModel(
    operators=[
        opinf.operators.LinearOperator(),   # No entries specified.
        opinf.operators.InputOperator(B_),  # Entries set to B_.
    ]
)
```

In this case, [`LIT_model.fit()`](opinf.models.ContinuousModel.fit) only determines the entries of the {class}`LinearOperator` object using Operator Inference.
The known information for $\Bhat$ is absorbed into the matrix $\Z$, so the Operator Inference regression $\Z \approx \Ohat\D\trp$ is now defined with the matrices

$$
\begin{aligned}
    \Z
    &= [~(\z_0 - \Bhat\u_0)~~\cdots~~(\z_{k-1} - \Bhat\u_{k-1})~] \in \RR^{r\times k},
    \\
    \Ohat
    &= \Ahat \in \RR^{r \times r},
    \qquad
    \D\trp
    = \Qhat \in \RR^{r \times k}.
\end{aligned}
$$

Using ordinary least-squares regression, the optimization problem is given by

$$
\begin{aligned}
    &\min_{\Ahat}\sum_{j=0}^{k-1}\left\|
        \Ahat\qhat_j - (\z_j - \Bhat\u_j)
    \right\|_2^2.
\end{aligned}
$$

A more likely scenario is that $\Bhat\in\RR^{r \times m}$ is derived from a known $\B\in\RR^{n \times m}$, which is the subject of the next section.
:::

+++

(sec-operators-projection)=
### Learning Operators via Projection

+++

The goal of Operator Inference is to learn operator entries from data because the operators of a high-fidelity / full-order model are unknown or computationally inaccessible.
However, in some scenarios a subset of the full-order model operators are known, in which case the corresponding reduced-order model operators can be determined through *intrusive projection*.
Consider a full-order operator $\Op:\RR^{n}\times\RR^{m}\to\RR^{n}$, written $\Op_{\ell}(\q,\u)$, where

- $\q\in\RR^n$ is the full-order state, and
- $\u\in\RR^m$ is the input.

Given a *trial basis* $\Vr\in\RR^{n\times r}$ and a *test basis* $\Wr\in\RR^{n\times r}$, the corresponding intrusive projection of $\Op_{\ell}$ is the operator $\Ophat_{\ell}:\RR^{r}\times\RR^{m}\to\RR^{r}$ defined by

$$
\begin{aligned}
    \Ophat_{\ell}(\qhat, \u)
    = (\Wr\trp\Vr)^{-1}\Wr\trp\Op_{\ell}(\Vr\qhat, \u)
\end{aligned}
$$

where
- $\qhat\in\RR^{r}$ is the reduced-order state, and
- $\u\in\RR^{m}$ is the input (the same as before).

This approach uses the low-dimensional state approximation $\q \approx \Vr\qhat$.
If $\Wr = \Vr$, the result is called a *Galerkin projection*.
Note that if $\Vr$ has orthonormal columns, we have in this case the simplification

$$
\begin{aligned}
    \Ophat_{\ell}(\qhat, \u)
    = \Vr\trp\Op_{\ell}(\Vr\qhat, \u).
\end{aligned}
$$

If $\Wr \neq \Vr$, the result is called a *Petrov--Galerkin projection*.

+++

:::{admonition} Example
:class: tip

Consider the bilinear operator
$\Op(\q,\u) = \N[\u\otimes\q]$ where $\N\in\RR^{n \times nm}$.
This type of operator can represented as a {class}`StateInputOperator`.
The intrusive Petrov--Galerkin projection of $\Op$ is the bilinear operator

$$
\begin{aligned}
    \Ophat(\qhat,\u)
    = (\Wr\trp\Vr)^{-1}\Wr\trp\N[\u\otimes\Vr\qhat]
    = \Nhat[\u\otimes\qhat]
\end{aligned}
$$

where $\Nhat = (\Wr\trp\Vr)^{-1}\Wr\trp\N(\I_m\otimes\Vr) \in \RR^{r\times rm}$.
Hence, $\Ophat$ can also be represented as a {class}`StateInputOperator`.
Using Galerkin projection ($\Wr = \Vr$), $\Nhat$ simplifies to $\Nhat = \Vr\trp\N(\I_m\otimes\Vr)$.
:::

Every operator class has a [`galerkin()`](OperatorTemplate.galerkin) method that performs intrusive projection.

+++

### Custom Nonparametric Operators

+++

New nonparametric operators can be defined by inheriting from {class}`OperatorTemplate` or, for operators that can be calibrated through Operator Inference, {class}`OpInfOperator`.
Once implemented, the [`verify()`](OperatorTemplate.verify) method may be used to test for consistency between [`apply()`](OperatorTemplate.apply) and the other methods outlined below.

For an arbitrary operator (not calibrated through Operator Inference), use the following inheritance template.

```{code-cell} ipython3
:tags: [hide-input]

class MyOperator(opinf.operators.OperatorTemplate):
    """Custom non-OpInf nonparametric operator."""

    # Constructor -------------------------------------------------------------
    def __init__(self, args_and_kwargs):
        """Construct the operator and set the state_dimension."""
        raise NotImplementedError

    # Required properties and methods -----------------------------------------
    @property
    def state_dimension(self):
        """Dimension of the state that the operator acts on."""
        return NotImplemented

    def apply(self, state, input_=None):
        """Apply the operator to the given state / input."""
        raise NotImplementedError

    # Optional methods --------------------------------------------------------
    def jacobian(self, state, input_=None):
        """Construct the state Jacobian of the operator."""
        raise NotImplementedError

    def galerkin(self, Vr: np.ndarray, Wr=None):
        """Get the (Petrov-)Galerkin projection of this operator."""
        raise NotImplementedError

    def save(self, savefile, overwrite=False):
        """Save the operator to an HDF5 file."""
        raise NotImplementedError

    @classmethod
    def load(cls, loadfile):
        """Load an operator from an HDF5 file."""
        raise NotImplementedError

    def copy(self):
        """Make a copy of the operator.
        If not implemented, copy.deepcopy() is used.
        """
        raise NotImplementedError
```

See {class}`OperatorTemplate` for details on the arguments for each method.

For a new Operator Inference operator, use the following inheritance template.

```{code-cell} ipython3
:tags: [hide-input]

class MyOpInfOperator(opinf.operators.OpInfOperator):
    """Custom nonparametric OpInf operator."""

    # Required methods --------------------------------------------------------
    @opinf.utils.requires("entries")
    def apply(self, state=None, input_=None):
        """Apply the operator to the given state / input."""
        raise NotImplementedError

    @staticmethod
    def datablock(states, inputs=None):
        """Return the data matrix block corresponding to the operator."""
        raise NotImplementedError

    @staticmethod
    def operator_dimension(r=None, m=None):
        """Column dimension of the operator entries matrix."""
        raise NotImplementedError

    # Optional methods --------------------------------------------------------
    @opinf.utils.requires("entries")
    def jacobian(self, state, input_=None):
        """Construct the state Jacobian of the operator.
        NOTE: If this method is omitted it is assumed that the Jacobian is
        zero, implying that the operator does not depend on the state.
        """
        raise NotImplementedError

    def galerkin(self, Vr, Wr=None):
        """Get the (Petrov-)Galerkin projection of this operator."""
        raise NotImplementedError
```

See {class}`OpInfOperator` for details on the arguments for each method.

+++

:::{admonition} Developer Notes
:class: note

- If the operator depends on the input $\u$, the class should also inherit from {class}`InputMixin` and set the [`input_dimension`](InputMixin.input_dimension) attribute.
- The [`jacobian()`](OperatorTemplate.jacobian) method is optional, but {mod}`opinf.models` objects have a `jacobian()` method that calls `jacobian()` for each of its operators. In an {class}`opinf.models.ContinuousModel`, the Jacobian is required for some time integration strategies used in [`predict()`](opinf.models.ContinuousModel.predict).
- The [`galerkin()`](OperatorTemplate.galerkin) method is optional, but {mod}`opinf.models` objects have a `galerkin()` method that calls `galerkin()` for each of its operators.
- The [`save()`](OperatorTemplate.save) and [`load()`](OperatorTemplate.load) methods should be implemented using {func}`opinf.utils.hdf5_savehandle()` and {func}`opinf.utils.hdf5_loadhandle()`, respectively.
:::

+++

#### Example: Hadamard Product with a Fixed Vector

+++

Consider the operator $\Ophat_{\ell}(\qhat, \u) = \qhat \ast \hat{\s}$, where $\hat{\s}\in\RR^{r}$ is a constant vector and $\ast$ denotes the [Hadamard (elementwise) product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)) (`*` in NumPy).
To implement an operator for this class, we first calculate its state Jacobian and determine the operator produced by (Petrov--)Galerkin projection.

Let $\qhat = [~\hat{q}_1~~\cdots~~\hat{q}_r~]\trp$ and $\hat{\s} = [~\hat{s}_1~~\cdots~~\hat{s}_r~]\trp$, i.e., the $i$-th entry of $\Ophat_{\ell}(\qhat, \u)$ is $\hat{q}_i\hat{s}_i$.
Then the $(i,j)$-th entry of the Jacobian is

$$
\begin{aligned}
    \frac{\partial}{\partial \hat{q}_j}\left[\hat{q}_i\hat{s}_i\right]
    = \begin{cases}
        \hat{s}_i & \textrm{if}~i = j,
        \\
        0 & \textrm{else}.
    \end{cases}
\end{aligned}
$$

That is, $\ddqhat\Ophat_{\ell}(\qhat, \u) = \operatorname{diag}(\hat{\s}).$

Now consider a version of this operator with a large state dimension, $\Op_{\ell}(\q, \u) = \q \ast \s$ for $\q,\s\in\mathbb{R}^{n}$.
For basis matrices $\Vr,\Wr\in\mathbb{R}^{n \times r}$, the Petrov--Galerkin projection of $\Op_{\ell}$ is given by

$$
\begin{aligned}
    \Ophat_{\ell}(\qhat, \u)
    = (\Wr\trp\Vr)^{-1}\Wr\trp\Op_{\ell}(\Vr\qhat, \u)
    = (\Wr\trp\Vr)^{-1}\Wr\trp((\Vr\qhat)\ast\s).
\end{aligned}
$$

It turns out that this product can be written as a matrix-vector product $\Ahat\qhat$ where $\Ahat$ depends on $\Vr$ and $\s$.
Therefore, `galerkin()` should return a {class}`LinearOperator` with entries matrix $\Ahat$.

The following class inherits from {class}`OperatorTemplate`, stores $\hat{\s}$ and sets the state dimension $r$ in the constructor, and implements the methods outlined the inheritance template.

```{code-cell} ipython3
class HadamardOperator(opinf.operators.OperatorTemplate):
    """Custom non-OpInf nonparametric operator for the Hadamard product."""

    # Constructor -------------------------------------------------------------
    def __init__(self, s):
        """Construct the operator and set the state_dimension."""
        self.svector = np.array(s)
        self._jac = np.diag(self.svector)

    # Required properties and methods -----------------------------------------
    @property
    def state_dimension(self):
        """Dimension of the state that the operator acts on."""
        return self.svector.shape[0]

    def apply(self, state, input_=None):
        """Apply the operator to the given state / input."""
        svec = self.svector
        if state.ndim == 2:
            svec = svec.reshape((self.state_dimension, 1))
        return state * svec

    # Optional methods --------------------------------------------------------
    def jacobian(self, state, input_=None):
        """Construct the state Jacobian of the operator."""
        return self._jac

    def galerkin(self, Vr, Wr=None):
        """Get the (Petrov-)Galerkin projection of this operator."""
        if Wr is None:
            Wr = Vr
        n = self.state_dimension
        r = Vr.shape[1]

        M = la.khatri_rao(Vr.T, np.eye(n)).T
        Ahat = Wr.T @ (M.reshape((n, r, n)) @ self.svector)
        if not np.allclose((WrTVr := Wr.T @ Vr), np.eye(r)):
            Ahat = la.solve(WrTVr, Ahat)
        return opinf.operators.LinearOperator(Ahat)

    def save(self, savefile, overwrite=False):
        """Save the operator to an HDF5 file."""
        with opinf.utils.hdf5_savehandle(savefile, overwrite) as hf:
            hf.create_dataset("svector", data=self.svector)

    @classmethod
    def load(cls, loadfile):
        """Load an operator from an HDF5 file."""
        with opinf.utils.hdf5_loadhandle(loadfile) as hf:
            return cls(hf["svector"][:])

    def copy(self):
        """Make a copy of the operator."""
        return self.__class__(self.svector)
```

```{code-cell} ipython3
r = 10
svec = np.random.standard_normal(r)
hadamard = HadamardOperator(svec)

print(hadamard)
```

```{code-cell} ipython3
hadamard.verify()
```

#### Example: Weighted Hadamard Input Operator

+++

Consider an operator

$$
\begin{aligned}
    \Ophat_{\ell}(\qhat,\u)
    = \Ohat_{\ell}(\u\ast\u),
\end{aligned}
$$

where $\Ohat_{\ell}\in\RR^{r \times m}$ and $\ast$ is the Hadamard (elementwise) product.
The matrix $\Ohat_{\ell}$ can be calibrated with Operator Inference.
Since $\Ophat_{\ell}$ does not depend on the state $\qhat$, the state Jacobian is zero.

```{code-cell} ipython3
class HadamardInputOperator(
    opinf.operators.OpInfOperator,
    opinf.operators.InputMixin,
):
    """Custom nonparametric OpInf operator: Hadamard of inputs."""

    # Required methods --------------------------------------------------------
    @property
    def input_dimension(self):
        """Dimension of the input that the operator acts on."""
        return self.entries.shape[1] if self.entries is not None else None

    @opinf.utils.requires("entries")
    def apply(self, state=None, input_=None):
        """Apply the operator to the given state / input."""
        return self.entries @ (input_**2)

    @staticmethod
    def datablock(states, inputs=None):
        """Return the data matrix block corresponding to the operator."""
        return inputs**2

    @staticmethod
    def operator_dimension(r=None, m=None):
        """Column dimension of the operator entries matrix."""
        return m

    # Optional methods --------------------------------------------------------
    def galerkin(self, Vr, Wr=None):
        """Get the (Petrov-)Galerkin projection of this operator."""
        if Wr is None:
            Wr = Vr
        r = Vr.shape[1]

        Ohat = Wr.T @ self.entries
        if not np.allclose((WrTVr := Wr.T @ Vr), np.eye(r)):
            Ohat = la.solve(WrTVr, Ohat)

        return self.__class__(Ohat)
```

```{code-cell} ipython3
# Instantiate an operator without entries.
hadamard_input = HadamardInputOperator()

print(hadamard_input)
```

```{code-cell} ipython3
hadamard_input.verify()
```

```{code-cell} ipython3
m = 4
Ohat = np.random.random((r, m))
hadamard_input.set_entries(Ohat)

print(hadamard_input)
```

```{code-cell} ipython3
hadamard_input.verify()
```

(sec-operators-parametric)=
## Parametric Operators

+++

An operator is called _parametric_ if it depends on an independent parameter vector
$\bfmu\in\RR^{p}$, i.e., $\Ophat_{\ell} = \Ophat_{\ell}(\qhat,\u;\bfmu)$
When the parameter vector is fixed, a parametric operator becomes nonparametric.
In particular, a parametric operator's [`evaluate()`](ParametricOperatorTemplate.evaluate) method accepts a parameter vector $\bfmu$ and returns an instance of a nonparametric operator.

+++

Parametric OpInf operators have the form
$\Ophat_{\ell}(\qhat,\u;\bfmu) = \Ohat_{\ell}(\bfmu)\d_{\ell}(\qhat,\u)$ defined by the matrix-valued function $\Ohat_{\ell}:\RR^{p}\to\RR^{r\times d_\ell}$ and (as in the nonparametric case) the data vector $\d_{\ell}:\RR^{r}\times\RR^{m}\to\RR^{d_\ell}$.
This module provides two options for the parameterization of $\Ohat_{\ell}(\bfmu)$: [affine expansion](sec-operators-affine) and [elementwise interpolation](sec-operators-interpolated).
In each case, Operator Inference begins with $s$ training parameter values $\bfmu_{0},\ldots,\bfmu_{s-1}$ and corresponding state, input, and left-hand side data $\{(\qhat_{i,j},\u_{i,j},\z_{i,j})\}_{j=0}^{k_{i}-1}$ for each training parameter value $\bfmu_{i}$.
A regression of the form $\Z \approx \Ohat\D\trp$ is formed as in the nonparametric case, with the structure of the matrices $\Ohat$ and $\D$ depending on the choice of parameterization for each $\Ohat_{\ell}(\bfmu)$.

+++

:::{admonition} Example
:class: tip
Let $\bfmu = [~\mu_{0}~~\mu_{1}~]\trp$.
The operator

$$
\begin{aligned}
    \Ophat_1(\qhat,\u;\bfmu) = (\mu_{0}\Ahat_{0} + \mu_{1}^{2}\Ahat_{1})\qhat
\end{aligned}
$$

is a parametric OpInf operator because it can be written as $\Ophat_1(\qhat,\u;\bfmu) = \Ohat_1(\bfmu)\d_1(\qhat,\u)$ with $\Ohat_1(\bfmu) = \mu_{0}\Ahat_{0} + \mu_{1}^{2}\Ahat_{1}$ and $\d_1(\qhat,\u) = \qhat$.

This operator can be represented with an {class}`AffineLinearOperator`.
For a given parameter value, the [`evaluate()`](AffineLinearOperator.evaluate) method returns a {class}`LinearOperator` instance.
:::

+++

(sec-operators-affine)=
### Affine Operators

+++

Affine parametric OpInf operators $\Ophat_{\ell}(\qhat,\u;\bfmu) = \Ohat_{\ell}(\bfmu)\d_{\ell}(\qhat,\u)$ parameterize the operator matrix $\Ohat_{\ell}(\bfmu)$ as a sum of constant matrices with parameter-dependent scalar coefficients,

$$
\begin{aligned}
    \Ohat_{\ell}(\bfmu)
    &= \sum_{a=0}^{A_{\ell}-1}\theta_{\ell}^{(a)}\!(\bfmu)\,\Ohat_{\ell}^{(a)},
\end{aligned}
$$

where each $\theta_{\ell}^{(a)}:\RR^{p}\to\RR$ is a scalar-valued function and each $\Ohat_{\ell}^{(a)}\in\RR^{r\times d_\ell}$ is constant.
Affine expansions are grouped such that the coefficient functions $\theta_{\ell}^{(0)},\ldots,\theta_{\ell}^{(A_{\ell}-1)}$ are linearly independent.

+++

Affine parametric operators arise in Operator Inference settings because linear projection preserves affine structure.

:::{dropdown} Preservation of Affine Structure

Consider a full-order affine parametric OpInf operator

$$
\begin{aligned}
    \Op_{\ell}(\q,\u;\bfmu)
    = \sum_{a=0}^{A_{\ell}-1}\theta_{\ell}^{(a)}\!(\bfmu)\,\Op_{\ell}^{(a)}\!(\q, \u).
\end{aligned}
$$

Given a trial basis $\Vr\in\RR^{n\times r}$ and a test basis $\Wr\in\RR^{n\times r}$, the [intrusive projection](sec-operators-projection) of $\Op_{\ell}$ is the operator

$$
\begin{aligned}
    \Ophat_{\ell}(\qhat, \u; \bfmu)
    &= (\Wr\trp\Vr)^{-1}\Wr\trp\Op_{\ell}(\Vr\qhat, \u; \bfmu)
    \\
    &= (\Wr\trp\Vr)^{-1}\Wr\trp \sum_{a=0}^{A_{\ell}-1}\theta_{\ell}^{(a)}\!(\bfmu)\,\Op_{\ell}^{(a)}\!(\V\qhat, \u)
    \\
    &= \sum_{a=0}^{A_{\ell}-1}\theta_{\ell}^{(a)}\!(\bfmu)\,(\Wr\trp\Vr)^{-1}\Wr\trp\Op_{\ell}^{(a)}\!(\V\qhat, \u)
    = \sum_{a=0}^{A_{\ell}-1}\theta_{\ell}^{(a)}\!(\bfmu)\,\Ophat_{\ell}^{(a)}\!(\qhat, \u),
\end{aligned}
$$

where $\Ophat_{\ell}^{(a)}\!(\qhat, \u) = (\Wr\trp\Vr)^{-1}\Wr\trp\Op_{\ell}^{(a)}\!(\V\qhat, \u)$ is the intrusive projection of $\Op_{\ell}^{(a)}$.
That is, the intrusive projection of an affine expansion is an affine expansion of intrusive projections, and both expansions feature the same coefficient functions.
:::

+++

Available affine parametric operators are listed below.

```{eval-rst}
.. currentmodule:: opinf.operators

.. autosummary::
    :nosignatures:

    AffineConstantOperator
    AffineLinearOperator
    AffineQuadraticOperator
    AffineCubicOperator
    AffineInputOperator
    AffineStateInputOperator
```

+++

Affine parametric operators are instantiated with a function $\boldsymbol{\theta}_{\ell}(\mu) = [~\theta_{\ell}^{(0)}(\bfmu)~~\cdots~~\theta_{\ell}^{(A_{\ell}-1)}(\bfmu)~]\trp$ for the affine expansion coefficients, the number of terms $A_{\ell}$ in the expansion, and with or without the operator matrices $\Ohat_{\ell}^{(1)},\ldots,\Ohat_{\ell}^{(A_{\ell})}$.

```{code-cell} ipython3
thetas = lambda mu: np.array([mu[0], mu[1] ** 2])
A = opinf.operators.AffineLinearOperator(thetas, nterms=2)
print(A)
```

```{code-cell} ipython3
# Set the constant operator matrices in the affine expansion.
r = 5
Ahats = [np.ones((r, r)), np.eye(r)]
A.set_entries(Ahats, fromblock=False)

print(A)
```

```{code-cell} ipython3
A.entries
```

```{code-cell} ipython3
# Evaluate the parametric operator at a fixed parameter value,
# resulting in a nonparametric operator.
A_nonparametric = A.evaluate([2, 4])
print(A_nonparametric)
```

```{code-cell} ipython3
A_nonparametric.entries
```

```{code-cell} ipython3
# The parameter dimension p = 2 was recorded when A was evaluated.
print(A)
```

:::{dropdown} Operator Inference for Affine Operators

Consider a model with a single affine operator,

$$
\begin{aligned}
    \z
    = \Ophat_{1}(\qhat,\u;\bfmu)
    &= \left(\sum_{a=0}^{A_{1}-1}\theta_{1}^{(a)}\!(\bfmu)\,\Ohat_{1}^{(a)}\right)\d_1(\qhat,\u).
\end{aligned}
$$

For the moment we will neglect the $\ell = 1$ subscript.
With data $\{(\qhat_{i,j},\u_{i,j},\z_{i,j})\}_{j=0}^{k_{i}-1}$ corresponding to $s$ training parameter values $\bfmu_0,\ldots,\bfmu_{s-1}$, we seek the $A$ operator matrices $\Ohat^{(0)},\ldots,\Ohat^{(A-1)}$ such that for each parameter index $i$ and time index $j$, we have

$$
\begin{aligned}
    \z_{i,j}
    \approx \Ophat(\qhat_{i,j},\u_{i,j};\bfmu_{i})
    &= \left(\sum_{a=0}^{A-1}\theta^{(a)}\!(\bfmu_{i})\,\Ohat^{(a)}\right)\d(\qhat_{i,j},\u_{i,j})
    \\
    &= \left[\begin{array}{ccc}
    \Ohat^{(0)} & \cdots & \Ohat^{(A-1)}
    \end{array}\right]
    \underbrace{\left[\begin{array}{c}
    \theta^{(0)}\!(\bfmu_{i})\,\d(\qhat_{i,j},\u_{i,j})
    \\ \vdots \\
    \theta^{(A-1)}\!(\bfmu_{i})\,\d(\qhat_{i,j},\u_{i,j})
    \end{array}\right]}_{\d_{i,j}\in\RR^{dA}},
\end{aligned}
$$

where $d$ is the output dimension of $\d$.
Collecting these expressions for each time index $j = 0, \ldots, k_i - 1$ (but keeping the parameter index $i$ fixed for the moment) results in

$$
\begin{aligned}
    \underbrace{\left[\begin{array}{ccc}
    \z_{i,0} & \cdots & \z_{i,k_i-1}
    \end{array}\right]}_{\Z_i\in\RR^{r\times k_i}}
    \approx \left[\begin{array}{ccc}
    \Ohat^{(0)} & \cdots & \Ohat^{(A-1)}
    \end{array}\right]
    \underbrace{\left[\begin{array}{ccc}
    \d_{i,0} & \cdots & \d_{i,k_i-1}
    \end{array}\right]}_{\D_i\trp\in\RR^{dA \times k_i}}.
\end{aligned}
$$

Finally, we concatenate each of these expressions for each parameter index $i = 0,\ldots, s-1$ to arrive at

$$
\begin{aligned}
    \underbrace{\left[\begin{array}{ccc}
    \Z_{0} & \cdots & \Z_{s-1}
    \end{array}\right]}_{\Z\in\RR^{r\times K}}
    \approx \left[\begin{array}{ccc}
    \Ohat^{(0)} & \cdots & \Ohat^{(A-1)}
    \end{array}\right]
    \underbrace{\left[\begin{array}{ccc}
    \D_{0}\trp & \cdots & \D_{s-1}\trp
    \end{array}\right]}_{\D\trp\in\RR^{dA \times K}},
\end{aligned}
$$

where $K = \sum_{i=0}^{s-1}k_i$, the total number of available data instances.
This is the familiar $\Z \approx \Ohat\D\trp$ where $\Ohat = [~\Ohat^{(1)}~~\cdots~~\Ohat^{(A)}~]$, which can be solved for $\Ohat$ using {mod}`opinf.lstsq`.

The construction of $\D$ is taken care of through the [`datablock()`](AffineLinearOperator.datablock) method of the affine 

For models with multiple affine operators, the operator matrix $\Ohat$ is further concatenated horizontally to accommodate the operator matrices from each affine expansion, and the data matrix $\D\trp$ gains additional block rows.
:::

+++

(sec-operators-interpolated)=
### Interpolatory Operators

+++

Interpolatory parametric OpInf operators define the parametric dependence of the operator matrix on $\bfmu$ through elementwise interpolation.
That is,

$$
\begin{aligned}
    \Ophat_{\ell}(\qhat,\u;\bfmu)
    = \Ohat_{\ell}(\bfmu)\d_{\ell}(\qhat,\u),
\end{aligned}
$$

where $\Ohat_{\ell}(\bfmu)$ is determined by interpolating $s$ matrices $\Ohat_{\ell}^{(0)},\ldots,\Ohat_{\ell}^{(s-1)}$.
In the context of Operator Inference, $s$ is the number of training parameter values.

+++

Available interpolatory operators are listed below.

```{eval-rst}
.. currentmodule:: opinf.operators

.. autosummary::
    :nosignatures:

    InterpConstantOperator
    InterpLinearOperator
    InterpQuadraticOperator
    InterpCubicOperator
    InterpInputOperator
    InterpStateInputOperator
```

+++

Interpolatory operators can be instantiated with no arguments.

```{code-cell} ipython3
B = opinf.operators.InterpInputOperator()
print(B)
```

```{code-cell} ipython3
s = 9  # Number of training parameters
p = 1  # Dimension of the training parameters.
r = 4  # Dimension of the states.
m = 2  # Dimension of the inputs.

training_parameters = np.random.standard_normal((s, p))
operator_matrices = [np.random.random((r, m)) for _ in range(s)]

B.set_training_parameters(training_parameters)
print(B)
```

```{code-cell} ipython3
B.set_entries(operator_matrices, fromblock=False)
print(B)
```

```{code-cell} ipython3
B_nonparametric = B.evaluate(np.random.standard_normal(p))
print(B_nonparametric)
```

:::{dropdown} Operator Inference for Interpolatory Operators

Consider a model with a single affine operator,

$$
\begin{aligned}
    \z
    = \Ophat_{1}(\qhat,\u;\bfmu)
    = \Ohat_{1}(\bfmu)\d_{1}(\qhat,\u).
\end{aligned}
$$

For the moment we will neglect the $\ell = 1$ subscript.
With data $\{(\qhat_{i,j},\u_{i,j},\z_{i,j})\}_{j=0}^{k_{i}-1}$ corresponding to $s$ training parameter values $\bfmu_0,\ldots,\bfmu_{s-1}$, we seek $s$ operator matrices $\Ohat^{(0)},\ldots,\Ohat^{(s-1)}\in\RR^{r\times d}$ such that for each parameter index $i$ and time index $j$, we have

$$
\begin{aligned}
    \z_{i,j}
    \approx \Ophat(\qhat_{i,j},\u_{i,j};\bfmu_{i})
    = \Ohat^{(i)}\d(\qhat_{i,j},\u_{i,j}),
\end{aligned}
$$

which comes from the interpolation condition $\Ohat^{(i)} = \Ohat_{1}(\bfmu_{i})$ for $i = 0,\ldots,s-1$.
Because only one operator matrix $\Ohat^{(i)}$ defines the operator action at each parameter value for which we have data, we have $s$ independent nonparametric Operator Inference problems:

$$
\begin{aligned}
    \Z_i
    &\approx \Ohat^{(i)}\D_i\trp,
    \\
    \Z_i
    &= \left[\begin{array}{ccc}
        \z_{i,0} & \cdots & \z_{i,k_{i}-1}
    \end{array}\right]\in\RR^{r \times k_{i}},
    \\
    \D_i\trp
    &= \left[\begin{array}{ccc}
        \d(\qhat_{i,0}, \u_{i,0}) & \cdots & \d(\qhat_{i,k_{i}-1}, \u_{i,k_{i}-1})
    \end{array}\right]\in\RR^{d\times k_{i}}
\end{aligned}
$$

The InterpolatedModel classes represent models comprised solely of interpolatory operators.
If interpolatory operators are mixed with other operators (nonparametric or affine parametric), the $\Ohat\D\trp$ block of the problem for the interpolatory operator is included as follows:

$$
\begin{aligned}
    \left[\begin{array}{ccc}
    \Ohat^{(0)} & \cdots & \Ohat^{(s-1)}
    \end{array}\right]
    \left[\begin{array}{cccc}
    \D_0\trp & \0 & \cdots & \0 \\
    \0 & \D_1\trp & \cdots & \0 \\
    \vdots & \vdots & \ddots & \vdots \\
    \0 & \0 & \cdots & \D_{s-1}\trp
    \end{array}\right]
\end{aligned}
$$
:::

+++

:::{dropdown} Mixing Nonparametric and Parametric Operators

Consider a system of ODEs with a mix of parametric and nonparametric operators,

$$
\begin{aligned}
    \ddt\qhat(t)
    = \left(\mu_{0}\Ahat^{(0)} + \cos(\mu_{1})\Ahat^{(1)}\right)\qhat(t) + \Hhat[\qhat(t) \otimes \qhat(t)] + \Bhat(\bfmu)\u(t).
\end{aligned}
$$

This model can be written in the general form {eq}`eq:operators:model` with three operators:

- $\Ophat_1(\qhat,\u;\bfmu) = \left(\theta^{(0)}\!(\bfmu)\,\Ahat^{(0)} + \theta^{(1)}\!(\bfmu)\,\Ahat^{(1)}\right)\qhat(t)$, an affine-parametric linear operator where $\theta^{(0)}\!(\bfmu) = \mu_{0}$ and $\theta^{(1)}\!(\bfmu) = \cos(\mu_{1})$;
- $\Ophat_2(\qhat,\u) = \Hhat[\qhat(t) \otimes \qhat(t)]$, a nonparametric quadratic operator; and
- $\Ophat_3(\qhat,\u;\bfmu) = \Bhat(\bfmu)\u(t)$, a parametric input operator without a specified parametric structure.

If $\Bhat(\bfmu)$ is parameterized with interpolation, the Operator Inference problem to learn the operator matrices can be written as $\Z \approx \Ohat\D\trp$ in the following way.
Let $\Qhat_i\in\RR^{r\times k_i}$ and $\U_i\in\RR^{m \times k_i}$ collect the state and input data for training parameter value $\bfmu_i$, with corresponding state time derivative data $\Z_i = \dot{\Qhat}_i\in\RR^{r\times k_i}$ for $i = 0,\ldots, s-1$.
We then have

$$
\begin{aligned}
    \Z
    &= \left[\begin{array}{ccc}
        \Z_0 & \cdots & \Z_{s-1}
    \end{array}\right]\in\RR^{r\times K}
    \\
    \Ohat
    &= \left[\begin{array}{cc|c|ccc}
        \Ahat^{(0)} & \Ahat^{(1)} & \Hhat & \Bhat^{(0)} & \cdots & \Bhat^{(s-1)}
    \end{array}\right]\in\RR^{r \times d}
    \\
    \D\trp
    &= \left[\begin{array}{}
        \theta^{(0)}\!(\bfmu_0)\Qhat_{0} & \cdots & \theta^{(0)}\!(\bfmu_s)\Qhat_{s} \\
        \theta^{(1)}\!(\bfmu_0)\Qhat_{0} & \cdots & \theta^{(1)}\!(\bfmu_s)\Qhat_{s} \\ \hline
        \Qhat_{0}\odot\Qhat_{0} & \cdots & \Qhat_{s}\odot\Qhat_{s} \\ \hline
        \U_{0} & \cdots & \0 \\
        \vdots & \ddots & \0 \\
        \0 & \cdots & \U_{s-1}
    \end{array}\right]\in\RR^{d \times K},
\end{aligned}
$$

where $K = \sum_{i=0}^{s-1}k_i$ is the total number of data snapshots and $d = 2r + r(r+1)/2 + sm$ is the total operator dimension.
Note that the operator and data matrices have blocks corresponding to each of the three operators in the model.
:::