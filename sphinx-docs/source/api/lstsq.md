---
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
file_format: mystnb
---

# `opinf.lstsq`

+++

```{eval-rst}
.. automodule:: opinf.lstsq

.. currentmodule:: opinf.lstsq

.. autosummary::
   :toctree: _autosummaries
   :nosignatures:

   SolverTemplate
   PlainSolver
   L2Solver
   L2DecoupledSolver
   TikhonovSolver
   TikhonovDecoupledSolver
   TruncatedSVDSolver
   TotalLeastSquaresSolver
```

+++

:::{admonition} Overview
:class: note

- `opinf.lstsq` classes are solve Operator Inference regression problems.
- `opinf.models` classes handle the construction of the regression matrices from snapshot data, pass these matrices to the solver's [`fit()`](SolverTemplate.fit) method, call the solver's [`solve()`](SolverTemplate.solve) method, and interpret the solution in the context of the model structure.
:::

+++

:::{admonition} Example Data
:class: tip
The examples on this page use data matrices composed of compressed heat flow data.
You can [download the data here](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/raw/data/lstsq_example.npz) to repeat the demonstration.
:::

```{code-cell} ipython3
import numpy as np
import scipy.optimize as opt

import opinf
```

## Operator Inference Regression Problems

+++

Operator Inference uses data to learn the entries of an $r \times d$ *operator matrix* $\Ohat$ by solving a regression problem, stated generally as

::::{margin}
:::{admonition} What is $\Z$?
For continuous models (systems of ordinary differential equations), $\Z$ consists of the time derivatives of the snapshots; for discrete models (discrete dynamical systems), $\Z$ also contains state snapshots.
:::
::::

$$
\begin{aligned}
    \text{find}\quad\Ohat\quad\text{such that}\quad
    \Z \approx \Ohat\D\trp
    \quad\Longleftrightarrow\quad
    \D\Ohat\trp \approx \Z\trp,
\end{aligned}
$$ (eq:lstsq:general)

where $\D$ is the $k \times d$ *data matrix* formed from state and input snapshots and $\Z$ is the $r \times k$ matrix of left-hand side data.

This module defines classes for solving different variations of the regression {eq}`eq:lstsq:general` given the data matrices $\D$ and $\Z$.

+++ {"vscode": {"languageId": "plaintext"}}

::::{admonition} Example
:class: tip

Suppose we want to construct a linear time-invariant (LTI) system,

$$
\begin{align}
    \ddt\qhat(t)
    = \Ahat\qhat(t) + \Bhat\u(t),
    \qquad
    \Ahat\in\RR^{r \times r},
    ~
    \Bhat\in\RR^{r \times m}.
\end{align}
$$ (eq:lstsq:ltiexample)

The operator matrix is $\Ohat = [~\Ahat~~\Bhat~]\in\RR^{r \times d}$ with column dimension $d = r + m$.
To learn $\Ohat$ with Operator Inference, we need data for $\qhat(t)$, $\u(t)$, and $\ddt\qhat(t)$.
For $j = 0, \ldots, k-1$, let

- $\qhat_{j}\in\RR^r$ be a measurement of the (reduced) state at time $t_{j}$,
- $\dot{\qhat}_{j} = \ddt\qhat(t)\big|_{t=t_{j}} \in \RR^r$ be the time derivative of the state at time $t_{j}$, and
- $\u_{j} = \u(t_j) \in \RR^m$ be the input at time $t_{j}$.

In this case, the data matrix $\D$ is given by $\D = [~\Qhat\trp~~\U\trp~]\in\RR^{k \times d}$, where

$$
\begin{aligned}
    \Qhat = \left[\begin{array}{ccc}
        & & \\
        \qhat_0 & \cdots & \qhat_{k-1}
        \\ & &
    \end{array}\right]
    \in \RR^{r\times k},
    \qquad
    \U = \left[\begin{array}{ccc}
        & & \\
        \u_0 & \cdots & \u_{k-1}
        \\ & &
    \end{array}\right]
    \in \RR^{m \times k}.
\end{aligned}
$$

The left-hand side data is $\Z = \dot{\Qhat} = [~\dot{\qhat}_0~~\cdots~~\dot{\qhat}_{k-1}~]\in\RR^{r\times k}$.

:::{dropdown} Derivation
We seek $\Ahat$ and $\Bhat$ such that

$$
\begin{aligned}
    \dot{\qhat}_{j}
    \approx \Ahat\qhat_j + \Bhat\u_j,
    \qquad j = 0, \ldots, k-1.
\end{aligned}
$$

Using the snapshot matrices $\Qhat$, $\U$, and $\dot{\Qhat}$ defined above, we want

$$
\begin{aligned}
    \dot{\Qhat}
    \approx \Ahat\Qhat + \Bhat\U
    = [~\Ahat~~\Bhat~]\left[\begin{array}{c} \Qhat \\ \U \end{array}\right],
    \quad\text{or}
    \\
    [~\Qhat\trp~~\U\trp~][~\Ahat~~\Bhat~]\trp \approx \dot{\Qhat}\trp,
\end{aligned}
$$

which is $\D\Ohat\trp \approx \Z\trp$.

More precisely, a regression problem for $\Ohat$ with respect to the data triples $(\qhat_j, \u_j, \dot{\qhat}_j)$ can be written as

$$
\begin{aligned}
    \argmin_{\Ahat,\Bhat}\sum_{j=0}^{k-1}\left\|
        \Ahat\qhat_j + \Bhat\u_j - \dot{\qhat}_j
    \right\|_{2}^{2}
    &= \argmin_{\Ahat,\Bhat}\left\|
        \Ahat\Qhat + \Bhat\U - \dot{\Qhat}
    \right\|_{F}^{2}
    \\
    &= \argmin_{\Ahat,\Bhat}\left\|
        [~\Ahat~~\Bhat~]\left[\begin{array}{c} \Qhat \\ \U \end{array}\right] - \Z
    \right\|_{F}^{2}
    \\
    &= \argmin_{\Ahat,\Bhat}\left\|
        [~\Qhat\trp~~\U\trp~][~\Ahat~~\Bhat~]\trp - \Z\trp
    \right\|_{F}^{2},
\end{aligned}
$$

which is $\argmin_{\Ohat}\|\D\Ohat\trp - \Z\trp\|_F^2$.
:::
::::

+++

## Default Solver

+++

Most often, we pose {eq}`eq:lstsq:general` as a linear least-squares regression,

$$
\begin{aligned}
    \argmin_{\Ohat} \|\D\Ohat\trp - \Z\trp\|_F^2.
\end{aligned}
$$ (eq:lstsq:plain)

Note that the matrix least-squares problem {eq}`eq:lstsq:plain` decouples into $r$ independent vector least-squares problems, i.e.,

$$
\begin{aligned}
    \argmin_{\ohat_i} \|\D\ohat_i - \z_i\|_2^2,
    \quad i = 1, \ldots, r,
\end{aligned}
$$

where $\ohat_i$ and $\z_i$ are the $i$-th rows of $\Ohat$ and $\Z$, respectively.

The {class}`PlainSolver` class solves {eq}`eq:lstsq:plain` without any additional terms.
This is the default solver used if another solver is not specified in the constructor of an {mod}`opinf.models` class.

```{code-cell} ipython3
# Load and extract the example data.
data = np.load("lstsq_example.npz")
D = data["data_matrix"]
Z = data["lhs_matrix"]

print(f"{D.shape=}, {Z.shape=}")

# Infer problem dimensions from the data.
k, d = D.shape
r = Z.shape[0]

print(f"{r=}, {d=}, {k=}")
```

```{code-cell} ipython3
solver = opinf.lstsq.PlainSolver()
print(solver)
```

```{code-cell} ipython3
# Fit the solver to the regression data.
solver.fit(D, Z)
print(solver)
```

```{code-cell} ipython3
# Solve the least-squares problem for the operator matrix.
Ohat = solver.solve()
print(f"{Ohat.shape=}")
```

```{code-cell} ipython3
# Check how well the solution matches the data.
print("\nOptimization residuals:", solver.residual(Ohat), sep="\n")
```

## Tikhonov Regularization

+++


It is often advantageous to add a *regularization term* $\mathcal{R}(\Ohat)$ to penalize the entries of the inferred operators.
This can prevent over-fitting to data and promote stability in the learned reduced-order model {cite}`mcquarrie2021combustion`.
The least-squares regression {eq}`eq:lstsq:plain` then becomes

$$
\begin{aligned}
    \argmin_{\Ohat}\|
        \D\Ohat\trp - \Z\trp
    \|_{F}^{2} + \mathcal{R}(\Ohat).
\end{aligned}
$$

A [Tikhonov regularization](https://en.wikipedia.org/wiki/Ridge_regression#Tikhonov_regularization) term has the form

$$
\begin{aligned}
    \mathcal{R}(\Ohat)
    = \sum_{i=1}^{r}\|\bfGamma_i\ohat_i\|_2^2,
\end{aligned}
$$

where $\ohat_1,\ldots,\ohat_r$ are the rows of $\Ohat$ and each $\bfGamma_1,\ldots,\bfGamma_r$ is a $d \times d$ symmetric positive-definite matrix.
In this case, the decoupled regressions for the rows of $\Ohat$ are given by

$$
\begin{aligned}
    \argmin_{\ohat_i} \|\D\ohat_i - \z_i\|_2^2 + \|\bfGamma_i\ohat_i\|_2^2,
    \quad i = 1, \ldots, r.
\end{aligned}
$$

The following classes solve Tikhonov-regularized least-squares Operator Inference regressions for different choices of the regularization term $\mathcal{R}(\Ohat)$.

| Solver class                     | Description                                      | Regularization $\mathcal{R}(\Ohat)$            |
| :------------------------------- | :----------------------------------------------- | :--------------------------------------------: |
| {class}`L2Solver`                | One scalar regularizer for all $\ohat_i$         | $\lambda^{2}\|\|\Ohat\trp\|\|_F^2$             |
| {class}`L2DecoupledSolver`       | Different scalar regularizers for each $\ohat_i$ | $\sum_{i=1}^{r}\lambda_i^2\|\|\ohat_i\|\|_2^2$ |
| {class}`TikhonovSolver`          | One matrix regularizer for all $\ohat_i$         | $\|\|\bfGamma\Ohat\trp\|\|_F^2$                |
| {class}`TikhonovDecoupledSolver` | Different matrix regularizers for each $\ohat_i$ | $\sum_{i=1}^{r}\|\|\bfGamma_i\ohat_i\|\|_2^2$  |

```{code-cell} ipython3
# Use a single scalar regularizer lambda=1e-6.
l2solver = opinf.lstsq.L2Solver(regularizer=1e-6).fit(D, Z)
print(l2solver)
```

```{code-cell} ipython3
Ohat_L2 = l2solver.solve()
l2solver.residual(Ohat_L2)
```

```{code-cell} ipython3
# Use a different scalar regularizer for each row of the operator matrix.
lambdas = np.logspace(-10, r - 11, r)
l2dsolver = opinf.lstsq.L2DecoupledSolver(regularizer=lambdas)
l2dsolver.fit(D, Z)
print(l2dsolver)
```

```{code-cell} ipython3
Ohat_L2d = l2dsolver.solve()
l2dsolver.residual(Ohat_L2d)
```

```{code-cell} ipython3
# Use a single diagonal matrix regularizer.
gammadiag = np.full(d, 1e-8)
gammadiag[-1] = 1e-4
tiksolver = opinf.lstsq.TikhonovSolver(regularizer=gammadiag)
tiksolver.fit(D, Z)
print(tiksolver)
```

```{code-cell} ipython3
Ohat_tik = tiksolver.solve()
tiksolver.residual(Ohat_tik)
```

```{code-cell} ipython3
# Use a single non-diagonal matrix regularizer.
offdiag = 0.1 * gammadiag[:-1]
Gamma = np.diag(gammadiag) + np.diag(offdiag, k=1) + np.diag(offdiag, k=-1)
tiksolver.regularizer = Gamma
print(tiksolver)
```

```{code-cell} ipython3
Ohat_tik = tiksolver.solve()
tiksolver.residual(Ohat_tik)
```

```{code-cell} ipython3
# Use a different matrix regularizer for each row of the operator matrix.
diags = np.exp(np.random.random((r, d)) - 10)
tikdsolver = opinf.lstsq.TikhonovDecoupledSolver(regularizer=diags)
tikdsolver.fit(D, Z)
print(tikdsolver)
```

## Truncated SVD

+++

The {class}`TruncatedSVDSolver` class approximates the solution to the ordinary least-squares problem {eq}`eq:lstsq:plain` by solving the related problem

$$
\begin{aligned}
    \argmin_{\Ohat}\|\tilde{\D}\Ohat\trp - \Z\trp\|_{F}^{2}
\end{aligned}
$$

where $\tilde{\D}$ is the best rank-$d'$ approximation of $\D$ for some given $d' < \min(k,d)$, i.e.,

$$
\begin{aligned}
    \tilde{D}
    = \argmin_{\D' \in \RR^{k \times d}}
    \|\D' - \D\|_{F}
    \quad\textrm{such that}\quad
    \operatorname{rank}(\D') = d'.
\end{aligned}
$$

This approach is [related to Tikhonov regularization](https://math.stackexchange.com/questions/1084677/tikhonov-regularization-vs-truncated-svd) and is based on the [truncated singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition#Truncated_SVD) of the data matrix $\D$.
The optimization residual is guaranteed to be higher than when using the full SVD as in {class}`PlainSolver`, but the condition number of the truncated SVD system is lower than that of the original system.
Truncation can play a similar role to regularization, but the hyperparameter here (the number of columns to use) is an integer, whereas the regularization hyperparameter $\lambda$ for {class}`L2Solver` may be any positive number.

```{code-cell} ipython3
tsvdsolver = opinf.lstsq.TruncatedSVDSolver(-2)
tsvdsolver.fit(D, Z)
print(tsvdsolver)
```

```{code-cell} ipython3
Ohat = tsvdsolver.solve()
tsvdsolver.residual(Ohat)
```

```{code-cell} ipython3
# Change the number of columns used without recomputing the SVD.
tsvdsolver.num_svdmodes = 8
print(tsvdsolver)
```

```{code-cell} ipython3
tsvdsolver.residual(tsvdsolver.solve())
```

## Total Least-Squares

+++

Linear least-squares models for $\D\Ohat\trp \approx \Z\trp$ assume error in $\Z$ only, i.e.,

$$
\begin{aligned}
    \D\Ohat\trp = \Z\trp + \Delta_{\Z\trp}
\end{aligned}
$$

for some $\Delta_{\Z\trp} \in \RR^{r\times k}$
[Total least-squares](https://en.wikipedia.org/wiki/Total_least_squares) is an alternative approach that assumes possible error in the data matrix $\D$ as well as in $\Z$, i.e.,

$$
\begin{aligned}
    (\D + \Delta_{\D})\Ohat\trp = \Z\trp + \Delta_{\Z\trp}
\end{aligned}
$$

for some $\Delta_{\D}\in\RR^{k \times d}$ and $\Delta_{\Z\trp}\in\RR^{r \times k}$.

The {class}`TotalLeastSquaresSolver` class performs a total least-squares solve for $\Ohat$.

```{code-cell} ipython3
totalsolver = opinf.lstsq.TotalLeastSquaresSolver()
totalsolver.fit(D, Z)
print(totalsolver)
```

```{code-cell} ipython3
Ohat_total = totalsolver.solve()
totalsolver.residual(Ohat_total)
```

## Custom Solvers

+++

New solvers can be defined by inheriting from {class}`SolverTemplate`.
Once implemented, the [`verify()`](SolverTemplate.verify) method may be used to test for consistency between the required methods.

```{code-cell} ipython3
class MySolver(opinf.lstsq.SolverTemplate):
    """Custom solver for the Operator Inference regression."""

    # Constructor -------------------------------------------------------------
    def __init__(self, hyperparameters):
        """Set any solver hyperparameters.
        If there are no hyperparameters, __init__() may be omitted.
        """
        super().__init__()
        # Process / store hyperparameters here.

    # Required methods --------------------------------------------------------
    def solve(self):
        """Solve the regression and return the operator matrix."""
        raise NotImplementedError

    # Optional methods --------------------------------------------------------
    # These may be deleted if not implemented.
    def fit(self, data_matrix, lhs_matrix):
        """Save data matrices and prepare to solve the regression problem.
        This method should do as much work as possible that does not depend on
        the hyperparameters in preparation for solving the regression.
        If there are no hyperparameters, fit() may be omitted.
        """
        super().fit(data_matrix, lhs_matrix)
        # Prepare for regression here.

    def save(self, savefile, overwrite=False):
        """Save the solver to an HDF5 file."""
        raise NotImplementedError

    @classmethod
    def load(cls, loadfile):
        """Load a solver from an HDF5 file."""
        raise NotImplementedError

    def copy(self):
        """Make a copy of the solver.
        If not implemented, copy.deepcopy() is used.
        """
        raise NotImplementedError
```

See {class}`SolverTemplate` for solver attributes and details on the arguments for each method.

+++

### Example: Wrapping a Least-squares Routine

+++

The following class shows how to wrap an existing numerical least-squares solver routine.
Here, we use `scipy.optimize.nnls()`, which solves the least-squares problem with non-negative constraints, i.e., the entries of $\Ohat$ will all be positive.
**This is just a demonstration** -- the entries of a good $\Ohat$ are rarely all positive.

```{code-cell} ipython3
class NNSolver(opinf.lstsq.SolverTemplate):
    """Least-squares solver with non-negativity constraints."""

    def __init__(self, maxiter=None, atol=None):
        """Save keyword arguments for scipy.optimize.nnls()."""
        super().__init__()
        self.options = dict(maxiter=maxiter, atol=atol)

    def solve(self):
        """Solve the regression and return the operator matrix."""
        # Allocate space for the operator matrix entries.
        Ohat = np.empty((self.r, self.d))

        # Solve the nonnegative least-squares for each operator matrix row.
        for i in range(self.r):
            Ohat[i] = opt.nnls(
                self.data_matrix, self.lhs_matrix[i], **self.options
            )[0]

        return Ohat

        # Alternative implementation:
        return np.array(
            [
                opt.nnls(self.data_matrix, z_i, **self.options)[0]
                for z_i in self.lhs_matrix
            ]
        )
```

```{code-cell} ipython3
solver = NNSolver().fit(D, Z)
print(solver)
```

```{code-cell} ipython3
solver.verify()
```

```{code-cell} ipython3
Ohat_nn = solver.solve()
print(f"{Ohat_nn.shape=}")

# Check that the entries of the operator matrix are nonnegative.
print("Minimal entry of Ohat_nn:", Ohat_nn.min())
```