# What is Operator Inference?

The goal of Operator Inference (OpInf) is to construct a low-dimensional, computationally inexpensive system whose solutions are close to those of some high-dimensional system for which we have 1) training data and 2) some knowledge about the system structure.
The main steps are the following.

1.  [**Get training data**](subsec-training-data). Gather and [preprocess](../api/pre.ipynb) high-dimensional data to learn a low-dimensional model from. This package has a few common preprocessing tools, but the user must bring the data to the table.
2.  [**Compute a low-dimensional state representation**](subsec-basis-computation). Approximate the high-dimensional data with only a few degrees of freedom. The simplest approach is to take the SVD of the high-dimensional training data, extract the first few left singular vectors, and use these vectors as a new coordinate basis.
3.  [**Set up and solve a low-dimensional regression**](subsec-opinf-regression). Use the low-dimensional representation of the training data to determine a reduced-order model (ROM) that best fits the data in a minimum-residual sense. This is the core objective of the package.
4.  [**Solve the reduced-order model**](subsec-rom-evaluation). Use the learned model to make computationally efficient predictions.

:::{figure} ../../images/summary.svg
---
width: 80 %
---
The OpInf workflow.
:::

This page gives an overview of OpInf by walking through each of these steps.

## Problem Statement

::::{margin}
:::{note}
We are most often interested in spatial discretizations of partial differential equations (PDEs), but {eq}`eq:opinf-example-fom` does not necessarily have to be related to a PDE.
:::
::::

Consider a system of ordinary differential equations (ODEs) with state $\q(t)\in\RR^{n}$ and inputs $\u(t)\in\RR^{m}$,

$$
\begin{aligned}
    \ddt\q(t) = \f(t, \q(t), \u(t)),
\end{aligned}
$$ (eq:opinf-example-fom)

where $\f:\RR\times\RR^n\times\RR^m\to\RR^n$.
We call {eq}`eq:opinf-example-fom` the _full-order model_ (FOM).
Given samples of the state $\q(t)$ and corresponding inputs $\u(t)$, OpInf learns a surrogate system for {eq}`eq:opinf-example-fom` with the much smaller state $\qhat(t) \in \RR^{r}, r \ll n,$ and a polynomial structure, for example:

::::{margin}
:::{note}
The $\otimes$ operator is called the [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product).
:::
::::

$$
\begin{aligned}
    \ddt\qhat(t)
    = \chat
    + \Ahat\qhat(t)
    + \Hhat[\qhat(t)\otimes\qhat(t)]
    + \Bhat\u(t).
\end{aligned}
$$ (eq:opinf-example-rom)

We call {eq}`eq:opinf-example-rom` a _reduced-order model_ (ROM) for {eq}`eq:opinf-example-fom`.
Our goal is to infer the _reduced-order operators_ $\chat \in \RR^{r}$, $\Ahat\in\RR^{r\times r}$, $\Hhat\in\RR^{r\times r^{2}}$, and/or $\Bhat\in\RR^{r\times m}$ using data from {eq}`eq:opinf-example-fom`.
The user specifies which terms to include in the model.

::::{important}
:name: projection-preserves-structure

The right-hand side of the ROM {eq}`eq:opinf-example-rom` is a polynomial with respect to the state $\qhat(t)$:
$\chat$ are the constant terms, $\Ahat\qhat(t)$ are the linear terms, $\Hhat[\qhat(t)\otimes\qhat(t)]$ are the quadratic terms, with input terms $\Bhat\u(t)$.
The user must choose which terms to include in the ROM, and this choice should be motivated by the structure of the FOM {eq}`eq:opinf-example-fom` and the way that the reduced-order state $\qhat(t)$ approximates the full-order state $\q(t)$.

For example, suppose the FOM is a linear time-invariant (LTI) system, written as

$$
\begin{aligned}
    \ddt\q(t)
    = \A\q(t) + \B\u(t)
\end{aligned}
$$

for some $\A\in\RR^{n\times n}$ and $\B\in\RR^{n\times m}$.
Let $\Vr\in\RR^{n\times r}$ be a matrix with orthonormal columns and suppose we approximate the FOM state with the linear relationship $\q(t) \approx \Vr\qhat(t)$.
Then the ROM should mirror the LTI structure of the FOM:

$$
\begin{aligned}
    \ddt\qhat(t)
    = \Ahat\qhat(t) + \Bhat\u(t),
\end{aligned}
$$

with $\Ahat\in\RR^{r\times r}$ and $\Bhat\in\RR^{r\times m}$.

<!-- Alternatively, if we use the approximation $\q(t) \approx \Vr\qhat(t) + \bar{\q}$ for some fixed $\bar{\q}\in\RR^n$, then the ROM should have a slightly modified structure:

$$
\begin{aligned}
    \ddt\qhat(t)
    = \chat + \Ahat\qhat(t) + \Bhat\u(t),
\end{aligned}
$$

where $\chat\in\RR^{r}$. -->

:::{dropdown} Motivation
**Projection preserves polynomial structure** {cite}`benner2015pmorsurvey,peherstorfer2016opinf`.
The classical (Galerkin) projection-based ROM for {eq}`eq:opinf-example-rom` is obtained by substituting $\q(t)$ with its low-dimensional approximation and enforcing an orthogonality condition.

For example, if $\q(t) \approx \Vr\qhat(t)$, then the Galerkin ROM is

$$
\begin{aligned}
    \ddt\qhat(t) = \Vr\trp\f(t, \Vr\qhat(t), \u(t)).
\end{aligned}
$$

Now suppose that $\f$ can be written in a polynomial form,

$$
\begin{aligned}
    \f(t,\q(t),\u(t))
    = \c + \A\q(t) + \H[\q(t)\otimes\q(t)] + \B\u(t),
\end{aligned}
$$

where $\c\in\RR^{n}$, $\A\in\RR^{n\times n}$, $\H\in\RR^{n\times n^2}$, and $\B\in\RR^{r\times m}$.
We then have

$$
\begin{aligned}
    \Vr\trp\f(t, \Vr\qhat(t), \u(t))
    = \Vr\trp\left(\c + \A\Vr\qhat(t) + \H[(\Vr\qhat(t))\otimes(\Vr\qhat(t))] + \B\u(t)\right).
\end{aligned}
$$

Since $\Vr\trp\Vr$ is the identity matrix and using $(\mathbf{XY})\otimes(\mathbf{ZW}) = (\mathbf{X}\otimes \mathbf{Z})(\mathbf{Y}\otimes\mathbf{W})$, this reduces to

$$
\begin{aligned}
    \Vr\trp\f(t, \Vr\qhat(t), \u(t))
    = \chat + \Ahat\qhat(t) + \Hhat[\qhat(t)\otimes\qhat(t)] + \Bhat\u(t),
\end{aligned}
$$

where $\chat = \Vr\trp\c$, $\Ahat = \Vr\trp\A\Vr$, $\Hhat = \Vr\trp\H\left(\Vr\otimes\Vr\trp\right)$, and $\Bhat = \Vr\trp\B$.
Hence, the Galerkin ROM has the same polynomial form as $\f$.

OpInf aims to learn appropriate reduced-order operators $\chat$, $\Ahat$, $\Hhat$, and/or $\Bhat$ _from data_ and is therefore useful for situations where $\c$, $\A$, $\H$, and/or $\B$ are not explicitly available for matrix computations.
:::
::::

(subsec-training-data)=
## Get Training Data

OpInf learns ROMs from full-order state/input data.
Start by gathering solution and input data and organizing them columnwise into the _state snapshot matrix_ $\Q$ and _input matrix_ $\U$,

$$
\begin{aligned}
    \Q
    &= \left[\begin{array}{cccc}
        & & & \\
        \q_{1} & \q_{2} & \cdots & \q_{k}
        \\ & & &
    \end{array}\right]
    \in \RR^{n \times k},
    &
    \U
    &= \left[\begin{array}{cccc}
        & & & \\
        \u_{1} & \u_{2} & \cdots & \u_{k}
        \\ & & &
    \end{array}\right]
    \in \RR^{m \times k},
\end{aligned}
$$

where $n$ is the dimension of the (discretized) state, $m$ is the dimension of the input, $k$ is the number of available data points, and the columns of $\Q$ and $\U$ are the solution to the FOM at some time $t_j$:

$$
\begin{aligned}
    \ddt\q(t)\bigg|_{t = t_j}
    = \f(t_{j}, \q_{j}, \u_{j}).
\end{aligned}
$$

:::{tip}
Raw dynamical systems data often needs to be lightly preprocessed before it can be used in OpInf.
Preprocessing can promote stability in the inference of the reduced-order operators and improve the stability and accuracy of the resulting ROM {eq}`eq:opinf-example-rom`.
Common preprocessing steps include
1. Variable transformations / lifting to induce a polynomial structure.
2. Centering or shifting data to account for boundary conditions.
3. Scaling / nondimensionalizing the variables represented in the state.

See {mod}`opinf.lift` and {mod}`opinf.pre` for details and examples.
:::

OpInf uses a regression problem to compute the reduced-order operators, which requires state data ($\Q$), input data ($\U$), as well as data for the corresponding time derivatives:

$$
\begin{aligned}
    \dot{\Q}
    = \left[\begin{array}{cccc}
        & & & \\
        \dot{\q}_{1} &
        \dot{\q}_{2} & \cdots &
        \dot{\q}_{k}
        \\ & & &
    \end{array}\right]
    \in \RR^{n \times k},
    \qquad
    \dot{\q}_{j} = \ddt\q(t)\bigg|_{t = t_j} \in \RR^{n}.
\end{aligned}
$$

:::{note}
If the time derivatives cannot be computed directly by evaluating $\f(t_{j}, \q_{j}, \u_{j})$, in some cases they may be inferred from the state snapshots.
The simplest approach is to use [finite differences](https://en.wikipedia.org/wiki/Numerical_differentiation) of the state snapshots, see {mod}`opinf.ddt`.
:::

:::{warning}
If you do any preprocessing on the states, be sure to use the time derivatives of the _processed states_, not of the original states.
:::

<!-- :::{note}
OpInf can also be used to learn discrete dynamical systems with polynomial structure, for example,

$$
    \q_{j+1}
    = \A\q_{j}
    + \H(\q_{j}\otimes\q_{j})
    + \B\u_{j}.
$$

In this case, the left-hand side data is a simply subset of the state snapshot matrix.
::: -->
<!-- See TODO for more details. -->

(subsec-basis-computation)=
## Compute a Low-dimensional State Representation

The purpose of learning a ROM is to achieve a computational speedup.
This is accomplished by introducing an approximate representation of the $n$-dimensional state using only $r \ll n$ degrees of freedom.
The most common approach is to represent the state as a linear combination of $r$ vectors:

$$
    \q(t)
    \approx \Vr \qhat(t)
    = \sum_{i=1}^{r}\v_{i}\hat{q}_{i}(t),
$$ (eq-opinf-basis-def)

where

$$
    \Vr
    = \left[\begin{array}{ccc}
        & & \\
        \v_{1} & \cdots & \v_{r}
        \\ & &
    \end{array}\right] \in \RR^{n \times r},
    \qquad
    \qhat
    = \left[\begin{array}{c}
        \hat{q}_{1}(t) \\ \vdots \\ \hat{q}_{r}(t)
    \end{array}\right] \in \RR^{r}.
$$

We call $\Vr \in \RR^{n \times r}$ the _basis matrix_ and typically require that it have orthonormal columns.
The basis matrix is the link between the high-dimensional state space of the full-order model {eq}`eq:opinf-example-fom` and the low-dimensional state space of the ROM {eq}`eq:opinf-example-rom`.
See {mod}`opinf.basis` for tools to compute the basis $\Vr\in\RR^{n \times r}$ and select an appropriate dimension $r$.

(subsec-opinf-regression)=
## Set up and Solve a Low-dimensional Regression

OpInf determines the operators $\chat$, $\Ahat$, $\Hhat$, and/or $\Bhat$ by solving the following data-driven regression:

$$
\begin{aligned}
    \min_{\chat,\Ahat,\Hhat,\Bhat}\sum_{j=0}^{k-1}\left\|
        \chat
        + \Ahat\qhat_{j}
        + \Hhat[\qhat_{j} \otimes \qhat_{j}]
        + \Bhat\u_{j}
        - \dot{\qhat}_{j}
    \right\|_{2}^{2}
    + \mathcal{R}(\chat,\Ahat,\Hhat,\Bhat),
\end{aligned}
$$ (eq:opinf-lstsq-residual)

where
+ $\qhat_{j} = \Vr\trp\q(t_{j})$ is the state at time $t_{j}$ represented in the coordinates of the basis,
+ $\dot{\qhat}_{j} = \ddt\Vr\trp\q(t)\big|_{t=t_{j}}$ is the time derivative of the state at time $t_{j}$ in the coordinates of the basis,
+ $\u_{j} = \u(t_j)$ is the input at time $t_{j}$, and
+ $\mathcal{R}$ is a _regularization term_ that penalizes the entries of the learned operators.

The least-squares minimization {eq}`eq:opinf-lstsq-residual` can be written in the more standard form

$$
\min_{\Ohat}\left\|
    \D\Ohat\trp - \mathbf{Z}\trp
\right\|_{F}^{2} + \mathcal{R}(\Ohat),
$$

where

::::{margin}
:::{note}
The $\odot$ operator is called the [Khatri-Rao product](https://en.wikipedia.org/wiki/Khatri%E2%80%93Rao_product#Column-wise_Kronecker_product) and indicates taking the Kronecker product column by column.
<!-- $$
\left[\begin{array}{cccc}
    \mathbf{z}_{0} & \mathbf{z}_{1} & \cdots & \mathbf{z}_{k-1}
\end{array}\right]
\odot
\left[\begin{array}{cccc}
    \mathbf{w}_{0} & \mathbf{w}_{1} & \cdots & \mathbf{w}_{k-1}
\end{array}\right]
= \left[\begin{array}{cccc}
    \mathbf{z}_{0}\otimes\mathbf{w}_{0} & \mathbf{z}_{1}\otimes\mathbf{w}_{1} & \cdots & \mathbf{z}_{k-1}\otimes\mathbf{w}_{k-1}
\end{array}\right].
$$ -->
:::
::::

$$
\begin{aligned}
    \Ohat
    &= \left[~\chat~~\Ahat~~\Hhat~~\Bhat~\right]\in\RR^{r\times d(r,m)},
    &\text{(unknown operators)}
    \\
    \D
    &= \left[~\mathbf{1}_{k}~~\Qhat\trp~~(\Qhat\odot\Qhat)\trp~~\U\trp~\right]\in\RR^{k\times d(r,m)},
    &\text{(known data)}
    \\
    \Qhat
    &= \left[~\qhat_0~~\qhat_1~~\cdots~~\qhat_{k-1}~\right]\in\RR^{r\times k}
    &\text{(snapshots)}
    \\
    \mathbf{Z}
    &= \left[~\dot{\qhat}_0~~\dot{\qhat}_1~~\cdots~~\dot{\qhat}_{k-1}~\right]\in\RR^{r\times k},
    &\text{(time derivatives)}
    \\
    \U
    &= \left[~\u_0~\u_1~\cdots~\u_{k-1}~\right]\in\RR^{m\times k},
    &\text{(inputs)}
\end{aligned}
$$

in which $d(r,m) = 1 + r + r(r+1)/2 + m$ and $\mathbf{1}_{k}\in\RR^{k}$ is a vector of ones.

:::{dropdown} Derivation
The Frobenius norm of a matrix is the square root of the sum of the squared entries.
If $\mathbf{Z}$ has entries $z_{ij}$, then

$$
\begin{aligned}
    \|\mathbf{Z}\|_{F}
    = \sqrt{\text{trace}(\mathbf{Z}\trp\mathbf{Z})}
    = \sqrt{\sum_{i,j}z_{ij}^{2}}.
\end{aligned}
$$

Writing $\mathbf{Z}$ in terms of its columns,

$$
\begin{aligned}
    \mathbf{Z}
    = \left[\begin{array}{c|c|c|c}
        &&& \\
        \mathbf{z}_0 & \mathbf{z}_1 & \cdots & \mathbf{z}_{k-1} \\
        &&&
    \end{array}\right],
\end{aligned}
$$

we have

$$
\begin{aligned}
    \|\mathbf{Z}\|_F^2 = \sum_{j=0}^{k-1}\|\mathbf{z}_j\|_2^2.
\end{aligned}
$$

Furthermore, $\|\mathbf{Z}\|_{F} = \|\mathbf{Z}\trp\|_{F}$.
Using these two properties, we can rewrite the least-squares residual as follows:

$$
\begin{aligned}
    \sum_{j=0}^{k-1}\left\|
        \chat
        + \Ahat\qhat_{j}
        + \Hhat[\qhat_{j} \otimes \qhat_{j}]
        + \Bhat\u_{j}
        - \dot{\qhat}_{j}
    \right\|_{2}^{2}
    &= \left\|
        \chat\mathbf{1}\trp
        + \Ahat\Qhat
        + \Hhat[\Qhat \odot \Qhat]
        + \Bhat\U
        - \dot{\Qhat}
    \right\|_{F}^{2}
    \\
    &= \left\|
        \left[\begin{array}{cccc}
            \chat & \Ahat & \Hhat & \Bhat
        \end{array}\right]
        \left[\begin{array}{c}
            \mathbf{1}\trp
            \\ \Qhat
            \\ \Qhat \odot \Qhat
            \\ \U
        \end{array}\right]
        - \dot{\Qhat}
    \right\|_{F}^{2}
    \\
    &= \left\|
        \left[\begin{array}{c}
            \mathbf{1}\trp
            \\ \Qhat
            \\ \Qhat \odot \Qhat
            \\ \U
        \end{array}\right]\trp
        \left[\begin{array}{cccc}
            \chat & \Ahat & \Hhat & \Bhat
        \end{array}\right]\trp
        - \dot{\Qhat}\trp
    \right\|_{F}^{2}
    \\
    &= \left\|
        \left[\begin{array}{cccc}
            \mathbf{1}
            & \Qhat\trp
            & [\Qhat \odot \Qhat]\trp
            & \U\trp
        \end{array}\right]
        \left[\begin{array}{c}
            \chat\trp
            \\ \Ahat\trp
            \\ \Hhat\trp
            \\ \Bhat\trp
        \end{array}\right]
        - \dot{\Qhat}\trp
    \right\|_{F}^{2},
\end{aligned}
$$

which is the standard form given above.
:::

:::{important}
For the most common choices of $\mathcal{R}$, the OpInf learning problem {eq}`eq:opinf-lstsq-residual` has a unique solution if and only if the data matrix $\D$ has full column rank.
A necessary condition for this to happen is $k \ge d(r,m)$, that is, the number of training snapshots $k$ should exceed the number of reduced-order operator entries to be learned for each system mode.
If you are experiencing poor performance with OpInf ROMs, try decreasing $r$, increasing $k$, or adding a regularization term to improve the conditioning of the learning problem.
:::

:::{note}
Let $\ohat_{1},\ldots,\ohat_{r}\in\RR^{d(r,m)}$ be the rows of $\Ohat$.
If the regularization can be written as

$$
\begin{align*}
    \mathcal{R}(\Ohat)
    = \sum_{i=1}^{r}\mathcal{R}_{i}(\ohat_{i}),
\end{align*}
$$

then the OpInf regression decouples along the rows of $\Ohat$ into $r$ independent least-squares problems:

$$
\begin{align*}
    \min_{\Ohat}\left\{\left\|
        \D\Ohat\trp - \mathbf{Y}\trp
    \right\|_{F}^{2} + \mathcal{R}(\Ohat)\right\}
    =
    \sum_{i=1}^{r}\min_{\ohat_{i}}\left\{\left\|
        \D\ohat_{i} - \mathbf{y}_{i}
    \right\|_{2}^{2} + \mathcal{R}_{i}(\ohat_{i})\right\},
\end{align*}
$$

where $\mathbf{y}_{i},\ldots,\mathbf{y}_{r}$ are the rows of $\mathbf{Y}$.
:::

(subsec-rom-evaluation)=
## Solve the Reduced-order Model


Once the reduced-order operators have been determined, the corresponding ROM {eq}`eq:opinf-example-rom` can be solved rapidly to make predictions.
The computational cost of solving the ROM scales with $r$, the number of degrees of freedom in the low-dimensional representation of the state.

For example, we may use the ROM to obtain approximate solutions of the FOM {eq}`eq:opinf-example-fom` with
- new initial conditions $\q_{0}$,
- a different input function $\u(t)$,
- a longer time horizon than the training data,
- different system parameters.

:::{important}
The accuracy of any data-driven model depends on how well the training data represents the full-order system.
We should not expect a ROM to perform well under conditions that are wildly different than the training data.
The [**Getting Started**](../tutorials/basics.ipynb) tutorial demonstrates this concept in the context of prediction for new initial conditions.
:::

## Brief Example

Suppose we have the following variables.

| Variable | Symbol                 | Description                               |
| :------- | :--------------------- | :---------------------------------------- |
| `Q`      | $\Q\in\RR^{n\times k}$ | State snapshot matrix                     |
| `t`      | $t$                    | Time domain for snapshot data             |
| `u`      | $\u:\RR\to\RR^{m}$     | Input function                            |

That is, `Q[:, j]` is the state corresponding to time `t[j]` and `u` is a function handle describing the input.
Then the following code learns a ROM of the form

$$
\begin{aligned}
    \ddt\qhat(t)
    = \Ahat\qhat(t) + \Hhat(\qhat(t)\otimes\qhat(t)) + \Bhat\u(t)
\end{aligned}
$$

from the training data, uses the ROM to reconstruct the training data, and computes the error of the reconstruction.

```python
import opinf

# Set up and calibrate a ROM with a rank-10 basis and quadratic model structure.
rom = opinf.ROM(
    basis=opinf.basis.PODBasis(num_vectors=10),
    ddt_estimator=opinf.ddt.UniformFiniteDifferencer(t),
    model=opinf.models.ContinuousModel(
        operators="AHB",
        solver=opinf.lstsq.L2Solver(regularizer=1e-6),
    )
).fit(states=Q, inputs=u(t))

# Solve the ROM over a specified time domain.
Q_rom = rom.predict(Q[:, 0], t, input_func=u)

# Compute the error of the ROM prediction.
absolute_error, relative_error = opinf.post.Lp_error(Q, Q_rom, t)
```

The [API pages](../api/main.md) for each submodule provide details on the arguments to the `ROM` class.
See [**Getting Started**](../tutorials/basics.ipynb) for an introductory tutorial.