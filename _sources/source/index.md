# Operator Inference in Python

[![License](https://img.shields.io/github/license/operator-inference/opinf)](https://github.com/operator-inference/opinf/blob/main/LICENSE)
[![Top language](https://img.shields.io/github/languages/top/operator-inference/opinf)](https://www.python.org)
![Code size](https://img.shields.io/github/languages/code-size/operator-inference/opinf)
[![Issues](https://img.shields.io/github/issues/operator-inference/opinf)](https://github.com/operator-inference/opinf/issues)
[![Latest commit](https://img.shields.io/github/last-commit/operator-inference/opinf)](https://github.com/operator-inference/opinf/commits/main)
[![PyPI](https://img.shields.io/pypi/wheel/opinf)](https://pypi.org/project/opinf/)

:::{attention}
This documentation is for `opinf` version `0.6.0`.
The `opinf` package is a research code that is still in rapid development.
New versions may introduce substantial new features or API adjustments.
See updates and notes for old versions [here](./opinf/changelog.md).
:::

This package is a Python implementation of Operator Inference (OpInf), a projection-based model reduction technique for learning polynomial reduced-order models of dynamical systems.
The procedure is data-driven and non-intrusive, making it a viable candidate for model reduction of "glass-box" systems where the structure of the governing equations is known but intrusive code queries are unavailable.

Get started with [**What is Operator Inference?**](./opinf/intro.ipynb) or head straight to [**Installation**](./opinf/installation.md) and the first tutorial, [**Getting Started**](./tutorials/basics.ipynb).
See [**Literature**](./opinf/literature.md) for a list of scholarly works on operator inference.

:::{image} ../images/summary.svg
:align: center
:width: 80 %
:::

---

## Contents

```{tableofcontents}
```
