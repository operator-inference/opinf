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

Get started with [**What is Operator Inference?**](./opinf/intro.md) or head straight to [**Installation**](./opinf/installation.md) and the first tutorial, [**Getting Started**](./tutorials/basics.md).
See [**Literature**](./opinf/literature.md) for a list of scholarly works on operator inference.

:::{image} ./_static/summary.svg
:align: center
:width: 80 %
:::

---

## Contents

# Operator Inference

```{toctree}
:maxdepth: 1
:caption: Operator Inference

opinf/intro
opinf/installation
opinf/changelog
opinf/literature
opinf/bibliography
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

tutorials/basics
tutorials/inputs
tutorials/parametric
```

```{toctree}
:maxdepth: 1
:caption: API Reference

api/main
api/lift
api/pre
api/basis
api/ddt
api/operators
api/lstsq
api/models
api/roms
api/post
api/utils
```

```{toctree}
:maxdepth: 1
:caption: Developer Guide

contributing/how_to_contribute
contributing/testing
contributing/documentation
contributing/notation
```