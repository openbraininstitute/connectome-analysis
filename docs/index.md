# Connectome Analysis

## General analyses of connectomes from a topological perspective

![Connectome Analysis banner](banner_BPP_connalysis.jpg)

## Overview

This package provides a library of general functions to analyze connectomes. Functions are divided into three groups:

* [Modelling](modelling.md): Functions to model (parametrize) the connectivity of connectomes
* [Randomization](randomization.md): Generation of randomized controls of connectomes
* [Network](network.md): Network analyses based on metrics of different types

## Tutorials

Check out a few short tutorials showing:

* How to implement most [unweighted topological network metrics](https://github.com/openbraininstitute/connectome-analysis/blob/master/tutorials/TDA_unweighted_networks.ipynb).
* How to compute [triad counts](https://github.com/openbraininstitute/connectome-analysis/blob/master/tutorials/counting_triads.ipynb).
* How to [model and extract](https://github.com/openbraininstitute/connectome-analysis/blob/master/tutorials/modelling.ipynb) distance dependent parameters from a connectome with a geometric embedding.
* How to generate [randomized controls of a given connectome](https://github.com/openbraininstitute/connectome-analysis/blob/master/tutorials/randomization.ipynb).

## Installation

To install, run the following command in your terminal:

```console
pip install git+https://github.com/openbraininstitute/connectome-analysis.git
```

For development installation requirements, see the [project README](https://github.com/openbraininstitute/connectome-analysis#development-installation).

```{toctree}
:maxdepth: 2
:caption: Contents:

modelling
randomization
network
```

