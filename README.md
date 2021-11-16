# HMC Tomography

[![Python version](https://img.shields.io/badge/python-3.7-blue)]() [![GitHub CI Build status](https://github.com/larsgeb/hmclab/workflows/Python%20application/badge.svg)]() [![codecov](https://codecov.io/gh/larsgeb/hmclab/branch/master/graph/badge.svg?token=6svV9YDRhd)](https://codecov.io/gh/larsgeb/hmclab) [![license](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![GitHub releases](https://img.shields.io/badge/download-latest%20release-green.svg)](https://github.com/larsgeb/hmclab/releases/latest)

Manual:
https://larsgeb.github.io/hmclab/

## How to work with this repositories' code

This repository is meant to be an introduction to Bayesian tomography using Hamiltonian Monte Carlo. We designed a general Monte Carlo sampler that is applied to multiple tomographic problems.

## Installing the packge

For full installation instructions, [see here](https://larsgeb.github.io/hmclab/setup.html).

Directly to your environment:

```
pip install -e git+git@github.com:larsgeb/hmclab.git@master#egg=hmclab
```

From the project root directory:

```
pip install -e .
```

### Development dependencies

If you want to develop within this repo, we recommend a few extra packages. They can also be installed using pip.

In Bash:

```
pip install -e git+git@github.com:larsgeb/hmclab.git@master#egg=hmclab[dev] # from github repo
pip install -e .[dev] # from local clone
```

... or Zsh (which requires escapes for brackets):

```
pip install -e git+git@github.com:larsgeb/hmclab.git@master#egg=hmclab\[dev\] # from github repo
pip install -e .\[dev\] # from local clone
```

## Code coverage

We test our code tests for coverage using codecov. The project apge can be found [here](https://codecov.io/gh/larsgeb/hmclab).

Codecov graph follows below. An interactive version can be found on the codecov project page.

![codecov graph](https://codecov.io/gh/larsgeb/hmclab/graphs/sunburst.svg?token=6svV9YDRhd)

> The inner-most circle is the entire project, moving away from the center are folders then, finally, a single file. The size and color of each slice is representing the number of statements and the coverage, respectively.
