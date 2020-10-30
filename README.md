# HMC Tomography

![GitHub CI Build status](https://github.com/larsgeb/hmc-tomography/workflows/Python%20application/badge.svg) [![Travis CI Build status](https://travis-ci.com/larsgeb/hmc-tomography.svg?token=G43u7wF834znRn3jm2mR&branch=master)](https://travis-ci.com/larsgeb/hmc-tomography) [![codecov](https://codecov.io/gh/larsgeb/hmc-tomography/branch/master/graph/badge.svg?token=6svV9YDRhd)](https://codecov.io/gh/larsgeb/hmc-tomography) ![GitHub](https://img.shields.io/github/license/larsgeb/hmc-tomography) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![GitHub Pre-Releases](https://img.shields.io/github/downloads/larsgeb/hmc-tomography/latest/total)](https://github.com/larsgeb/hmc-tomography/releases/latest)

Manual:
https://larsgeb.github.io/hmc-tomography/

## How to work with this repositories' code

This repository is meant to be an introduction to Bayesian tomography using Hamiltonian Monte Carlo. We designed a general Monte Carlo sampler that is applied to multiple tomographic problems.

## Installing the packge

For full installation instructions, [see here](https://larsgeb.github.io/hmc-tomography/setup.html).

Directly to your environment:

```
pip install -e git+git@github.com:larsgeb/hmc-tomography.git@master#egg=hmc_tomography
```

From the project root directory:

```
pip install -e .
```

### Development dependencies

If you want to develop within this repo, we recommend a few extra packages. They can also be installed using pip.

In Bash:

```
pip install -e git+git@github.com:larsgeb/hmc-tomography.git@master#egg=hmc_tomography[dev] # from github repo
pip install -e .[dev] # from local clone
```

... or Zsh (which requires escapes for brackets):

```
pip install -e git+git@github.com:larsgeb/hmc-tomography.git@master#egg=hmc_tomography\[dev\] # from github repo
pip install -e .\[dev\] # from local clone
```

## Integration tests

We test our code using TravisCI on as many platforms and Python versions as possible. Currently, we are testing the following configurations:

| Testing environments | Python 3.7                                                                                                                                                                                                                                                                     |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Windows              | ![](https://badges.herokuapp.com/travis.com/larsgeb/hmc-tomography?branch=master&env=OS_PY=windows37&label=Windows%20-%20Python%203.7)                                                                                                                                         |
| Ubuntu               | ![](https://badges.herokuapp.com/travis.com/larsgeb/hmc-tomography?branch=master&env=OS_PY=bionic37&label=Bionic%20-%20Python%203.7) <br> ![](https://badges.herokuapp.com/travis.com/larsgeb/hmc-tomography?branch=master&env=OS_PY=xenial37&label=Xenial%20-%20Python%203.7) |
| macOS                | ![](https://badges.herokuapp.com/travis.com/larsgeb/hmc-tomography?branch=master&env=OS_PY=osx37&label=macOS%20-%20xcode10.2%20-%20Python%203.7)                                                                                                                               |

## Code coverage

We test our code tests for coverage using codecov. The project apge can be found [here](https://codecov.io/gh/larsgeb/hmc-tomography).

Codecov graph follows below. An interactive version can be found on the codecov project page.

![codecov graph](https://codecov.io/gh/larsgeb/hmc-tomography/graphs/sunburst.svg?token=6svV9YDRhd)

> The inner-most circle is the entire project, moving away from the center are folders then, finally, a single file. The size and color of each slice is representing the number of statements and the coverage, respectively.
