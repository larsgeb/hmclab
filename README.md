# HMC Tomography

[![Python version](https://img.shields.io/badge/python-3.7-blue)]() [![GitHub CI Build status](https://github.com/larsgeb/hmclab/workflows/Python%20application/badge.svg)]() [![codecov](https://codecov.io/gh/larsgeb/hmclab/branch/master/graph/badge.svg?token=6svV9YDRhd)](https://codecov.io/gh/larsgeb/hmclab) [![license](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![GitHub releases](https://img.shields.io/badge/download-latest%20release-green.svg)](https://github.com/larsgeb/hmclab/releases/latest)


**HMCLab** is a numerical laboratory for Monte Carlo sampling and other algorithms
in seismological and geophysical research. We provide all the ingredients to 
set up probabilistic (and deterministic) inverse problems, appraise them, and
analyse them. This includes a plethora of prior distributions, different 
physical modelling modules and various MCMC (and other) algorithms. 

Here is a partial inventory of what we provide:

**Prior distributions:**
- Normal
- Laplace
- Uniform
- Arbitrary composites of other priors
- Bayes rule
- User supplied distributions

**Physics:**
- Linear equations
- Straight ray tomography
- 3d source location
- 2d elastic full-waveform inversion
- User supplied physics

**Algorithms:**
- Hamiltonian Monte Carlo (and variations)
- Random Walk Metropolis Hastings
- Stein Variational Gradient Descent
- Gradient descent
- Interfaces to non-linear optimization methods from SciPy
- Animated versions of various algorithms

**Tutorials:**

0. [Getting started.ipynb](notebooks/tutorials/0%20-%20Getting%20started.ipynb)
1. [Tuning Hamiltonian Monte Carlo.ipynb](notebooks/tutorials/1%20-%20Tuning%20Hamiltonian%20Monte%20Carlo.ipynb)
2. [Separate priors per dimension.ipynb](notebooks/tutorials/2%20-%20Separate%20priors%20per%20dimension.ipynb)
3. [Creating your own inverse problem.ipynb](notebooks/tutorials/3%20-%20Creating%20your%20own%20inverse%20problem.ipynb)
4. [Running parallel Markov chains.ipynb](notebooks/tutorials/4%20-%20Running%20parallel%20Markov%20chains.ipynb)

**Demos:**
1. [Elastic 2d FWI](notebooks/examples/Elastic%202d%20FWI.ipynb)
2. [Locating quakes on Grimsvötn, Iceland](notebooks/examples/Locating%20quakes%20on%20Grimsvötn,%20Iceland.%20ipynb)
3. [Sampling linear equations](notebooks/examples/Sampling%20linear%20equations.ipynb)
4. [Sampling sparse linear equations](notebooks/examples/Sampling%20sparse%20linear%20equations.ipynb)

Manual:
https://larsgeb.github.io/hmclab/



## A flying start: Jupyter notebook server in a Docker with all dependencies

We build two docker images: one for AMD64 (should work on most Linux/Windows/MacOS intel systems), and ARM64 (should work on Mac M1). You can automatically get the appropriate version and start the notebook server by running:

```bash
docker run -p 8888:8888 larsgebraad/hmclab
```

You can then immediately go into your webbrowser to the address
`localhost:8888` to access the tutorial and demo notebooks.

If you have another service running at port 8888, modify the command and web
address like this:

```bash
docker run -p PORT:8888 larsgebraad/hmclab
localhost:PORT
```


## The long way around: installing the packge on your system

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

