# HMC Lab
## Andrea Zunino, Lars Gebraad, Andreas Fichtner

[![codecov](https://codecov.io/gh/larsgeb/hmclab/branch/master/graph/badge.svg?token=6svV9YDRhd)](https://codecov.io/gh/larsgeb/hmclab) [![license](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![GitHub releases](https://img.shields.io/badge/download-latest%20release-green.svg)](https://github.com/larsgeb/hmclab/releases/latest)

**HMC Lab** is a numerical laboratory for research in Bayesian seismology, written in Python and Julia. Jump to [Docker one-command setup](#docker-one-command-setup).

- **Website:** https://hmclab.science
- **Python documentation:** https://python.hmclab.science
- **Source code:** https://github.com/larsgeb/hmclab
- **Docker image:** https://hub.docker.com/repository/docker/larsgebraad/hmclab
- **Bug reports:** https://github.com/larsgeb/hmclab/issues

It provides all the ingredients to set up probabilistic (and deterministic) inverse
problems, appraise them, and analyse them. This includes a plethora of prior
distributions, different physical modelling modules and various MCMC (and
other) algorithms. 

In particular it provides prior distributions, physics and appraisal algorithms.

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

# <a name="docker"></a>Docker one-command setup

To get staerting with the tutorial and example notebooks, one can use a single command
in Docker. This will pull a Docker image based on the Jupyter Datascience stack. The
final container is approximately 5GB.

```bash
    docker run -p 9123:9123 larsgebraad/hmclab \
    start-notebook.sh --NotebookApp.token='hmclab'  \
    --NotebookApp.port='9123' --LabApp.default_url='/lab/tree/Home.ipynb'
```

Then either copy-past the link from your terminal, or navigate manually to [http://127.0.0.1:9123/lab/tree/Home.ipynb?token=hmclab](http://127.0.0.1:9123/lab/tree/Home.ipynb?token=hmclab).

# Online tutorial notebooks

All tutorial notebooks can also be accessed online in a non-interactive fashion. Simply 
use https://python.hmclab.science or use the following links:

**Tutorials:**

- [Getting started.ipynb](notebooks/tutorials/0%20-%20Getting%20started.ipynb)
- [Tuning Hamiltonian Monte Carlo.ipynb](notebooks/tutorials/1%20-%20Tuning%20Hamiltonian%20Monte%20Carlo.ipynb)
- [Separate priors per dimension.ipynb](notebooks/tutorials/2%20-%20Separate%20priors%20per%20dimension.ipynb)
- [Creating your own inverse problem.ipynb](notebooks/tutorials/3%20-%20Creating%20your%20own%20inverse%20problem.ipynb)
- [Running parallel Markov chains.ipynb](notebooks/tutorials/4%20-%20Running%20parallel%20Markov%20chains.ipynb)

**Demos:**

- [Sampling linear equations](notebooks/examples/Sampling%20linear%20equations.ipynb)
- [Sampling sparse linear equations](notebooks/examples/Sampling%20sparse%20linear%20equations.ipynb)
- [Locating quakes on Grimsvötn, Iceland](notebooks/examples/Locating%20quakes%20on%20Grimsvötn%2C%20Iceland.ipynb)
- [Elastic 2d FWI](notebooks/examples/Elastic%202d%20FWI.ipynb)


# The long way around: installing the package on your system

For full installation instructions, including creating a proper Python environment, [see the installation instructions](https://python.hmclab.science/setup.html). 

Start with making sure that you have HDF5 or h5py installed properly.

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

