# HMC Tomography

TravisCI unit test status: 
[![Build Status](https://travis-ci.com/larsgeb/hmc-tomography.svg?token=G43u7wF834znRn3jm2mR&branch=master)](https://travis-ci.com/larsgeb/hmc-tomography)

Manual:
https://larsgeb.github.io/hmc-tomography/

## How to work with this repositories' code

This repository is meant to be an introduction to Bayesian tomography using Hamiltonian Monte Carlo. We designed a general Monte Carlo sampler that is applied to multiple tomographic problems.

To make the code run machine independent, we use a virtual environment. This makes sure that everyone uses the same Python version and packages. Here we use Conda (Anaconda or Miniconda are both supported). Make sure that you have Conda installed: 

1.  https://www.anaconda.com/distribution/;
2.  https://docs.conda.io/en/latest/miniconda.html.

Also, make sure that you can activate anaconda environments ([i.e. add the executable to your PATH](https://support.anaconda.com/customer/en/portal/articles/2621189-conda-%22command-not-found%22-error)).

Then, find a suitable folder on your machine for the code. Git clone and change your directory to the newly cloned code.:
```
git clone https://github.com/larsgeb/hmc-tomography.git
cd hmc-tomography
```
Create the Conda environment. It will be called `hmc-tomography`.
```
conda env create -f environment.yml
```
Initialize it:
```
source activate hmc-tomography
```
You can now start running the codes! They are all designed to run from your current folder (i.e. the root repository folder).
