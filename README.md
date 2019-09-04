# HMC Tomography

TravisCI unit test status: 
[![Build Status](https://travis-ci.com/larsgeb/hmc-tomography.svg?token=G43u7wF834znRn3jm2mR&branch=master)](https://travis-ci.com/larsgeb/hmc-tomography)

Manual:
https://larsgeb.github.io/hmc-tomography/

## How to work with this repositories' code

This repository is meant to be an introduction to Bayesian tomography using Hamiltonian Monte Carlo. We designed a general Monte Carlo sampler that is applied to multiple tomographic problems.

To make the code run machine independent, we use a virtual environment. This makes sure that everyone uses the same Python version and packages. We offer two environment options, Conda and VirtualEnv.

## Installing the package from source

Directly to your environment:
```
pip install -e git+git@github.com:larsgeb/hmc-tomography.git@master#egg=hmc_tomography&subdirectory=hmc_tomography
```

From the base directory:

```
python3 setup.py sdist bdist_wheel 
pip install dist/hmc_tomography_lars_gebraad-0.0.1-py3-none-any.whl --force-reinstall
```

## Using Anaconda/Miniconda
Anaconda or Miniconda are both supported. Make sure that you have one of these installed: 

1.  https://www.anaconda.com/distribution/;
2.  https://docs.conda.io/en/latest/miniconda.html.

Also, make sure that you can activate anaconda environments ([i.e. add the executable to your PATH](https://support.anaconda.com/customer/en/portal/articles/2621189-conda-%22command-not-found%22-error)). 

For new Conda users, you might not want to automatically start Anaconda every time you open a command line shell. You can do that by using the following from the shell:
```
$ conda config --set auto_activate_base false
```

Then, find a suitable folder on your machine for the code. Git clone and change your directory to the newly cloned code.:
```
$ git clone https://github.com/larsgeb/hmc-tomography.git
$ cd hmc-tomography
```
Create the Conda environment. It will be called `hmc-tomography`.
```
$ conda env create -f environment.yml
```
Initialize it:
```
$ source activate hmc-tomography
```

To be sure, you can check whether or not the ```python``` or ```python3``` command gives the correct result: 
```bash
$ which python                                                         
[conda installation location]/anaconda3/envs/hmc-tomography/bin/python3
```

You can now start running the codes! They are all designed to run from the bin folder.

## Using VirtualEnv

Make sure that you have VirtualEnv installed through Pip in your interpreter of choice: https://virtualenv.pypa.io/en/latest/installation/ . Also make sure the ```virtualenv``` command is in your path.

To make a new environment for the project:

```bash
$ virtualenv venv-hmc-tomography
```

Activate it:

```bash
$ source venv-hmc-tomography/bin/activate
```

To be sure, you can check whether or not the ```python``` or ```python3``` command gives the correct result: 
```bash
$ which python                                                            
[your repo clone location]/hmc-tomography/venv-hmc-tomography/bin/python
```
Now, to install all the required packages:
