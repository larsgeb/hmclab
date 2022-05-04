#################
HMC Lab // Python
#################

These pages provide the documentation of the Python side of HMC Lab. Here you'll find
notebooks detailing how different concepts in Bayesian inference using Markov chain
Monte Carlo work. In particular, extra attention is given on how to use the Hamiltonian
Monte Carlo algorithm on (your) geophysical inverse problems.

HMC Lab is tested on Python 3.8+. In theory, it should work well on any system that has
access to Conda. 

Quickstart
----------

To download the repo, create a Conda environment, and install all dependencies, run the
following: 

.. code-block:: bash    

    > $ git clone https://github.com/larsgeb/hmclab.git
    > $ cd hmclab
    > $ conda env create -f environment.yml
    > $ conda activate hmclab
    > $ pip install -e .

The resulting Conda environment should be able to run all notebooks found in
hmclab/notebooks. See the installation page for more detailed instructions.


.. toctree::
    :maxdepth: 1
    :caption: Contents:
    :hidden:

    self
    hmc
    setup
    notebooks
    api/index
    py-modindex
    genindex


.. centered:: Andrea Zunino, Andreas Fichtner, Lars Gebraad


