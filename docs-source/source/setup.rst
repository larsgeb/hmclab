Installation
============

Installing the :code:`hmc_tomography` package is dead-simple. It requires you to have 
Python 3.7 on your system. Virtual environments and :code:`Conda` are also fine. 

Installing the package in a new Conda environment
*************************************************

Activate your environment of choice. To create e.g. a new :code:`Conda` environment with the
appropriate Python version, run the following from your terminal:

.. code-block:: bash    
    
    > $ conda create -n hmc-tomography python=3.7

Now to install the package, we need to first activate this distribution:

.. code-block:: bash    
    
    > $ conda activate hmc-tomography

There's at the moment two options to install the package:
    
1. Install the code directly from GitHub;
2. Clone the GitHub repo and install from that directory.

Option one simply requires you to run the following command from your shell (with the
appropriate environment activated):

.. code-block:: bash    
    
    > $ pip install -e git+git@github.com:larsgeb/hmc-tomography.git@master#egg=hmc_tomography

Option two requires you to run the following commands (with the appropriate environment
activated):

.. code-block:: bash    
    
    > $ git clone git@github.com:larsgeb/hmc-tomography.git
    > $ cd hmc-tomography
    > $ pip install -e .

If the command succeeds, you now have access to the package from your Python 3.7 
distribution by importing :code:`hmc_tomography`:

.. code-block:: python

    import hmc_tomography

Installing development dependencies
***********************************

If you want to develop within this repo, we recommend a few extra packages. They can 
also be installed using pip.

In :code:`Bash`:

.. code-block:: bash    
    
        # from github
    > $ pip install -e \ 
        git+git@github.com:larsgeb/hmc-tomography.git@master#egg=hmc_tomography[dev]
    
        # from local clone
    > $ pip install -e .[dev] 

... or :code:`Zsh`, which requires escapes for brackets:

.. code-block:: bash    
    
        # from github
    > $ pip install -e\ 
        git+git@github.com:larsgeb/hmc-tomography.git@master#egg=hmc_tomography\[dev\] 
    
        # from local clone
    > $ pip install -e .\[dev\] 
