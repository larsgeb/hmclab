Installation
============

Installing the :code:`hmc_tomography` package is dead-simple. It requires you to have 
Python 3.7 on your system. Using either virtual environments or :code:`Conda` is
recommended.

Quick installation
******************

Run the following inside your local copy of the source code:

.. code-block:: bash    
    
    > $ conda env create -f environment.yml
    > $ conda activate hmctomo
    > $ pip install -e .

After this you can use the code anywhere on your system, or start a Jupyter Notebook
server and have a look at our examples.

Installing the package
**********************

Activate your environment of choice. This can be your system Python, a virtualenv, Conda
environment, etc. 

Environment setup
^^^^^^^^^^^^^^^^^

To create e.g. a new :code:`Conda` environment with the appropriate
Python version and dependencies, run the following from your terminal:

.. code-block:: bash    
    
    > $ conda create -n hmctomo python=3.7
    > $ conda activate hmctomo
    > $ conda install notebook pip

You can also create this environment byu first downloading this repo, and running the
following command inside of it:

.. code-block:: bash    

    > $ conda env create -f environment.yml
    > $ conda activate hmctomo

This does exactly the same as the previous commands.

Installing the package
^^^^^^^^^^^^^^^^^^^^^^

There's at the moment three ways to install the package:
    
1. Install the code directly from GitHub;
2. Clone the GitHub repo and install from that directory.
3. Download the :code:`.zip` file of the repo, unzip, and install from that directory.


Installation option 1
---------------------

Option one simply requires you to run the following command from your shell (with the
appropriate environment activated):

.. code-block:: bash    
    
    > $ pip install -e git+git@github.com:larsgeb/hmc-tomography.git@master#egg=hmc_tomography

This won't work as long as the GitHub repo is private. If you've set up SSH keys with 
your GitHub account, and we've granted you access, you can run the following command 
instead:

.. code-block:: bash    

    > $ pip install -e git+ssh://git@github.com/larsgeb/hmc-tomography.git#egg=hmc_tomography

Installation option 2
---------------------

Option two requires you to run the following commands (with the appropriate environment
activated):

.. code-block:: bash    
    
    > $ git clone git@github.com:larsgeb/hmc-tomography.git
    > $ cd hmc-tomography
    > $ pip install -e .

This also won't work as long as the GitHub repo is private and you don't have access. 

Installation option 3
---------------------

Option three requires you to decompress the :code:`.zip` file and open a terminal in 
the resulting folder (such that you see the files :code:`setup.py`, :code:`README.md`, 
etc. Once you have activated the proper environment in your shell, run the following:

.. code-block:: bash    
    
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
