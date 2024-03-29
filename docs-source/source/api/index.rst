API reference
=============

.. module:: hmclab

.. warning:: 

    The API reference includes all finished methods with full signature. However, a lot
    of the descriptions are still missing. Additionally, we expect that some arguments 
    might still change names, potentially breaking code. Be aware of this in future
    updates.

These packages describe the modules, classes and methods that constitute the 
:code:`hmclab` package. Here you'll find detailed explanations of how all the 
components work together.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Overview of submodules

   distributions/index
   massmatrices/index
   samplers/index
   optimizers/index
   visualization/index

.. autosummary::

    Distributions
    MassMatrices
    Samplers
    Optimizers
    Visualization

.. autosummary:: 
   :toctree: _autosummary
   :template: custom-class-template.rst
   :nosignatures:
   :recursive:

   Samples