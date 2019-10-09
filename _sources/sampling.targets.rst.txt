###########################
Target (likelihood) objects
###########################

.. module:: hmc_tomography.Targets

Available targets:

.. autosummary:: hmc_tomography.Targets._AbstractTarget
                 hmc_tomography.Targets.LinearMatrix
                 hmc_tomography.Targets.Himmelblau
                 hmc_tomography.Targets.Empty

What are targets?
"""""""""""""""""

Target ABC
""""""""""

Implementing your own Prior
***************************

Target ABC reference
********************

.. autoclass:: hmc_tomography.Targets._AbstractTarget
    :members:

Target objects reference
""""""""""""""""""""""""

Linear forward model target
***************************

.. autoclass:: hmc_tomography.Targets.LinearMatrix
   :members:

    .. automethod:: __init__

Two-dimensional Himmelblau function
***********************************

.. autoclass:: hmc_tomography.Targets.Himmelblau
   :members:

Null function
*************

.. autoclass:: hmc_tomography.Targets.Empty
   :members:
