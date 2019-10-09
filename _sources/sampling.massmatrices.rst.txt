###################
Mass matrix objects
###################

.. module:: hmc_tomography.MassMatrices

Available mass matrices:

.. autosummary:: hmc_tomography.MassMatrices._AbstractMassMatrix
                 hmc_tomography.MassMatrices.Unit
                 hmc_tomography.MassMatrices.Diagonal
                 hmc_tomography.MassMatrices.LBFGS


What are mass matrices?
"""""""""""""""""""""""

Mass matrix ABC
"""""""""""""""

This class is the abstract base class. Our package does not **require** you to
inherit from this class, but it makes checking for required methods easier.

Implementing your own Mass Matrix
*********************************


Mass Matrix ABC reference
*************************

.. autoclass:: hmc_tomography.MassMatrices._AbstractMassMatrix
    :members:

Mass matrices reference
"""""""""""""""""""""""

Unit mass matrix
****************

.. autoclass:: hmc_tomography.MassMatrices.Unit
    :members:

Diagonal mass matrix
********************

.. autoclass:: hmc_tomography.MassMatrices.Diagonal
    :members:

LBFGS-style adaptive mass matrix
********************************

.. autoclass:: hmc_tomography.MassMatrices.LBFGS
    :members:

