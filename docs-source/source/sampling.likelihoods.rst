###############################
Likelihood (likelihood) objects
###############################

.. module:: hmc_tomography.Likelihoods

Available targets:

.. autosummary:: hmc_tomography.Likelihoods._AbstractLikelihood
                 hmc_tomography.Likelihoods.LinearMatrix
                 hmc_tomography.Likelihoods.Himmelblau
                 hmc_tomography.Likelihoods.Empty

What are targets?
"""""""""""""""""

Likelihood ABC
""""""""""""""

Implementing your own Likelihood
********************************

Likelihood ABC reference
************************

.. autoclass:: hmc_tomography.Likelihoods._AbstractLikelihood
    :members:

Likelihood objects reference
""""""""""""""""""""""""""""

Linear forward model target
***************************

.. autoclass:: hmc_tomography.Likelihoods.LinearMatrix
   :members:

    .. automethod:: __init__

Two-dimensional Himmelblau function
***********************************

.. autoclass:: hmc_tomography.Likelihoods.Himmelblau
   :members:

Null function
*************

.. autoclass:: hmc_tomography.Likelihoods.Empty
   :members:
