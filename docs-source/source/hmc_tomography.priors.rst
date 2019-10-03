#############
Prior objects
#############

Prior objects encode prior information on parameters. They are tyically
in the form of normalizable distributions (with some exceptions).

.. _PriorABC-label:

Prior ABC
"""""""""

.. autoclass:: hmc_tomography.Priors._AbstractPrior
    :inherited-members:

Well behaved priors
"""""""""""""""""""

Prior distributions available to the HMC sampler.

.. autoclass:: hmc_tomography.Priors.Normal
    :members:

.. autoclass:: hmc_tomography.Priors.LogNormal
    :members:

Ill-behaved priors
""""""""""""""""""

Prior distributions which are not normalizable. A Markov chain over these
distributions without any other regularization will typically 'walk away' to
infinity at some point.

.. autoclass:: hmc_tomography.Priors.UnboundedUniform
    :members:


