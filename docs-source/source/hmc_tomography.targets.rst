##############
Target objects
##############

Targets or likelihood distributions for reference. Your own target class should
inherit from the :ref:`TargetABC-label`.

Implementing your own targets
"""""""""""""""""""""""""""""

.. _TargetABC-label:

Target ABC
""""""""""

.. autoclass:: hmc_tomography.Targets._AbstractTarget
    :inherited-members:

Target classes
""""""""""""""

Simple likelihood distributions available to the HMC sampler, all based on
the :ref:`TargetABC-label`.

.. autoclass:: hmc_tomography.Targets.Himmelblau
   :members:

.. autoclass:: hmc_tomography.Targets.Empty
   :members: