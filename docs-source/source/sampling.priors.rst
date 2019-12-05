#############
Prior objects
#############

.. module:: hmc_tomography.Priors

Prior objects encode prior information on parameters. They are tyically
in the form of normalizable distributions (with some exceptions).

Avialable priors:

.. autosummary:: hmc_tomography.Priors._AbstractPrior
                 hmc_tomography.Priors.Normal
                 hmc_tomography.Priors.Sparse
                 hmc_tomography.Priors.LogNormal
                 hmc_tomography.Priors.Uniform
                 hmc_tomography.Priors.CompositePrior
                 hmc_tomography.Priors.AdditivePrior
                 hmc_tomography.Priors.MultiplicativePrior


What are priors?
""""""""""""""""

Prior ABC
"""""""""

This class is the abstract base class. Our package does not **require** you to
inherit from this class, but it makes checking for required methods easier.

Implementing your own Prior
***************************

Creating a new class based upon this ABC in a separate file is done as
follows::

    from hmc_tomography.Priors import _AbstractPrior
    class newPrior(_AbstractPrior):
        pass

However, when one would try to instantiate this object::

    prior = newPrior()

a `TypeError` is raised; because the abstract (read: required) methods are
not implemented yet:

``TypeError: Can't instantiate abstract class newPrior with abstract methods
[..]``

Prior ABC reference
*******************

.. autoclass:: hmc_tomography.Priors._AbstractPrior
    :members:

Prior distributions reference
"""""""""""""""""""""""""""""

Normal distribution
*******************

.. autoclass:: hmc_tomography.Priors.Normal
    :members:

Laplace distribution
********************

.. autoclass:: hmc_tomography.Priors.Sparse
    :members:

Logarithmic normal distribution
*******************************

.. autoclass:: hmc_tomography.Priors.LogNormal
    :members:

Uniform distribution
********************

.. autoclass:: hmc_tomography.Priors.Uniform
    :members:

Composite distribution
**********************

.. autoclass:: hmc_tomography.Priors.CompositePrior
    :members:

Additive distribution
*********************

.. autoclass:: hmc_tomography.Priors.AdditivePrior
    :members:

Multiplicative distribution
***************************

.. autoclass:: hmc_tomography.Priors.MultiplicativePrior
    :members:

