"""Distribution classes and associated methods.

The classes in this module describe statistical distributions. Most of them are 
non-normalized having implications for quantifying the evidence term of Bayes' rule.

All of the classes inherit from :class:`._AbstractDistribution`; a base class outlining
required methods and their signatures (required in- and outputs). 

.. note::

    A tutorial on implementing your own distributions can be found at 
    :ref:`/examples/4 - Creating your own inverse problem.ipynb`.

"""
from hmc_tomography.Distributions.base import (
    _AbstractDistribution,
    StandardNormal1D,
    Normal,
    Laplace,
    Uniform,
    CompositeDistribution,
    AdditiveDistribution,
    Himmelblau,
    BayesRule,
)

from hmc_tomography.Distributions.LinearMatrix import LinearMatrix

__all__ = [
    "_AbstractDistribution",
    "StandardNormal1D",
    "Normal",
    "Laplace",
    "Uniform",
    "CompositeDistribution",
    "AdditiveDistribution",
    "Himmelblau",
    "BayesRule",
    "LinearMatrix",
]
