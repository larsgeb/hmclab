"""Distribution classes and associated methods.

The classes in this module describe statistical distributions. Most of them are
non-normalized having implications for quantifying the evidence term of Bayes' rule.

All of the classes inherit from :class:`._AbstractDistribution`; a base class outlining
required methods and their signatures (required in- and outputs).

.. note::

    A tutorial on implementing your own distributions can be found at
    :ref:`/notebooks/tutorials/3 - Creating your own inverse problem.ipynb`.

"""
# import warnings as _warnings # TODO check why this was here

from hmclab.Distributions.base import (
    AdditiveDistribution,
    BayesRule,
    CompositeDistribution,
    Himmelblau,
    Laplace,
    Normal,
    StandardNormal1D,
    Uniform,
    Mixture,
    EvaluationLimiter_ClassConstructor,
    _AbstractDistribution,
)
from hmclab.Distributions.LinearMatrix import LinearMatrix
from hmclab.Distributions.SourceLocation import SourceLocation2D
from hmclab.Distributions.SourceLocation import SourceLocation3D
from hmclab.Distributions.Transforms import TransformToLogSpace
from hmclab.Distributions.LayeredRayTracing2D import LayeredRayTracing2D


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
    "SourceLocation2D",
    "SourceLocation3D",
    "LayeredRayTracing2D",
    "Mixture",
    "EvaluationLimiter_ClassConstructor",
    "TransformToLogSpace",
]

# Try to import 2D FWI examples if psvWave is installed, otherwise, don't fail
try:
    from hmclab.Distributions.ElasticFullWaveform2D import ElasticFullWaveform2D

    __all__ += ["ElasticFullWaveform2D"]

except ModuleNotFoundError:  # as e:
    pass
