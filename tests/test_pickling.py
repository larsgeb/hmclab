import pytest as _pytest
import dill as _dill
from hmclab import Distributions as _Distributions
from hmclab.Helpers.CustomExceptions import InvalidCaseError as _InvalidCaseError
import numpy as _numpy
import pytest as _pytest
import hmclab as _hmclab

_ad = _hmclab.Distributions._AbstractDistribution
_as = _hmclab.Samplers._AbstractSampler


dimensions = [1, 2, 5, 50]
distribution_classes = _ad.__subclasses__()
sampler_classes = _as.__subclasses__()
sampler_classes.remove(_hmclab.Samplers._AbstractVisualSampler)


@_pytest.mark.parametrize("pclass", distribution_classes)
@_pytest.mark.parametrize("dimensions", dimensions)
def test_creation(pclass: _Distributions._AbstractDistribution, dimensions: int):
    # Create the object
    try:
        distribution: _Distributions._AbstractDistribution = pclass.create_default(
            dimensions
        )
    except _InvalidCaseError:
        return _pytest.skip("Invalid case")

    # Check if a subtype of mass matrices
    assert issubclass(type(distribution), _Distributions._AbstractDistribution)

    # Check if the right amount of dimensions
    assert distribution.dimensions == dimensions

    try:
        assert _dill.pickles(distribution)
    except AssertionError as e:
        print(_dill.detect.badtypes(distribution, depth=1))
        raise e


@_pytest.mark.parametrize("sampler_class", sampler_classes)
def test_basic_sampling(
    sampler_class: _as,
):
    sampler_instance = sampler_class()

    assert isinstance(sampler_instance, _as)

    try:
        assert _dill.pickles(sampler_instance)
    except AssertionError as e:
        print(_dill.detect.badtypes(sampler_instance, depth=1))
        raise e
