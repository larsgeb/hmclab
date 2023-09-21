from hmclab.Distributions.base import AbstractDistribution
from hmclab.Distributions import MultivariateNormal
from hmclab.base import _parse_vector_input, _is_vector
import numpy, dill, pytest

inverse_problems_to_test = [MultivariateNormal]


@pytest.mark.parametrize("inverse_problem", inverse_problems_to_test)
def test_dimensions(inverse_problem: AbstractDistribution):
    assert type(inverse_problem.dimensionality) == int
    assert inverse_problem.dimensionality > 0


@pytest.mark.xfail
def test_impossible_input_float():
    parsed_array = _parse_vector_input(1.0, size=3)
    assert isinstance(parsed_array, numpy.ndarray)
    assert parsed_array.size == 3


def test_weird_input():
    weird_array = numpy.ones((1, 1, 15, 1))
    parsed_array = _parse_vector_input(weird_array, size=15)
    assert isinstance(parsed_array, numpy.ndarray)
    assert parsed_array.size == 15


def test_undefined_size():
    weird_array = numpy.ones((10,))
    parsed_array = _parse_vector_input(weird_array)
    assert isinstance(parsed_array, numpy.ndarray)
    assert parsed_array.size == 10


def test_inconsistent_vector_size():
    weird_array = numpy.ones((10, 2))
    parsed_array = _parse_vector_input(weird_array)
    assert isinstance(parsed_array, numpy.ndarray)
    assert parsed_array.size == 20


@pytest.mark.xfail
def test_too_long_array():
    array = numpy.ones((2,))
    parsed_array = _parse_vector_input(array, size=1)
    assert isinstance(parsed_array, numpy.ndarray)
    assert parsed_array.size == 3


@pytest.mark.xfail
def test_silly_input():
    array = "I'm not an array!"
    parsed_array = _parse_vector_input(array, size=1)
    assert isinstance(parsed_array, numpy.ndarray)
    assert parsed_array.size == 3


def test_if_vector():
    array = numpy.ones((10,))
    assert _is_vector(array)


def test_if_specific_length_vector():
    array = numpy.ones((10,))
    assert _is_vector(array, size=10)


def test_weird_list_size():
    weird_list = [[0.0, 3.0], [1.0, 2.0]]
    parsed_array = _parse_vector_input(weird_list)
    assert isinstance(parsed_array, numpy.ndarray)
    assert parsed_array.size == 4


def test_length_one_list():
    weird_short_list = [[[4.32]]]
    parsed_array = _parse_vector_input(weird_short_list, size=1)
    assert isinstance(parsed_array, numpy.ndarray)
    assert parsed_array.size == 1


def test_float():
    weird_value = 4.3
    parsed_array = _parse_vector_input(weird_value, size=1)
    assert isinstance(parsed_array, numpy.ndarray)
    assert parsed_array.size == 1
