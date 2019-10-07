"""A module with assorted samples processing functions.
"""
import numpy as _numpy


def autocorrelation(x):
    result = _numpy.correlate(x - _numpy.mean(x), x - _numpy.mean(x), mode="full")
    return result[int(result.size / 2) :] / _numpy.max(result)


def crosscorrelation(x, y):
    result = _numpy.correlate(x - _numpy.mean(x), y - _numpy.mean(y), mode="full")
    return result[int(result.size / 2) :] / _numpy.max(_numpy.abs(result))
