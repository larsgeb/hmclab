import numpy


def autocorrelation(x):
    result = numpy.correlate(x, x, mode="full")
    return result[int(result.size / 2) :] / numpy.max(result)
