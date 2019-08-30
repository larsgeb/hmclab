import numpy


def autocorrelation(x):

    result = numpy.correlate(x - numpy.mean(x), x - numpy.mean(x), mode="full")
    return result[int(result.size / 2) :] / numpy.max(result)
