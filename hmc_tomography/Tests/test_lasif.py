# Sampler framework
import pytest as _pytest

import hmc_tomography as _hmc_tomography

verbosity = 0


@_pytest.mark.xfail(reason="LASIF not found", raises=AttributeError)
def test_lasif_creation():
    _hmc_tomography.Distributions.LasifFWI(
        "/home/larsgebraad/LASIF_tutorials/lasif_project", verbosity
    )


@_pytest.mark.xfail(reason="LASIF not found", raises=AttributeError)
def test_get_set():

    inv_prob = _hmc_tomography.Distributions.LasifFWI(
        "/home/larsgebraad/LASIF_tutorials/lasif_project", verbosity
    )

    m = inv_prob.current_model
    inv_prob.current_model = m + 1

    assert (m == inv_prob.current_model - 1).all()


@_pytest.mark.xfail(reason="LASIF not found", raises=AttributeError)
def test_lasif_gradient():

    inv_prob = _hmc_tomography.Distributions.LasifFWI(
        "/home/larsgebraad/LASIF_tutorials/lasif_project", verbosity
    )
    m1 = inv_prob.current_model

    m2 = m1 * 0.99995

    x1 = inv_prob.misfit(m1)

    g1 = inv_prob.gradient(m1)
    g2 = inv_prob.gradient(m1, multiply_mass=True)

    x2 = inv_prob.misfit(m2)

    import numpy

    numpy.save("x1", x1)
    numpy.save("x2", x2)
    numpy.save("m1", m1)
    numpy.save("m2", m2)
    numpy.save("g1", g1)
    numpy.save("g2", g2)


# x1 = numpy.load("x1.npy")
# x2 = numpy.load("x2.npy")
# m1 = numpy.load("m1.npy")
# m2 = numpy.load("m2.npy")
# g1 = numpy.load("g1.npy")
# g2 = numpy.load("g2.npy")
