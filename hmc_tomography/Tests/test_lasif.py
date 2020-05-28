# Sampler framework
import pytest as _pytest

import hmc_tomography as _hmc_tomography

import numpy as _numpy

verbosity = 0
percentages = [-5.0, -1.0, -0.1, 0.1, 1.0, 5.0]
parameters = ["VP", "VS", "RHO"]


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

    inv_prob.current_model = m * 0.5
    assert (m == inv_prob.current_model * 2).all()


@_pytest.mark.parametrize("percentage", percentages)
@_pytest.mark.parametrize("parameter", parameters)
@_pytest.mark.xfail(reason="LASIF not found", raises=AttributeError)
def test_mesh_write(percentage, parameter):

    inv_prob = _hmc_tomography.Distributions.LasifFWI(
        "/home/larsgebraad/LASIF_tutorials/lasif_project", verbosity
    )

    m1 = inv_prob.current_model

    # Calculate relevant distances
    coordinates = inv_prob.mesh["MODEL"]["coordinates"][...]
    approximate_midpoint = coordinates.mean(axis=0).mean(axis=0)
    average_variation = coordinates.std(axis=0).mean()
    coordinates - approximate_midpoint
    scaled_distance = ((coordinates - approximate_midpoint) ** 6).sum(axis=2) / (
        (0.5 * average_variation) ** 6
    )

    # Calculate the anomaly function
    percentage_anomaly = (percentage / 100.0) * _numpy.exp(-scaled_distance)
    fraction_of_original = 1.0 + percentage_anomaly

    # Inject the anomaly into one parameter
    index_parameter = inv_prob.mesh_parametrization_fields.index(parameter)
    inv_prob.mesh["MODEL"]["data"][:, index_parameter, :] *= fraction_of_original

    m2 = inv_prob.current_model

    assert not _numpy.all(m1 == m2)

    # Write out the anomaly
    # inv_prob.write_current_mesh(f"anomaly_{percentage}_{parameter}.h5")


@_pytest.mark.parametrize("parameter", parameters)
@_pytest.mark.xfail(reason="LASIF not found", raises=AttributeError)
def test_lasif_gradient(parameter):

    inv_prob = _hmc_tomography.Distributions.LasifFWI(
        "/home/larsgebraad/LASIF_tutorials/lasif_project", verbosity
    )

    inv_prob.__del__()
    # m1 = inv_prob.current_model

    # x1 = inv_prob.misfit(m1)

    # g1 = inv_prob.gradient(m1)
    # g2 = inv_prob.gradient(m1, multiply_mass=True)

    # _numpy.save(f"x1_blob", x1)
    # _numpy.save(f"m1_blob", m1)
    # _numpy.save(f"g1_blob", g1)
    # _numpy.save(f"g2_blob", g2)

    x1 = _numpy.load(f"x1_blob.npy")
    m1 = _numpy.load(f"m1_blob.npy")
    # g1 = _numpy.load(f"g1_blob.npy")
    g2 = _numpy.load(f"g2_blob.npy")

    for percentage in percentages:

        # Recreate, to reset mesh
        inv_prob = _hmc_tomography.Distributions.LasifFWI(
            "/home/larsgebraad/LASIF_tutorials/lasif_project", verbosity
        )

        # Calculate anomaly in appropriate parameter
        coordinates = inv_prob.mesh["MODEL"]["coordinates"][...]
        approximate_midpoint = coordinates.mean(axis=0).mean(axis=0)
        average_variation = coordinates.std(axis=0).mean()
        coordinates - approximate_midpoint
        scaled_distance = ((coordinates - approximate_midpoint) ** 6).sum(axis=2) / (
            (0.5 * average_variation) ** 6
        )
        percentage_anomaly = (percentage / 100.0) * _numpy.exp(-scaled_distance)
        fraction_of_original = 1.0 - percentage_anomaly
        index_parameter = inv_prob.mesh_parametrization_fields.index(parameter)

        # Manually set the mesh (BAD PRACTICE) and invalidate gradients!
        inv_prob.mesh["MODEL"]["data"][:, index_parameter, :] *= fraction_of_original
        inv_prob._last_computed_misfit = None
        inv_prob._last_computed_gradient = None

        m2 = inv_prob.current_model
        x2 = inv_prob.misfit(m2)

        inv_prob.__del__()

        _numpy.save(f"x2_blob_{percentage}_{parameter}", x2)
        _numpy.save(f"m2_blob_{percentage}_{parameter}", m2)

        predicted_change = -(m2 - m1).T @ g2
        actual_change = x2 - x1

        print(f"Gradient test blob: percentage {percentage}, parameter {parameter}")
        print(f"predicted change: {predicted_change}")
        print(f"actual change:    {actual_change}")

    print("\r\nPost-test summary:\r\n\r\n")

    for percentage in percentages:
        x2 = _numpy.load(f"x2_blob_{percentage}_{parameter}")
        m2 = _numpy.load(f"m2_blob_{percentage}_{parameter}")

        predicted_change = -(m2 - m1).T @ g2
        actual_change = x2 - x1

        print(f"Gradient test blob: percentage {percentage}, parameter {parameter}")
        print(f"predicted change: {predicted_change}")
        print(f"actual change:    {actual_change}")


for parameter in parameters:
    test_lasif_gradient(parameter)
