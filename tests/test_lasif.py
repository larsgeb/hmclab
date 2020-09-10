# Sampler framework
import pytest as _pytest

import hmc_tomography as _hmc_tomography

import numpy as _numpy

import os as _os

verbosity = 0
percentages = [-5.0, -1.0, -0.1, 0.1, 1.0, 5.0]
parameters = ["VP", "VS", "RHO"]
project_folder = (
    "/home/larsgebraad/Documents/Global probabilistic "
    "Full-waveform inversion/Working LASIF Project/lasif_project"
)


def test_lasif_creation():
    inv_prob = _hmc_tomography.Distributions.LasifFWI(project_folder, verbosity)

    print(f"Inverse problem dimensions: {inv_prob.dimensions}")


def test_get_set():

    inv_prob = _hmc_tomography.Distributions.LasifFWI(project_folder, verbosity)

    m = inv_prob.current_model

    inv_prob.current_model = m * 0.5
    assert (m == inv_prob.current_model * 2).all()


@_pytest.mark.parametrize("percentage", percentages)
@_pytest.mark.parametrize("parameter", parameters)
def test_mesh_write(percentage, parameter):

    inv_prob = _hmc_tomography.Distributions.LasifFWI(project_folder, verbosity)

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
    inv_prob.write_current_mesh(f"anomaly_{percentage}_{parameter}.h5")


def test_random_field():

    inv_prob = _hmc_tomography.Distributions.LasifFWI(project_folder, verbosity)

    m = inv_prob.current_model

    inv_prob.current_model = 100 * _numpy.random.randn(inv_prob.dimensions, 1)

    # Write out the anomaly
    inv_prob.write_current_mesh(f"anomaly_random_no_mass.h5")

    mass_matrix = _numpy.tile(_numpy.load("mass_matrix.npy").flatten()[:, None], [3, 1])

    inv_prob.current_model = (
        100 * _numpy.random.randn(inv_prob.dimensions, 1) / mass_matrix
    )

    inv_prob.write_current_mesh(f"anomaly_random_with_mass.h5")


@_pytest.mark.run(order=1)
def test_compute_gradient():

    if (
        _os.path.isfile("x1_blob.npy")
        and _os.path.isfile("m1_blob.npy")
        and _os.path.isfile("g1_blob.npy")
        and _os.path.isfile("g2_blob.npy")
    ):
        _pytest.skip()

    inv_prob = _hmc_tomography.Distributions.LasifFWI(project_folder, verbosity)

    m1 = inv_prob.current_model
    x1 = inv_prob.misfit(m1)

    g1 = inv_prob.gradient(m1, multiply_mass=False)
    g2 = inv_prob.gradient(m1, multiply_mass=True)

    _numpy.save("x1_blob.npy", x1)
    _numpy.save("m1_blob.npy", m1)
    _numpy.save("g1_blob.npy", g1)
    _numpy.save("g2_blob.npy", g2)


@_pytest.mark.run(order=2)
@_pytest.mark.parametrize("parameter", parameters)
@_pytest.mark.parametrize("percentage", percentages)
def test_gradient_step(parameter, percentage):

    if _os.path.isfile(f"x2_blob_{percentage}_{parameter}.npy") and _os.path.isfile(
        f"m2_blob_{percentage}_{parameter}.npy"
    ):
        _pytest.skip()

    # Recreate, to reset mesh
    inv_prob = _hmc_tomography.Distributions.LasifFWI(project_folder, verbosity)

    # # Calculate anomaly in appropriate parameter
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

    _numpy.save(f"x2_blob_{percentage}_{parameter}", x2)
    _numpy.save(f"m2_blob_{percentage}_{parameter}", m2)


@_pytest.mark.run(order=3)
@_pytest.mark.parametrize("parameter", parameters)
@_pytest.mark.parametrize("percentage", percentages)
def test_gradient_summary(parameter, percentage):

    x1 = _numpy.load(f"x1_blob.npy")
    m1 = _numpy.load(f"m1_blob.npy")
    g1 = _numpy.load(f"g1_blob.npy")
    g2 = _numpy.load(f"g2_blob.npy")
    x2 = _numpy.load(f"x2_blob_{percentage}_{parameter}.npy")
    m2 = _numpy.load(f"m2_blob_{percentage}_{parameter}.npy")

    predicted_change_A = (m2 - m1).T @ g1
    predicted_change_B = (m2 - m1).T @ g2
    actual_change = x2 - x1

    print(f"\r\n\r\nGradient test blob: percentage {percentage}, parameter {parameter}")
    print(f"predicted change: {predicted_change_A}")
    print(f"predicted change: {predicted_change_B}")
    print(f"actual change:    {actual_change}")
    print(f"Misfits:          {x1} {x2}")
