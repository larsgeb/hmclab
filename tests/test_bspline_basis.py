# Sampler framework
from numpy.lib import polynomial
import hmc_tomography as _hmc_tomography
import numpy as _numpy
import matplotlib.pyplot as _plt
import pandas as _pandas
from scipy import interpolate as _interpolate

verbosity = 0
percentages = [-5.0, -1.0, -0.1, 0.1, 1.0, 5.0]
parameters = ["VP", "VS", "RHO"]
project_folder = (
    "/home/larsgebraad/Documents/Global probabilistic "
    "Full-waveform inversion/Working LASIF Project/lasif_project"
)


def test_lasif_creation_spline():

    prem = _pandas.read_csv("tests/configurations/PREM_1s_IDV.csv", skiprows=[0])

    radius = prem['radius[unit="km"]']
    vp = prem['Vpv[unit="km/s"]'] * 1e3
    vs = prem['Vsv[unit="km/s"]'] * 1e3
    density = prem['density[unit="g/cm^3"]'] * 1e3

    interpolator_vp = _interpolate.interp1d(radius, vp, kind="linear")
    interpolator_vs = _interpolate.interp1d(radius, vs, kind="linear")
    interpolator_density = _interpolate.interp1d(radius, density, kind="linear")
    interpolators = {
        "VP": interpolator_vp,
        "VS": interpolator_vs,
        "RHO": interpolator_density,
    }

    inv_prob = _hmc_tomography.Distributions.LasifFWI(
        project_folder,
        verbosity,
        spline={
            "dof_per_parameter": 6,
            "polynomial_order": 3,
            "knot_locations": None,
            "interfaces": [5971, 6151, 6346.6, 6356, 6368],
            "collocation_interfaces": [5971, 6151, 6291, 6346.6, 6356, 6368],
            "background_interpolators": interpolators,
        },
    )

    # sample = 250 * _numpy.random.randn(inv_prob.dimensions, 10)
    # inv_prob.plot_bspline_marginals(sample, internal_points=5000)
    # inv_prob.plot_bspline_models(sample, internal_points=5000)

    m = inv_prob.current_model
    inv_prob.current_model = m
    iterations = 10
    ms = [m.copy()] * (iterations + 1)
    xs = [None] * (iterations + 1)
    gs = [None] * iterations

    for i in range(iterations):
        g = inv_prob.gradient(m)
        xs[i] = inv_prob._last_computed_misfit
        m = m - g * 2.5e5
        gs[i] = g
        ms[i + 1] = m.copy()
    xs[-1] = inv_prob.misfit(m)

    # inv_prob.current_model = 100 * _numpy.ones_like(inv_prob.current_model)
    # m = inv_prob.current_model

    assert False

    for i in range(iterations + 1):
        inv_prob.plot_bspline_models(ms[i], internal_points=5000, axis=_plt.gca())
    # inv_prob.spline_basis["VP"].plot_basis_functions()
    # _plt.tight_layout()
    # _plt.show()

    # inv_prob.spline_basis["VS"].plot_basis_functions()
    # _plt.tight_layout()
    # _plt.show()

    # inv_prob.spline_basis["RHO"].plot_basis_functions()
    # _plt.tight_layout()
    # _plt.show()
