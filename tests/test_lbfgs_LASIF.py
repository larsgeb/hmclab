import hmc_tomography as _hmc_tomography

import numpy as _numpy

verbosity = 0
percentages = [-5.0, -1.0, -0.1, 0.1, 1.0, 5.0]
parameters = ["VP", "VS", "RHO"]
project_folder = (
    "/home/larsgebraad/Documents/Global probabilistic "
    "Full-waveform inversion/Working LASIF Project/lasif_project"
)


def test_lbfgs():

    posterior_1 = _hmc_tomography.Distributions.LasifFWI(project_folder, verbosity)

    from scipy.optimize import minimize

    m1 = posterior_1.current_model

    def misfit_1(m):
        return posterior_1.misfit(m[:, None]) * 1e20

    def gradient_1(m):
        return posterior_1.gradient(m[:, None])[:, 0] * 1e20

    options = {"ftol": 1e-10, "disp": True, "maxiter": 10, "maxcor": 10}

    res_1 = minimize(misfit_1, m1, method="L-BFGS-B", jac=gradient_1, options=options,)

    _numpy.save("lbfgs_result.npy", res_1.x[:, None])
