# Sampler framework
import pytest as _pytest

import hmc_tomography as _hmc_tomography


@_pytest.mark.xfail(reason="LASIF not found", raises=AttributeError)
def test_lasif_creation():

    inv_prob = _hmc_tomography.Distributions.LasifFWI(
        "/home/larsgebraad/LASIF_tutorials/lasif_project"
    )

    print(inv_prob.current_model)
