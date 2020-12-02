import hmc_tomography as _hmc_tomography
import numpy as _numpy
import os as _os
import pytest as _pytest
import time as _time
import threading
import time
import _thread


def interruptor():
    # Simulate a CTRL+C event
    _time.sleep(0.1)
    _thread.interrupt_main()


# Build a slow version just so that we don't generate crazy amounts of samples
class SlowStandardNormal(_hmc_tomography.Distributions.StandardNormal1D):
    def misfit(self, m):
        _time.sleep(0.0005)
        return super().misfit(m)

    def gradient(self, m):
        _time.sleep(0.0005)
        return super().gradient(m)


def test_break():

    # Create some arbitrary posterior
    prior = _hmc_tomography.Distributions.Uniform([-1], [1])
    posterior = _hmc_tomography.Distributions.BayesRule([prior, SlowStandardNormal()])

    filename = "temporary_file.h5"
    if _os.path.exists(filename):
        _os.remove(filename)

    sampler = _hmc_tomography.Samplers.HMC()

    # Start an interrupt timer
    x = threading.Thread(target=interruptor)
    x.start()

    # Start sampling, which should take longer than the timer (1000x2x0.0005x2)
    sampler.sample(
        filename,
        posterior,
        proposals=1000,
        ram_buffer_size=1,
        amount_of_steps=2,
        stepsize=0.03,
    )

    if not _os.path.exists(filename):
        _pytest.fail("Samples file wasn't created")

    # Assert that the last sample was written out correctly
    with _hmc_tomography.Samples(filename=filename) as samples:
        assert not _numpy.all(samples[:, -1] == 0.0)

    _os.remove(filename)
