import hmclab as _hmclab
import numpy as _numpy
import os as _os
import pytest as _pytest
import time as _time
import threading
import _thread
import uuid as _uuid


def interruptor():
    # Simulate a CTRL+C event
    _time.sleep(0.3)
    _thread.interrupt_main()


# Build a slow version just so that we don't generate crazy amounts of samples
class SlowStandardNormal(_hmclab.Distributions.StandardNormal1D):
    def misfit(self, m):
        _time.sleep(0.01)
        return super().misfit(m)

    def gradient(self, m):
        _time.sleep(0.01)
        return super().gradient(m)


@_pytest.mark.parametrize("execution_number", range(10))
def test_break(execution_number):

    # Create some arbitrary posterior
    prior = _hmclab.Distributions.Uniform([-1], [1])
    posterior = _hmclab.Distributions.BayesRule([prior, SlowStandardNormal()])

    unique_name = _uuid.uuid4().hex.upper()
    filename = f"temporary_file_{unique_name}.h5"
    if _os.path.exists(filename):
        _os.remove(filename)  # pragma: no cover

    sampler = _hmclab.Samplers.HMC()

    # Start an interrupt timer
    x = threading.Thread(target=interruptor)
    x.start()

    # Start sampling, which should take longer than the timer (1000x2x0.0005x2)
    sampler.sample(
        filename,
        posterior,
        proposals=10000,
        amount_of_steps=2,
        stepsize=0.03,
        disable_progressbar=True,
    )

    if not _os.path.exists(filename):
        _pytest.fail("Samples file wasn't created")

    # Assert that the last sample was written out correctly
    with _hmclab.Samples(filename=filename) as samples:
        assert not _numpy.all(samples[:, -1] == 0.0)

    _os.remove(filename)
