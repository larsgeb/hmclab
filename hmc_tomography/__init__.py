"""HMC Tomography module.
Copyright 2019-2020 Andrea Zunino, Andreas Fichtner, Lars Gebraad
"""
from hmc_tomography import (
    MassMatrices,
    Distributions,
    Samplers,
    Optimizers,
    Post,
)

from ._version import get_versions
__version__ = get_versions()["version"]
del get_versions

name = "hmc_tomography"
__all__ = [
    "MassMatrices",
    "Distributions",
    "Samplers",
    "Optimizers",
    "Post",
]

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
