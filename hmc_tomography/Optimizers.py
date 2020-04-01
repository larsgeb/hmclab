"""Optimizer classes and associated methods.

The classes in this module describe various numerical optimization routines. These 
routines can be used to find the minima of misfit function. This is directly related to
deterministic inversion.

All of the classes inherit from :class:`._AbstractOptimizer`; a base class outlining
required methods and their signatures (required in- and outputs). 

"""
import sys as _sys
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod

import h5py as _h5py
import numpy as _numpy
import time as _time
import tqdm as _tqdm
import warnings as _warnings
from typing import Tuple as _Tuple
import tqdm.auto as _tqdm_au

from hmc_tomography.Distributions import _AbstractDistribution


class _AbstractOptimizer(_ABC):
    """Abstract base class for optimization routines."""

    name: str = "Optimizer abstract base class"
    dimensions: int = -1
    target: _AbstractDistribution

    @_abstractmethod
    def iterate_once(self,) -> _numpy.ndarray:
        """
        Parameters
        ----------
        proposals
        online_thinning
        sample_ram_buffer_size
        samples_filename

        """
        pass

    @_abstractmethod
    def iterate(self,) -> _numpy.ndarray:
        """
        Parameters
        ----------
        proposals
        online_thinning
        sample_ram_buffer_size
        samples_filename

        """
        pass


class gradient_descent(_AbstractOptimizer):
    """An unscaled gradient descent optimization routine."""

    def __init__(
        self, target: _AbstractDistribution, epsilon: float = 0.1,
    ):
        # Setting the passed objects -------------------------------------------
        self.dimensions = target.dimensions
        self.target: _AbstractDistribution = target
        self.epsilon = epsilon

    def iterate_once(self, initial_model, epsilon: float = 0.1):
        # Initial model
        if initial_model is None:
            initial_model = _numpy.zeros((self.dimensions, 1))
        else:
            assert initial_model.shape == (self.dimensions, 1)

        g = self.target.gradient(initial_model)

        new_model = initial_model - g * epsilon

        return new_model

    def iterate(
        self,
        initial_model,
        epsilon: float = 0.1,
        nmax: int = 100,
        online_thinning: int = 1,
    ):
        if initial_model is None:
            m = _numpy.zeros((self.dimensions, 1))
        else:
            assert initial_model.shape == (self.dimensions, 1)
            m = initial_model

        xs = []
        ms = []

        # Create progress bar
        try:
            iterations = _tqdm_au.trange(
                nmax, desc="Iterating", leave=True, dynamic_ncols=True,
            )
        except:
            iterations = _tqdm_au.trange(nmax, desc="Iterating", leave=True,)

        for iteration in iterations:
            xs.append(self.target.misfit(m))
            ms.append(m)
            m = m - epsilon * (self.target.gradient(m))

        return m, _numpy.array(ms), _numpy.array(xs)


class simple_preconditioned_gradient_descent(_AbstractOptimizer):
    def __init__(
        self,
        target: _AbstractDistribution,
        epsilon: float = 0.1,
        regularization: float = 1.0,
    ):
        # Setting the passed objects -------------------------------------------
        self.dimensions = target.dimensions
        self.target: _AbstractDistribution = target
        self.epsilon = epsilon
        self.regularization = regularization

    def iterate_once(self, initial_model, epsilon: float = 0.1):
        # Initial model
        if initial_model is None:
            initial_model = _numpy.zeros((self.dimensions, 1))
        else:
            assert initial_model.shape == (self.dimensions, 1)

        g = self.target.gradient(initial_model)

        precond = _numpy.diag(1.0 / (_numpy.diag(g @ g.T) + self.regularization))

        new_model = initial_model - epsilon * (precond @ g)

        return new_model

    def iterate(
        self,
        initial_model,
        epsilon: float = 0.1,
        nmax: int = 100,
        online_thinning: int = 1,
    ):
        if initial_model is None:
            m = _numpy.zeros((self.dimensions, 1))
        else:
            assert initial_model.shape == (self.dimensions, 1)
            m = initial_model

        xs = []
        ms = []

        # Create progress bar
        try:
            iterations = _tqdm_au.trange(
                nmax, desc="Iterating", leave=True, dynamic_ncols=True,
            )
        except:
            iterations = _tqdm_au.trange(nmax, desc="Iterating", leave=True,)

        for iteration in iterations:
            xs.append(self.target.misfit(m))
            ms.append(m)

            g = self.target.gradient(m)

            precond = _numpy.diag(1.0 / (_numpy.diag(g @ g.T) + self.regularization))

            m = m - epsilon * (precond @ g)

        return m, _numpy.array(ms), _numpy.array(xs)
