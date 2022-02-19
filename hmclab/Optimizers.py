"""Optimizer classes and associated methods.

The classes in this module describe various numerical optimization routines. These
routines can be used to find the minima of misfit function. This is directly related to
deterministic inversion.

All of the classes inherit from :class:`._AbstractOptimizer`; a base class outlining
required methods and their signatures (required in- and outputs).

"""
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod

import numpy as _numpy
import tqdm.auto as _tqdm_au

from hmclab.Distributions import _AbstractDistribution

from typing import Tuple as _Tuple, List as _List


class _AbstractOptimizer(_ABC):
    """Abstract base class for optimization routines."""

    name: str = "Optimizer abstract base class"
    dimensions: int = -1
    target: _AbstractDistribution

    @_abstractmethod
    def iterate(
        self,
    ) -> _numpy.ndarray:
        pass


class gradient_descent(_AbstractOptimizer):
    def __init__(self):
        pass

    def iterate(
        self,
        target: _AbstractDistribution,
        initial_model: _numpy.ndarray = None,
        epsilon: float = 0.1,
        nmax: int = 100,
    ) -> _Tuple[_numpy.ndarray, float, _List[_numpy.ndarray], _List[float]]:

        dimensions = target.dimensions

        # If no initial model is given, start at zeros
        if initial_model is None:
            m = _numpy.zeros((dimensions, 1))
        else:
            assert initial_model.shape == (dimensions, 1)
            m = initial_model

        # Create progress bar
        try:
            iterations = _tqdm_au.trange(
                nmax,
                desc="Iterating",
                leave=True,
                dynamic_ncols=True,
            )
        except Exception:
            iterations = _tqdm_au.trange(
                nmax,
                desc="Iterating",
                leave=True,
            )

        # Compute initial misfit
        x = target.misfit(m)

        # Create the returns
        xs = []
        ms = []

        # Add starting model and misfit to the returns
        xs.append(x)
        ms.append(m)

        for _ in iterations:

            # Compute gradient
            g = target.gradient(m)
            # Update model
            m = m - epsilon * g

            # Compute misfit and store
            x = target.misfit(m)
            iterations.set_description(f"Misfit: {x:.4e}")
            # Place current model and misfit
            xs.append(x)
            ms.append(m)

        return m, x, _numpy.array(ms), _numpy.array(xs)


class preconditioned_gradient_descent(_AbstractOptimizer):
    def __init__(self):
        pass

    def iterate(
        self,
        target: _AbstractDistribution,
        initial_model: _numpy.ndarray = None,
        epsilon: float = 0.1,
        regularization: float = 1.0,
        nmax: int = 100,
    ) -> _Tuple[_numpy.ndarray, float, _List[_numpy.ndarray], _List[float]]:

        dimensions = target.dimensions

        # If no initial model is given, start at zeros
        if initial_model is None:
            m = _numpy.zeros((dimensions, 1))
        else:
            assert initial_model.shape == (dimensions, 1)
            m = initial_model

        # Create progress bar
        try:
            iterations = _tqdm_au.trange(
                nmax,
                desc="Iterating",
                leave=True,
                dynamic_ncols=True,
            )
        except Exception:
            iterations = _tqdm_au.trange(
                nmax,
                desc="Iterating",
                leave=True,
            )

        # Compute initial misfit
        x = target.misfit(m)

        # Create the returns
        xs = []
        ms = []

        # Add starting model and misfit to the returns
        xs.append(x)
        ms.append(m)

        for _ in iterations:

            # Compute gradient
            g = target.gradient(m)
            preconditioner = _numpy.diag(1.0 / (_numpy.diag(g @ g.T) + regularization))
            # Update model
            m = m - epsilon * (preconditioner @ g)

            # Compute misfit and store
            x = target.misfit(m)
            iterations.set_description(f"Misfit: {x:.4e}")
            # Place current model and misfit
            xs.append(x)
            ms.append(m)

        return m, x, _numpy.array(ms), _numpy.array(xs)
