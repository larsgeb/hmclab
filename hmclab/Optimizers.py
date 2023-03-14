"""Optimizer methods.

The methods in this module describe various numerical optimization routines. These
routines can be used to find the minima of misfit function. This is directly related to
deterministic inversion.

"""
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod

import numpy as _numpy
import tqdm.auto as _tqdm_au

from hmclab.Distributions import _AbstractDistribution

from typing import Tuple as _Tuple, List as _List


def gradient_descent(
    target: _AbstractDistribution,
    initial_model: _numpy.ndarray = None,
    epsilon: float = 0.1,
    iterations: int = 100,
    regularization: float = None,
    strictly_monotonic=False,
    disable_progressbar=False,
) -> _Tuple[_numpy.ndarray, float, _List[_numpy.ndarray], _List[float]]:
    """Gradient descent on the target misfit."""

    dimensions = target.dimensions

    # If no initial model is given, start at zeros
    if initial_model is None:
        m = _numpy.zeros((dimensions, 1))
    else:
        assert initial_model.shape == (dimensions, 1)
        m = initial_model

    # Create progress bar
    try:
        progressbar = _tqdm_au.trange(
            iterations,
            desc="Iterating",
            leave=True,
            dynamic_ncols=True,
            disable=disable_progressbar,
        )
    except Exception:
        progressbar = _tqdm_au.trange(
            iterations,
            desc="Iterating",
            leave=True,
            disable=disable_progressbar,
        )

    try:

        # Compute initial misfit
        x = target.misfit(m)

        # Create the returns
        xs = []
        ms = []

        # Add starting model and misfit to the returns
        xs.append(x)
        ms.append(m)

        for _ in progressbar:

            # Compute gradient
            g = target.gradient(m)

            if regularization is not None:
                preconditioner = _numpy.diag(
                    1.0 / (_numpy.diag(g @ g.T) + regularization)
                )
                # Update model
                g = preconditioner @ g

            # Update model
            m = m - epsilon * g

            # Compute misfit and store
            x = target.misfit(m)

            if _numpy.isnan(x) or _numpy.isinf(x):
                # Reset model and misfit
                x = xs[-1]
                m = ms[-1]
                # And exit loop
                progressbar.close()
                print("Encountered infinite or NaN values, terminating")
                break

            if x > xs[-1] and strictly_monotonic:
                # Reset model and misfit
                x = xs[-1]
                m = ms[-1]
                # And exit loop
                progressbar.close()
                print("Value is not strictly decreasing, terminating")
                break

            progressbar.set_description(f"Misfit: {x:.1e}")
            # Place current model and misfit
            xs.append(x)
            ms.append(m)
    except KeyboardInterrupt:
        pass

    return m, x, _numpy.array(ms), _numpy.array(xs)
