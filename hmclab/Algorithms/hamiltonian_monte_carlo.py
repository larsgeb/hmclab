from typing import Callable, Tuple

import numpy as np

from hmclab.Distributions.base import AbstractDistribution


# Define the Hamiltonian dynamics integrator (leapfrog) for vectors with
# optimized half steps
def leapfrog(
    position: np.ndarray,
    momentum: np.ndarray,
    log_prob_grad: Callable[[np.ndarray], np.ndarray],
    step_size: float,
    num_leapfrog_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform leapfrog integration for Hamiltonian Monte Carlo (HMC).

    Args:
        position (np.ndarray): Current position in parameter space.
        momentum (np.ndarray): Current momentum.
        log_prob_grad (Callable): Function to compute the gradient of the log
        probability.
        step_size (float): Step size for the leapfrog integrator.
        num_leapfrog_steps (int): Number of leapfrog steps.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing the updated position
        and momentum.
    """
    half_step_momentum = 0.5 * step_size * log_prob_grad(position)
    momentum -= half_step_momentum

    for _ in range(num_leapfrog_steps - 1):
        position += step_size * momentum
        momentum -= step_size * log_prob_grad(position)

    position += step_size * momentum
    momentum -= half_step_momentum

    return position, momentum


def hmc_sampler(
    target_distribution: AbstractDistribution,
    initial_position: np.ndarray,
    num_samples: int,
    num_leapfrog_steps: int = 10,
    target_acceptance_rate: float = 0.65,
    step_size: float = 0.1,
) -> np.ndarray:
    """
    Perform Hamiltonian Monte Carlo (HMC) sampling.

    Args:
        target_distribution (object): Target distribution with log_prob and
        log_prob_grad methods.
        initial_position (np.ndarray): Initial position in parameter space.
        num_samples (int): Number of samples to generate.
        num_leapfrog_steps (int): Number of leapfrog steps for each sample.
        target_acceptance_rate (float): Target acceptance rate for step size
        adaptation.
        step_size (float): Initial step size.

    Returns:
        np.ndarray: Array containing the generated samples.
    """
    log_prob = target_distribution.log_prob
    log_prob_grad = target_distribution.log_prob_grad

    samples = [initial_position]
    acceptance_rates = []

    for _ in range(num_samples):
        position = samples[-1].copy()
        momentum = np.random.normal(0, 1, size=position.shape)
        current_log_prob = log_prob(position)
        current_kinetic_energy = 0.5 * np.sum(momentum**2)

        # Leapfrog integration with num_leapfrog_steps
        position, momentum = leapfrog(
            position, momentum, log_prob_grad, step_size, num_leapfrog_steps
        )

        proposed_log_prob = log_prob(position)
        proposed_kinetic_energy = 0.5 * np.sum(momentum**2)

        # Metropolis acceptance step
        accept_prob = min(
            1,
            np.exp(
                current_log_prob
                - proposed_log_prob
                + current_kinetic_energy
                - proposed_kinetic_energy
            ),
        )

        # Adaptive step size adjustment
        acceptance_rates.append(accept_prob)

        if len(acceptance_rates) >= 50:
            recent_acceptance_rate = np.mean(acceptance_rates[-50:])
            if recent_acceptance_rate > target_acceptance_rate:
                step_size *= 1.1  # Increase step size
            else:
                step_size *= 0.9  # Decrease step size

        if np.random.rand() < accept_prob:
            samples.append(position)
        else:
            samples.append(samples[-1])

    return np.array(samples)
