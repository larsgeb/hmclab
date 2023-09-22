from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from hmclab.Distributions.base import AbstractDistribution


# Define the Hamiltonian dynamics integrator (leapfrog)
def _leapfrog(
    position: np.ndarray,
    momentum: np.ndarray,
    log_prob_grad: Callable[[np.ndarray], np.ndarray],
    step_size: float,
    num_leapfrog_steps: int,
    mass_matrix_inv: np.ndarray,
    store_trajectory: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Perform leapfrog integration for Hamiltonian Monte Carlo (HMC).

    Args:
        position (np.ndarray): Current position in parameter space.
        momentum (np.ndarray): Current momentum.
        log_prob_grad (Callable): Function to compute the gradient of the log
            probability.
        step_size (float): Step size for the leapfrog integrator.
        num_leapfrog_steps (int): Number of leapfrog steps.
        store_trajectory (bool, optional): Whether to store the leapfrog
            trajectory.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing the updated position
        and momentum.
    """
    leapfrog_trajectory = []
    if store_trajectory:
        # Initialize the leapfrog trajectory list and add the initial position
        leapfrog_trajectory.append(position.copy())

    # Perform the first half step of momentum update
    position += 0.5 * step_size * np.dot(mass_matrix_inv, momentum)

    # Perform the leapfrog steps
    for step in range(num_leapfrog_steps - 1):
        if store_trajectory:
            # If storing the trajectory, append the current position
            leapfrog_trajectory.append(position.copy())

        # Update momentum and position for the current step
        momentum -= step_size * log_prob_grad(position)
        position += step_size * np.dot(mass_matrix_inv, momentum)

    # Perform the final half step of momentum update
    if store_trajectory:
        # If storing the trajectory, append the final position
        leapfrog_trajectory.append(position.copy())
    momentum -= step_size * log_prob_grad(position)
    position += 0.5 * step_size * np.dot(mass_matrix_inv, momentum)

    if store_trajectory:
        # If storing the trajectory, append the final position
        leapfrog_trajectory.append(position.copy())

    # Return the updated position, momentum, and the leapfrog trajectory (if
    # stored)
    return position, momentum, leapfrog_trajectory


def hmc_sampler(
    target_distribution: AbstractDistribution,
    initial_position: np.ndarray,
    num_samples: int,
    num_leapfrog_steps: int = 10,
    target_acceptance_rate: float = 0.65,
    step_size: float = 0.1,
    mass_matrix: None | np.ndarray = None,
    visualise_leapfrog: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Hamiltonian Monte Carlo (HMC) sampling.

    Args:
        target_distribution (object): Target distribution with log_prob and
            log_prob_grad methods.
        initial_position (np.ndarray): Initial position in parameter space.
        num_samples (int): Number of samples to generate.
        num_leapfrog_steps (int, optional): Number of leapfrog steps for each
            sample.
        target_acceptance_rate (float, optional): Target acceptance rate for
            step size adaptation.
        step_size (float, optional): Initial step size.
        mass_matrix (None or np.ndarray, optional): Mass matrix for the HMC
            sampler.
        visualise_leapfrog (bool, optional): Whether to visualize the leapfrog
            trajectories.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the
        generated samples, the 50-sample average of acceptance rates, and the
        log probabilities.
    """

    log_prob = target_distribution.log_prob
    log_prob_grad = target_distribution.log_prob_grad

    samples = [initial_position]
    acceptance_rates = []
    log_probs = []

    if mass_matrix is None:
        mass_matrix = np.eye(initial_position.shape[0])
    mass_matrix_inv = np.linalg.inv(mass_matrix)
    mass_matrix_chol = np.linalg.cholesky(mass_matrix)

    leapfrog_trajectories: List = []

    for _ in range(num_samples):
        position = samples[-1].copy()
        momentum = np.random.normal(0, 1, size=position.shape)
        momentum = np.dot(mass_matrix_chol, momentum)
        current_log_prob = log_prob(position)
        current_kinetic_energy = 0.5 * np.dot(
            momentum, np.dot(mass_matrix_inv, momentum)
        )

        randomized_step_size = step_size * (
            1 + 0.5 * (2 * np.random.rand() - 1)
        )

        # Leapfrog integration with num_leapfrog_steps
        position, momentum, leapfrog_trajectory = _leapfrog(
            position,
            momentum,
            log_prob_grad,
            randomized_step_size,
            num_leapfrog_steps,
            mass_matrix_inv=mass_matrix_inv,
            store_trajectory=True,
        )

        proposed_log_prob = log_prob(position)
        proposed_kinetic_energy = 0.5 * np.dot(
            momentum, np.dot(mass_matrix_inv, momentum)
        )

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
            log_probs.append(proposed_log_prob)
            if visualise_leapfrog:
                leapfrog_trajectories.append(leapfrog_trajectory)

        else:
            samples.append(samples[-1])
            log_probs.append(current_log_prob)

    if visualise_leapfrog:
        plot_leapfrog_trajectories(np.array(leapfrog_trajectories))

    return np.array(samples), np.array(acceptance_rates), np.array(log_probs)


def plot_leapfrog_trajectories(trajectories: np.ndarray):
    """
    Plot leapfrog trajectories during Hamiltonian Monte Carlo (HMC) sampling.

    Args:
        trajectories (List[List[np.ndarray]]): List of leapfrog trajectories.
    """
    num_samples = len(trajectories)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    for i in range(num_samples):
        trajectory = np.array(trajectories[i])
        tr1 = trajectory[:, 0]
        tr2 = trajectory[:, 1]

        ax.plot(tr1, tr2, alpha=0.5)

    ax.set_xlabel("Parameter 1")
    ax.set_ylabel("Parameter 2")
    ax.set_title("Leapfrog Trajectories")

    plt.show()
