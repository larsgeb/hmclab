import numpy as np


# Define the Hamiltonian dynamics integrator (leapfrog) for vectors with
# optimized half steps
def leapfrog(position, momentum, log_prob_grad, step_size, num_leapfrog_steps):
    half_step_momentum = 0.5 * step_size * log_prob_grad(position)
    momentum -= half_step_momentum
    for _ in range(
        num_leapfrog_steps - 1
    ):  # Number of leapfrog steps (adjust as needed)
        position += step_size * momentum
        momentum -= step_size * log_prob_grad(position)
    position += step_size * momentum
    momentum -= half_step_momentum
    return position, momentum


def hmc_sampler(
    target_distribution,
    initial_position,
    num_samples,
    num_leapfrog_steps=int(10),
    target_acceptance_rate=0.65,
    step_size=0.1,
):
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
        accept_prob = float(
            min(
                1,
                np.exp(
                    current_log_prob
                    - proposed_log_prob
                    + current_kinetic_energy
                    - proposed_kinetic_energy
                ),
            )
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
