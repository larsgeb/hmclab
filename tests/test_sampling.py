from hmclab.Algorithms import hmc_sampler
from hmclab.Distributions import MultivariateNormal
from hmclab import Inversion
import numpy as np, matplotlib.pyplot as plt


def test_basic_inversion():
    mean_vector = [0, 0]
    covariance_matrix = [[1, 0.5], [0.5, 2]]
    multivariate_norm = MultivariateNormal(mean_vector, covariance_matrix)

    # Create an inversion instance with the MultivariateNormal distribution
    inversion = Inversion(target_distribution=multivariate_norm)
    # # Create an inversion instance
    # inversion = Inversion(target_distribution=target_distribution)

    # Define an initial position for the HMC sampler (1D example)
    initial_position = np.array(
        [0.0, 0.0]
    )  # Adjust the initial value as needed

    # Run the inversion
    inversion.run_sampler(
        hmc_sampler,
        initial_position=initial_position,  # Make sure you provide initial_position
        num_samples=100,
        num_leapfrog_steps=10,
        target_acceptance_rate=0.65,
        step_size=0.1,
    )

    # Save the results in the desired format (e.g., 'numpy', 'pandas', or 'netcdf')
    inversion.save_results("inversion_results.npz", format="numpy")

    loaded_inversion = Inversion(
        target_distribution=None
    )  # Initialize with a dummy target_distribution

    loaded_inversion.load_results("inversion_results.npz", format="numpy")

    # Assuming you have loaded the results and have a 'loaded_inversion' instance

    # Plot the marginal distributions of both dimensions
    loaded_inversion.plot_subset_marginal_distributions(
        bins=30, color="skyblue", grid=True
    )

    # Create a pairwise scatterplot for both dimensions
    loaded_inversion.plot_subset_pairwise_scatterplot(
        color="skyblue", grid=True
    )

    # Plot trace plots for the loaded samples
    loaded_inversion.plot_trace()

    # Plot autocorrelation plot for the loaded samples
    loaded_inversion.plot_autocorrelation()

    plt.close("all")

    # Compute and print MCMC statistics for the loaded samples
    loaded_inversion.compute_mcmc_statistics()

    loaded_inversion.summarise_algorithm_settings()


def test_mass_matrix():
    mean_vector = [0, 0]
    covariance_matrix = [[1, 2.5], [2.5, 20]]
    multivariate_norm = MultivariateNormal(mean_vector, covariance_matrix)

    # Create an inversion instance with the MultivariateNormal distribution
    inversion = Inversion(target_distribution=multivariate_norm)
    # # Create an inversion instance
    # inversion = Inversion(target_distribution=target_distribution)

    # Define an initial position for the HMC sampler (1D example)
    initial_position = np.array(
        [0.0, 0.0]
    )  # Adjust the initial value as needed

    # Run the inversion
    inversion.run_sampler(
        hmc_sampler,
        initial_position=initial_position,  # Make sure you provide initial_position
        num_samples=100,
        num_leapfrog_steps=100,
        target_acceptance_rate=0.65,
        step_size=0.01,
    )

    # Run the inversion
    inversion.run_sampler(
        hmc_sampler,
        initial_position=initial_position,  # Make sure you provide initial_position
        num_samples=100,
        num_leapfrog_steps=100,
        target_acceptance_rate=0.65,
        step_size=0.01,
        mass_matrix=np.linalg.inv(covariance_matrix),
    )

    # Save the results in the desired format (e.g., 'numpy', 'pandas', or 'netcdf')
    inversion.save_results("inversion_results.npz", format="numpy")


def test_total_samples():
    mean_vector = [0, 0]
    covariance_matrix = [[1, 0.5], [0.5, 2]]
    multivariate_norm = MultivariateNormal(mean_vector, covariance_matrix)

    # Create an inversion instance with the MultivariateNormal distribution
    inversion = Inversion(target_distribution=multivariate_norm)
    # # Create an inversion instance
    # inversion = Inversion(target_distribution=target_distribution)

    # Define an initial position for the HMC sampler (1D example)
    initial_position = np.array(
        [0.0, 0.0]
    )  # Adjust the initial value as needed

    # Run the inversion
    inversion.run_sampler(
        hmc_sampler,
        initial_position=initial_position,  # Make sure you provide initial_position
        num_samples=1000,
        num_leapfrog_steps=10,
        target_acceptance_rate=0.4,
        step_size=0.1,
    )

    assert inversion.samples.shape == (1001, multivariate_norm.dimensionality)

    # Run the inversion again
    inversion.run_sampler(
        hmc_sampler,
        initial_position=initial_position,  # Make sure you provide initial_position
        num_samples=1000,
        num_leapfrog_steps=10,
        target_acceptance_rate=0.2,
        step_size=0.1,
    )

    assert inversion.samples.shape == (2001, multivariate_norm.dimensionality)

    # Save the results in the desired format (e.g., 'numpy', 'pandas', or 'netcdf')
    inversion.save_results("inversion_results.npz", format="numpy")

    loaded_inversion = Inversion(target_distribution=None)

    loaded_inversion.load_results("inversion_results.npz", format="numpy")

    # Run the inversion again again
    loaded_inversion.run_sampler(
        hmc_sampler,
        initial_position=initial_position,  # Make sure you provide initial_position
        num_samples=1000,
        num_leapfrog_steps=10,
        target_acceptance_rate=0.1,
        step_size=0.1,
    )

    assert loaded_inversion.samples.shape == (
        3001,
        multivariate_norm.dimensionality,
    )

    loaded_inversion.save_results("inversion_results.npz", format="numpy")

    loaded_inversion2 = Inversion.load("inversion_results.npz")

    # Run the inversion again again
    loaded_inversion2.run_sampler(
        hmc_sampler,
        initial_position=initial_position,  # Make sure you provide initial_position
        num_samples=1000,
        num_leapfrog_steps=10,
        target_acceptance_rate=0.65,
        step_size=0.1,
    )

    assert loaded_inversion2.samples.shape == (
        4001,
        multivariate_norm.dimensionality,
    )

    loaded_inversion2.plot_trace()
    plt.show()


def test_leapfrog_visualisation():
    mean_vector = [0, 0]
    covariance_matrix = [[1, 0.5], [0.5, 2]]
    multivariate_norm = MultivariateNormal(mean_vector, covariance_matrix)

    # Create an inversion instance with the MultivariateNormal distribution
    inversion = Inversion(target_distribution=multivariate_norm)
    # # Create an inversion instance
    # inversion = Inversion(target_distribution=target_distribution)

    # Define an initial position for the HMC sampler (1D example)
    initial_position = np.array(
        [0.0, 0.0]
    )  # Adjust the initial value as needed

    # Run the inversion
    inversion.run_sampler(
        hmc_sampler,
        initial_position=initial_position,  # Make sure you provide initial_position
        num_samples=1000,
        num_leapfrog_steps=10,
        target_acceptance_rate=0.4,
        step_size=0.1,
        visualise_leapfrog=True,
    )
