from hmclab.Algorithms import hmc_sampler
from hmclab.Distributions import MultivariateNormal
from hmclab import Inversion
import numpy as np, matplotlib.pyplot as plt


def test_basic_inversion():
    mean_vector = [0, 0]
    covariance_matrix = [[1, 0.5], [0.5, 2]]
    multivariate_norm = MultivariateNormal(mean_vector, covariance_matrix)

    print(multivariate_norm)
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
        num_samples=10000,
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
