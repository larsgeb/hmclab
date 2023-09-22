"""
Inversion.py

This module provides a class for running and managing inversion processes.

Author: Lars Gebraad
"""
import matplotlib.pyplot as plt
import numpy as np

from .Distributions.base import AbstractDistribution

__all__ = ["Inversion"]


class Inversion:
    """
    Inversion class for running and managing inversion processes.

    Parameters:
        target_distribution (object): The target distribution to be inverted.

    Attributes:
        target_distribution (object): The target distribution.
        samples (numpy.ndarray): The samples generated by the inversion
        process.
        algorithm_settings (dict): Settings for the inversion algorithm.
    """

    def __init__(
        self, target_distribution: AbstractDistribution | None = None
    ):
        """
        Initialize the Inversion object with a target distribution.

        Args:
            target_distribution (object): The target distribution to be
            inverted.
        """
        if target_distribution is not None:
            assert isinstance(target_distribution, AbstractDistribution)
        self.target_distribution = target_distribution
        self.samples = None

    def run_sampler(self, sampler, **sampler_kwargs):
        """
        Run the sampler with provided arguments and store the samples.

        Args:
            sampler (function): The sampling algorithm function.
            **sampler_kwargs: Keyword arguments for the sampling algorithm.
        """
        self.algorithm_settings = sampler_kwargs.copy()
        self.algorithm_settings["algorithm"] = str(sampler.__name__)

        # Run the sampler with provided arguments
        samples, acceptance_rates, log_probs = sampler(
            self.target_distribution, **sampler_kwargs
        )

        if self.samples is None:
            self.samples = samples
            self.acceptance_rates = acceptance_rates
            self.log_probs = log_probs
        else:
            self.samples = np.vstack((self.samples, samples[1:, :]))
            self.acceptance_rates = np.hstack(
                (self.acceptance_rates, acceptance_rates[1:])
            )
            self.log_probs = np.hstack((self.log_probs, log_probs[1:]))

    def save_results(self, filename, format="numpy"):
        """
        Save inversion results to a file in the specified format.

        Args:
            filename (str): The name of the file to save results to.
            format (str): The format in which to save results ('numpy'
            supported).
        """
        if format == "numpy":
            self.save_results_numpy(filename)
        else:
            raise ValueError(
                "Unsupported format. Choose 'numpy'."
            )  # pragma: no cover

    def load_results(self, filename, format="numpy"):
        if format == "numpy":
            self.load_results_numpy(filename)
        else:
            raise ValueError(
                "Unsupported format. Choose 'numpy'."
            )  # pragma: no cover

    @classmethod
    def load(cls, filename):
        """
        Class method to load an Inversion instance from a file.

        Args:
            filename (str): The name of the file to load the Inversion
            instance from.

        Returns:
            Inversion: An Inversion instance loaded from the file.
        """
        # Create an instance of the Inversion class
        inversion = cls()

        # Load inversion results from the specified file
        inversion.load_results_numpy(filename)

        return inversion

    def save_results_numpy(self, filename):
        serialized_target_distribution = dill.dumps(self.target_distribution)

        # Save inversion results to a NumPy file, including algorithm settings
        # as a dictionary
        data_to_save = {
            "target_distribution": serialized_target_distribution,
            "target_name": self.target_distribution.name,
            "samples": self.samples,
            "acceptance_rates": self.acceptance_rates,
            "log_probs": self.log_probs,
            "algorithm_settings": self.algorithm_settings,
        }
        np.savez(
            filename, **data_to_save, allow_pickle=True
        )  # Allow pickle serialization

    def load_results_numpy(self, filename):
        # Load inversion results from a NumPy file, including algorithm
        # settings as a dictionary
        data = np.load(
            filename, allow_pickle=True
        )  # Allow pickle deserialization

        serialized_target_distribution = data["target_distribution"]
        self.target_distribution = dill.loads(serialized_target_distribution)

        self.target_name = data["target_name"]
        self.samples = data["samples"]
        self.acceptance_rates = data["acceptance_rates"]
        self.log_probs = data["log_probs"]

        # Retrieve the algorithm_settings dictionary
        # self.algorithm_settings = data['algorithm_settings']
        self.algorithm_settings = data["algorithm_settings"].item()

    def plot_trace(self):
        # Plot trace plots for the samples
        plt.figure(figsize=(12, 6))
        plt.title(f"Trace Plots for {self.target_name}")
        plt.xlabel("Sample Number")
        plt.ylabel("Sample Value")
        for i in range(self.samples.shape[1]):
            plt.plot(self.samples[:, i], label=f"Parameter {i+1}")
        plt.legend()
        plt.grid(True)
        plt.show(block=False)

    def plot_subset_marginal_distributions(
        self, dimensions=None, bins=30, color="skyblue", grid=True
    ):
        # Plot marginal distributions of selected dimensions

        if dimensions is None:
            dimensions = range(self.samples.shape[1])

        num_dimensions = len(dimensions)

        fig, axes = plt.subplots(
            1, num_dimensions, figsize=(4 * num_dimensions, 4)
        )

        for i, dim in enumerate(dimensions):
            ax = axes[i]
            ax.hist(
                self.samples[:, dim],
                bins=bins,
                density=True,
                color=color,
                alpha=0.7,
            )
            ax.set_xlabel(f"Parameter {dim + 1}")
            ax.set_ylabel("Density")

            if grid:
                ax.grid(True)  # Enable grid for the current subplot

        plt.tight_layout()
        plt.show(block=False)

    def plot_subset_pairwise_scatterplot(
        self, dimensions=None, color="skyblue", grid=True
    ):
        # Create a pairwise scatterplot for selected dimensions

        if dimensions is None:
            dimensions = range(self.samples.shape[1])

        num_dimensions = len(dimensions)

        fig, axes = plt.subplots(
            num_dimensions,
            num_dimensions,
            figsize=(4 * num_dimensions, 4 * num_dimensions),
        )

        for i, dim_i in enumerate(dimensions):
            for j, dim_j in enumerate(dimensions):
                ax = axes[i, j]
                if dim_i == dim_j:
                    ax.hist(
                        self.samples[:, dim_i], bins=30, color=color, alpha=0.7
                    )
                else:
                    ax.scatter(
                        self.samples[:, dim_i],
                        self.samples[:, dim_j],
                        color=color,
                        alpha=0.7,
                    )
                ax.set_xlabel(f"Parameter {dim_i + 1}")
                ax.set_ylabel(f"Parameter {dim_j + 1}")

                if grid:
                    ax.grid(True)  # Enable grid for the current subplot

        plt.tight_layout()
        plt.show(block=False)

    def compute_mcmc_statistics(self):
        # Compute and print MCMC statistics for each parameter
        statistics = {
            "Parameter": [],
            "Mean": [],
            "Std. Deviation": [],
            "Auto-correlation (lag 1)": [],
        }

        for i in range(self.samples.shape[1]):
            parameter_samples = self.samples[:, i]
            parameter_name = f"Parameter {i+1}"
            mean = np.mean(parameter_samples)
            std_deviation = np.std(parameter_samples)
            autocorr_lag1 = np.corrcoef(
                parameter_samples[:-1], parameter_samples[1:]
            )[0, 1]

            statistics["Parameter"].append(parameter_name)
            statistics["Mean"].append(mean)
            statistics["Std. Deviation"].append(std_deviation)
            statistics["Auto-correlation (lag 1)"].append(autocorr_lag1)

        print("MCMC Statistics:")
        for param, mean, std, autocorr in zip(
            statistics["Parameter"],
            statistics["Mean"],
            statistics["Std. Deviation"],
            statistics["Auto-correlation (lag 1)"],
        ):
            print(f"{param}:")
            print(f"  Mean: {mean}")
            print(f"  Std. Deviation: {std}")
            print(f"  Auto-correlation (lag 1): {autocorr}")

    def plot_autocorrelation(self):
        # Plot auto-correlation plots for each parameter
        plt.figure(figsize=(12, 6))
        plt.title(f"Auto-correlation Plots for {self.target_name}")
        plt.xlabel("Lag")
        plt.ylabel("Auto-correlation")
        for i in range(self.samples.shape[1]):  # Corrected indexing here
            plt.acorr(
                self.samples[:, i].flatten(),
                maxlags=50,
                label=f"Parameter {i+1}",
            )
        plt.legend()
        plt.grid(True)
        plt.show(block=False)

    def summarise_algorithm_settings(self):
        if hasattr(self, "algorithm_settings"):
            summary = ""
            max_key_length = max(
                len(key) for key in self.algorithm_settings.keys()
            )

            for key, value in self.algorithm_settings.items():
                summary += f"{key.ljust(max_key_length)}: {value}\n"

            print(summary)
        else:
            print("Algorithm_settings not available.")  # pragma: no cover
