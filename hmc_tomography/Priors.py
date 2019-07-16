"""
Prior distributions available to the HMC sampler.
"""


class Normal:
    """Normal distribution in model space.

    """

    def __init__(self, means, covariance):
        """

        Parameters
        ----------
        means : numpy.ndarray
            Means vector of the normal distribution.

        covariance : numpy.ndarray
            Covariance matrix of the normal distribution.

        """
        super().__init__()
        self.name = "Gaussian prior"
        self.means = means
        self.covariance = covariance

    def prior_misfit(self, position):
        """

        Parameters
        ----------
        position : numpy.ndarray
            Position vector to calculate prior misfit at.

        Returns
        -------

        """
        return (self.means - position).T @ (self.covariance @ (self.means - position))


class Priors:
    """Normal distribution in logarithmic model space.

    """

    def __init__(self, means, covariance):
        """

        Parameters
        ----------
        means
        covariance
        """
        super().__init__()
        self.name = "Log normal (logarithmic Gaussian) prior"
        self.means = means
        self.covariance = covariance

    def prior_misfit(self, position):
        """

        Parameters
        ----------
        position

        Returns
        -------

        """
        return (self.means - position).T @ (self.covariance @ (self.means - position))
