"""
Prior distributions available to the HMC sampler.
"""


class normal():
    """Normal distribution in model space.

    """

    def __init__(self, means, covariance):
        """

        Parameters
        ----------
        means
        covariance
        """
        super().__init__()
        self.name = "Gaussian prior"
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


class log_normal():
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
