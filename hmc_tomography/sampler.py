"""
Sampler classes and associated methods.
"""
import numpy, yaml
import hmc_tomography.priors as priors


class sampler():
    """Hamiltonian Monte Carlo class.

    """

    def __init__(self, config_file_path, quiet=False):
        """Constructor for an Hamiltonian Monte Carlo sampler object.

        Parameters
        ----------
        config_file_path : basestring
            Path to configuration file for sampler.

        """

        # Loading and parsing configuration ----------------------------------------------------------------------------
        # Open the configuration file.
        with open(config_file_path, 'r') as config_file:
            cfg = yaml.load(config_file, Loader=yaml.FullLoader)

        # Show an overview of the input file.
        if quiet is not True:
            print("Sections in configuration file:")
            for section in cfg:
                print("{:<20} {:>20} ".format(section, cfg[section]))

        # Parse dimensions
        if 'dimensions' in cfg:
            dimensions = int(cfg['dimensions'])
        else:
            raise Exception('Invalid configuration file.'
                            'Missing *AT LEAST* the dimensions of the inverse problem.'
                            'YML key: dimensions.')

        # Assign HMC variables
        self.prior = priors.normal
        self.momentum = numpy.zeros((dimensions, 1))
        self.position = numpy.zeros((dimensions, 1))
        self.mass_matrix = numpy.eye(dimensions, dimensions)

    def kinetic_energy(self, momentum: numpy.ndarray) -> float:
        """Function to compute kinetic energy for a given momentum.

        This method computes the kinetic energy associated with the given input momentum vector based on the Gaussian
        kinetic energy distribution.

        Parameters
        ----------
        momentum : numpy.ndarray
            momentum vector for which to compute the kinetic energy.

        Returns
        -------
        float
            Scalar value of kinetic energy


        """
        return .5 * numpy.asscalar(momentum.T @ self.mass_matrix @ momentum)
