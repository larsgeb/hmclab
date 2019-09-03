from abc import ABC, abstractmethod
from hmc_tomography.Targets import Target
from SalvusWrap import wrapper
import numpy
import hashlib


class Salvus(Target):

    name = "FWI Salvus inverse problem"
    dimensions = -1
    annealing = 1.0

    def __init__(
        self, toml_file: str, annealing: float = 1, quiet: bool = True
    ):
        """

        Parameters
        ----------
        annealing
        """
        self.wrapper = wrapper.SimpleWrap(toml_file)
        self.dimensions = self.wrapper.free_parameters
        self.get_model_vector = self.wrapper.get_model_vector
        self.hash = ""
        self._misfit = numpy.nan
        self.annealing = annealing
        self.quiet = quiet

    def misfit(self, coordinates: numpy.ndarray) -> float:
        """

        Parameters
        ----------
        coordinates

        Returns
        -------

        """
        if coordinates.shape != (self.dimensions, 1):
            raise ValueError()
        # coordinates.flags.writeable = False

        self.wrapper.set_model_vector(coordinates)

        if self.hash != hashlib.sha1(coordinates).hexdigest():
            self.wrapper.run(quiet=self.quiet)
            self._misfit = self.wrapper.compute_misfit()
            self.hash = hashlib.sha1(coordinates).hexdigest()

        print("Misfit:  %7.5e" % (self._misfit / self.annealing))
        return self._misfit / self.annealing

    def gradient(self, coordinates: numpy.ndarray) -> numpy.ndarray:
        """

        Parameters
        ----------
        coordinates

        Returns
        -------

        """

        # coordinates.flags.writeable = False
        if self.hash != hashlib.sha1(coordinates).hexdigest():
            self.misfit(coordinates)

        self.wrapper.run_adjoint(quiet=self.quiet)
        return self.wrapper.get_model_vector_gradient() / self.annealing
