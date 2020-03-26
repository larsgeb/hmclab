"""Distribution classes and associated methods.
"""
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod
from typing import List as _List
from typing import Union as _Union
import warnings as _warnings
import numpy as _numpy
import scipy as _scipy
import scipy.sparse as _sparse
from hmc_tomography.Distributions import _AbstractDistribution


class LinearMatrix(_AbstractDistribution):
    """Likelihood model based on a linear forward model given as
    :math:`G \\mathbf{m} = \\mathbf{d}`
    """

    def __init__(
        self,
        dimensions: int,
        G: _Union[_numpy.ndarray, _sparse.spmatrix] = None,
        d: _numpy.ndarray = None,
        data_covariance: _Union[
            float, _numpy.ndarray, _sparse.spmatrix
        ] = None,
        use_mkl: bool = False,
        use_cupy: bool = False,
    ):
        """Constructor for linear forward model likelihood.

        Parameters
        ==========
        dimensions : int
            Dimension of the model space.
        G : numpy.ndarray or scipy.sparse.spmatrix
            Numpy ndarray or scipy sparse matrix sized as (datapoints, dimensions)
            containing the linear forward model. Defaults to unit matrix.
        d : numpy.ndarray
            Numpy ndarray shaped as (datapoints, 1) containing observed datapoints.
            Defaults to a vector of zeros.
        data_covariance : float or numpy.ndarray or scipy.sparse.spmatrix
            Optional object representing either scalar data variance, a diagonal data
            variance matrix [numpy.ndarray shaped as (datapoints, 1)], a full data
            covariance matrix [numpy.ndarray shaped as (datapoints, datapoints)] or a
            sparse data covariance matrix [scipy.sparse.spmatrix shaped as
            (datapoints, datapoints)].
        use_mkl : bool
            Whether to use Intel's Math Kernel Library for the evaluation of Sparse
            matrix-vector products. The MKL libraries need to be properly set-up
            according to the documentation provided here:
            https://github.com/larsgeb/scipy_mkl_gemv.

        """
        if use_mkl and use_cupy:
            _warnings.warn(
                f"CuPy and MKL are both requested. MKL will take precedence.",
                Warning,
            )

        self.dimensions = dimensions

        # Make sure something was passed, or generate randomly if nothing was passed --
        if G is None and d is None:
            _warnings.warn(
                f"No forward model and data was supplied. "
                f"Defaulting to a unit matrix with zero data. ",
                Warning,
            )
            G = _numpy.eye(self.dimensions)
            d = _numpy.zeros((G.shape[0], 1))
        elif G is None or d is None:
            raise ValueError(
                "Either no forward model matrix or data was passed."
                "Not sure what to do. Aborting."
            )

        # Parse forward model matrix ---------------------------------------------------
        if type(G) == _numpy.ndarray or issubclass(type(G), _sparse.spmatrix):
            # Assert that the second dimension of the matrix corresponds to model space
            # dimension.
            assert G.shape[1] == dimensions
            self.G = G
        else:
            raise ValueError(
                "The forward model matrix type was not understood."
            )

        # Parse data vector ------------------------------------------------------------
        if type(d) == _numpy.ndarray:
            # Assert that the data vector is compatible with the matrix.
            assert d.shape == (G.shape[0], 1)
            self.d = d
        else:
            raise ValueError("The data vector type was not understood.")

        # Parse data covariance --------------------------------------------------------
        self.data_covariance_matrix: bool = False
        """Attribute to determine which misfit/gradient formula is needed."""

        if data_covariance is None:
            # No given data covariance
            self.data_covariance = 1.0

        elif type(data_covariance) is float:
            # Single data covariance float paased
            self.data_covariance = data_covariance

        elif type(data_covariance) is _numpy.ndarray:
            # Numpy array passed
            if data_covariance.shape == (d.size, 1):
                # Diagonal of a data covariance matrix passed
                self.data_covariance = data_covariance

            elif data_covariance.shape == (d.size, d.size):
                # Full data covariance matrix passed
                self.data_covariance_matrix = True
                self.data_covariance = data_covariance
                # TODO implement inverse
                raise NotImplementedError(
                    "Full data covariance matrix is not implemented yet."
                )

            else:
                # Something else passed?
                raise ValueError("")

        elif issubclass(type(data_covariance), _sparse.spmatrix):
            # Sparse matrix passed
            self.data_covariance_matrix = True

            if data_covariance.shape == (d.size, d.size):
                # Sparse data covariance matrix
                raise NotImplementedError(
                    "Sparse data covariance matrix is not implemented yet."
                )
            else:
                # Something else passed?
                raise ValueError(
                    "The sparse data covariance matrix was not understood."
                )
        else:
            # Not a supported type
            raise ValueError("The data covariance type was not understood.")

        self.use_mkl = False
        self.use_cupy = False
        if use_mkl:
            try:
                # Fails with OSError if MKL is not found
                from hmc_tomography.Helpers.mkl_interface import sparse_gemv

                # MKL binding works only for sparse matrices
                if type(G) != _sparse.csr_matrix:
                    self.G = _sparse.csr_matrix(G)

                # Precompute for gradients
                self.Gt = self.G.T.tocsr()
                self.use_mkl = True

                # Bind the needed function
                self.sparse_mkl_gemv_binding = sparse_gemv

                # TODO add logic for data covariance other than floats
            except OSError:
                _warnings.warn(
                    f"MKL not found, will evaluate matrix-vector products using SciPy.",
                    Warning,
                )
                self.use_mkl = False
            except Exception as e:
                _warnings.warn(f"Not using MKL because: {e}.", Warning)
                self.use_mkl = False
        elif use_cupy:
            try:
                # Import cupy and cupyx
                import cupy as _cupy
                import cupyx as _cupyx

                # Bind the modules
                self.cupy_binding = _cupy
                self.cupyx_binding = _cupyx

                # Convert to csr if needed (if not sparse matrix, creation will fail
                # later on anyway, no need to check here)
                if issubclass(type(G), _sparse.spmatrix):
                    if type(G) != _sparse.csr_matrix:
                        G = G.tocsr()

                # Transfer matrices to GPU
                self.gpu_G = self.cupyx_binding.scipy.sparse.csr_matrix(
                    _scipy.sparse.csr_matrix(G)
                )
                self.gpu_Gt = self.gpu_G.T.tocsr()
                self.gpu_d = self.cupy_binding.asarray(d)

                self.use_cupy = True
            except Exception as e:
                _warnings.warn(f"Not using CuPy because: {e}.", Warning)
                self.use_cupy = False

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        """
        """
        if self.use_mkl and not self.data_covariance_matrix:
            # Use Intel's MKL to perform sparse csr matrix vector product
            return (
                0.5
                * _numpy.linalg.norm(
                    (
                        self.sparse_mkl_gemv_binding(self.G, coordinates)
                        - self.d
                    ),
                    ord=2,
                )
                ** 2
                / self.data_covariance
            )
        elif self.use_cupy and not self.data_covariance_matrix:
            # Use CuPy to perform sparse csr matrix vector product using sparse cuBLAS
            return (
                0.5
                * self.cupy_binding.linalg.norm(
                    self.gpu_G.dot(self.cupy_binding.asarray(coordinates))
                    - self.gpu_d,
                    # ord=2,
                )
                .get()
                .item(0)
                ** 2
                / self.data_covariance
            )
        elif not self.data_covariance_matrix:
            # Data covariance is a single scalar or a diagonal, so we move the data
            # covariance operation out of the matrix-vector products
            return (
                0.5
                * _numpy.linalg.norm((self.G @ coordinates - self.d), ord=2)
                ** 2
                / self.data_covariance
            )

        else:
            # TODO implement other case
            raise NotImplementedError(
                "We don't have the full covariance matrix implementation yet, let us"
                "know if you would like this."
            )

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        """
        """

        if self.use_mkl and not self.data_covariance_matrix:
            # Use Intel's MKL to perform sparse csr matrix vector product
            return (
                self.sparse_mkl_gemv_binding(
                    self.Gt,
                    self.sparse_mkl_gemv_binding(self.G, coordinates) - self.d,
                )
            ) / self.data_covariance
        elif self.use_cupy and not self.data_covariance_matrix:
            return (
                self.gpu_Gt.dot(
                    self.gpu_G.dot(self.cupy_binding.asarray(coordinates))
                    - self.gpu_d
                ).get()
            ) / self.data_covariance
        elif not self.data_covariance_matrix:
            # Data covariance is a single scalar, so we move the data covariance
            # operation out of the matrix-vector products
            return (
                self.G.T
                @ (self.G @ coordinates - self.d)
                / self.data_covariance
            )
        else:
            raise NotImplementedError(
                "We don't have the full covariance matrix implementation yet, let us"
                "know if you would like this."
            )

    def generate(self):
        raise NotImplementedError()
