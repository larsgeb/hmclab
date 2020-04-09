from types import FunctionType as _FunctionType
from hmc_tomography.Priors import _AbstractPrior
import numpy as _numpy
import matplotlib.pyplot as _plt
import gc as _gc

# FEM tools
import fenics as _fenics
import dolfin as _dolfin
import ufl as _ufl

from scipy.sparse import csr_matrix as _csr_matrix
from scipy.sparse import csc_matrix as _csc_matrix
from scipy.sparse import diags as _diags
from scipy import linalg as _scipy_linalg
from numpy import linalg as _numpy_linalg
from scipy import special as special
from scipy.sparse.linalg import spsolve as _spsolve
from scipy.sparse.linalg import splu as _splu


class FenicsContinuous(_AbstractPrior):
    name = "Continuous GRF prior using Fenics"
    dimensions = _numpy.inf
    annealing = 1.0

    def __init__(
        self,
        mesh: _fenics.mesh = None,
        mean: str = None,
        operators=None,
        dense=False,
        lump_for_misfit: bool = False,
    ):
        self.lump_for_misfit = lump_for_misfit
        self.dense = dense
        self.superlu = True

        # Setting the mean of the GRF ------------------------------------------
        if mean is None or mean == "erf":  # Default or erf
            self._analytic_mean = self._example_mean_erf
        elif mean == "ricker":  # Ricker
            self._analytic_mean = self._example_mean_ricker
        elif type(mean) is not str:
            # Passed something else but not expression
            raise TypeError()
        else:  # Passed a str, assume it is a valid C++ expression
            self._analytic_mean = mean

        # Setting up the mesh --------------------------------------------------
        if mesh is None:  # No mesh was passed
            self.mesh = _dolfin.UnitIntervalMesh(50)
        elif (
            all([base != _dolfin.cpp.mesh.Mesh for base in type(mesh).__bases__])
            and type(mesh) != _dolfin.cpp.mesh.Mesh
        ):  # Something was passed, but it ain't a mesh
            raise TypeError()
        else:  # A mesh was passed
            self.mesh = mesh

        # Create function spaces for FEM ---------------------------------------
        self._function_space = _fenics.FunctionSpace(self.mesh, "CG", 1)
        self._trial_function = _fenics.TrialFunction(self._function_space)
        self._test_function = _fenics.TestFunction(self._function_space)

        # Creating operators for GRF -------------------------------------------
        if operators is None or operators == "smooth":
            bilinear_component, linear_component = self._example_operator_smooth()
        elif operators == "rough":
            bilinear_component, linear_component = self._example_operator_rough()
        elif operators == "independent":
            bilinear_component, linear_component = self._example_operator_independent()
        elif type(operators) == tuple:
            bilinear_component, linear_component = self._biharmonic_operator(
                operators[0], operators[1]
            )
        elif not all([_ufl.form.Form == operator for operator in operators]):
            raise TypeError()
        else:
            bilinear_component, linear_component = operators

        # Creating the discrete operators --------------------------------------
        self.stiffness_matrix, self.forcing_vector = _fenics.assemble_system(
            bilinear_component, linear_component
        )
        self.mass_matrix = _fenics.assemble(
            self._trial_function * self._test_function * _fenics.dx
        )
        stiffness_matrix_backend = _fenics.as_backend_type(self.stiffness_matrix).mat()
        mass_matrix_backend = _fenics.as_backend_type(self.mass_matrix).mat()
        ka, kb, kc = stiffness_matrix_backend.getValuesCSR()
        self.stiffness_matrix_sparse = _csr_matrix(
            (kc, kb, ka), stiffness_matrix_backend.size
        )
        del ka, kb, kc, stiffness_matrix_backend
        ma, mb, mc = mass_matrix_backend.getValuesCSR()
        self.mass_matrix_sparse = _csr_matrix(
            (mc, mb, ma), mass_matrix_backend.size
        ).tocsc()
        del ma, mb, mc, mass_matrix_backend
        _gc.collect()

        # Set discrete amount of dimensions ------------------------------------
        self.dimensions = self.mean_discrete().size

        # Some additional operators we might need ------------------------------
        # The inverse of the lumped mass matrix might be used when one wants to
        # speed up misfit computation. However, it is superseded by a SuperLU
        # decomposition, which is more accurate and relatively fast. Typically
        # we want the misfit computation to be as accurate as possible, so it's
        # best not to take shortcuts here.
        self.mass_lumped_inverse = None
        # We don't compute the mass square root yet, as it is only needed for
        # sample generation
        self.massSqrt = None

        # SuperLU significantly speeds up repeated linear solves with the same
        # LHS, as e.g. repeated sample generation and misfit computation.
        # The reason that we don't compute both decompositions is that the mass
        # decomposition is needed for gradients and misfits, while the
        # stiffness decomposition is needed for sample generation. Not every
        # use case needs sample generation, so it is only computed the first
        # time samples are requested.
        if self.superlu:
            self.stiffness_ludecomp = None
            self.mass_ludecomp = _splu(self.mass_matrix_sparse)

    def mean(self):
        return self._analytic_mean

    def mean_discrete(self):
        mean = _fenics.Function(self._function_space)
        E = _dolfin.Expression(self.mean(), element=self._function_space.ufl_element())
        mean.interpolate(E)
        return mean.vector().get_local()[:, _numpy.newaxis]

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        if len(coordinates.shape) == 1:
            coordinates = coordinates[:, _numpy.newaxis]

        dm = coordinates - self.mean_discrete()

        if self.superlu:
            # Use SuperLU decomposition to compute misfit with the sparse
            # operators -> Fast and accurate, but requires pre-computation
            return (
                0.5
                * dm.T
                @ self.stiffness_matrix_sparse
                @ self.mass_ludecomp.solve(self.stiffness_matrix_sparse @ dm)
            ).item()
        elif not self.dense and not self.lump_for_misfit:
            # Use a direct sparse solver to compute misfit with the sparse
            # operators -> Medium fast and accurate
            return (
                0.5
                * dm.T
                @ self.stiffness_matrix_sparse
                @ _spsolve(self.mass_matrix_sparse, self.stiffness_matrix_sparse @ dm)
            ).item()
        elif not self.dense:
            # Use a direct sparse solver to compute misfit with the approximate
            # lumped sparse operators -> Fast and inaccurate

            # Is the lumped inverse mass matrix already computed?
            self.check_lumped_inverse_mass()

            return (
                0.5
                * dm.T
                @ self.stiffness_matrix_sparse
                @ self.mass_lumped_inverse
                @ self.stiffness_matrix_sparse
                @ dm
            ).item()
        else:
            # Use a direct matrix multiplication to compute misfit with the
            # dense operators -> Fast when small, extreme on memory, and
            # accurate
            return (
                0.5 * _numpy.linalg.norm(self.combined_operator_inv @ dm) ** 2
            ).item()

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        if len(coordinates.shape) == 1:
            coordinates = coordinates[:, _numpy.newaxis]

        if self.superlu:
            return self.stiffness_matrix_sparse @ self.mass_ludecomp.solve(
                self.stiffness_matrix_sparse @ (coordinates - self.mean_discrete())
            )
        elif not self.dense and not self.lump_for_misfit:
            return (
                self.stiffness_matrix_sparse
                @ _spsolve(
                    self.mass_matrix_sparse,
                    self.stiffness_matrix_sparse @ (coordinates - self.mean_discrete()),
                )[:, _numpy.newaxis]
            )
        elif not self.dense:
            # Is the lumped inverse mass matrix already computed?
            self.check_lumped_inverse_mass()

            return (
                self.stiffness_matrix_sparse
                @ self.mass_lumped_inverse
                @ self.stiffness_matrix_sparse
                @ (coordinates - self.mean_discrete())
            )
        else:
            return (
                self.combined_operator_inv.T
                @ self.combined_operator_inv
                @ (coordinates - self.mean_discrete())
            )

    def corrector(self, coordinates: _numpy.ndarray, momentum: _numpy.ndarray):
        pass

    def plot_mesh(self):
        _dolfin.plot(self.mesh)

    def full_matrix_inv(self):
        """
        Please don't do this for big matrices.

        Returns
        -------

        """
        return (
            self.stiffness_matrix_sparse.todense()
            @ _numpy.linalg.inv(self.mass_matrix_sparse.todense())
            @ self.stiffness_matrix_sparse.todense()
        )

    def full_matrix(self):
        """
        Please don't do this for big matrices.

        Returns
        -------

        """
        inv_stiff = _numpy.linalg.inv(self.stiffness_matrix_sparse.todense())
        return inv_stiff @ self.mass_matrix_sparse.todense() @ inv_stiff

    def plot_operators(self):
        if self.mass_matrix_sparse.shape[0] < 500:
            # Don't do this for large systems
            _plt.subplot(121)
            stiffness_matrix_dense = self.stiffness_matrix_sparse.todense()
            extremum = _numpy.max(_numpy.abs(stiffness_matrix_dense))
            _plt.imshow(stiffness_matrix_dense, vmin=-extremum, vmax=extremum)
            _plt.title("Stiffness matrix", size=30)
            _plt.colorbar()
            _plt.subplot(122)
            mass_matrix_dense = self.mass_matrix_sparse.todense()
            extremum = _numpy.max(_numpy.abs(mass_matrix_dense))
            _plt.imshow(mass_matrix_dense, vmin=0, vmax=extremum)
            _plt.title("Mass matrix", size=30)
            _plt.colorbar()
        else:
            _plt.subplot(121)
            _plt.spy(self.stiffness_matrix_sparse, markersize=1)
            _plt.title("Stiffness matrix (sparse)", size=30)
            _plt.subplot(122)
            _plt.spy(self.mass_matrix_sparse, markersize=1)
            _plt.title("Mass matrix (sparse)", size=30)

        _plt.show()

    def plot_combined_operators(self):
        _plt.subplot(121)
        _plt.imshow(self.massSqrt)
        _plt.title("Square root mass matrix", size=30)
        _plt.subplot(122)
        _plt.imshow(self.combined_operator)
        _plt.title("Stiffness matrix", size=30)
        _plt.show()

    def greens_function_analysis(self, x=None):
        # A few different Greens function source locations
        if x is None:

            ix = _numpy.random.choice(
                self.mesh.coordinates().shape[0], 4, replace=False
            )
            x = self.mesh.coordinates()[ix, :]

        for i in range(x.__len__()):
            # Create the solution function
            solution = _fenics.Function(self._function_space)

            # Create a new RHS vector for every solve.
            forcing_vector_copy = self.forcing_vector.copy()

            # Create the source (a Dirac Delta, for the Green's function)
            dirac_source = _fenics.PointSource(
                self._function_space, _fenics.Point(x[i]), 1.0
            )

            # Add the dirac function to the forcing vector. Note that if this is
            # applied consecutively on the same object, two sources are added!
            # That's why we make a copy of the forcing_vector.
            dirac_source.apply(forcing_vector_copy)

            # Solve the PDE
            _fenics.solve(self.stiffness_matrix, solution.vector(), forcing_vector_copy)

            # Compute values on mesh

            # Plot the Green's functions
            _plt.subplot(int(x.__len__()), 1, i + 1)
            if solution.function_space().mesh().geometry().dim() == 1:
                mesh_values = solution.compute_vertex_values(self.mesh)
                _plt.plot(
                    self.mesh.coordinates()[
                        _numpy.argsort(self.mesh.coordinates()[:, 0])
                    ],
                    mesh_values[_numpy.argsort(self.mesh.coordinates()[:, 0])],
                )
            elif solution.function_space().mesh().geometry().dim() == 2:
                plot_2d_solution(solution)
                _plt.colorbar()

        _plt.tight_layout()

    def plot_samples(self, samples, alpha=1.0, color="k"):
        for i in range(samples.shape[1]):
            sample_values = samples[:, i]

            _plt.plot(
                self.mesh.coordinates()[_numpy.argsort(self.mesh.coordinates()[:, 0])],
                sample_values[_numpy.argsort(self.mesh.coordinates()[:, 0])],
                color=color,
                alpha=alpha,
            )

        # _plt.plot(
        #     self.mesh.coordinates()[
        #         _numpy.argsort(self.mesh.coordinates()[:, 0])
        #     ],
        #     mean_vec,
        #     "r",
        #     alpha=0.5,
        # )
        _plt.xlim([0, 1])
        # _plt.ylim([-0.5, 0.5])

    def generate_n(self, n_samples):
        if self.massSqrt is None:
            if self.dense:
                # If we're using dense matrices, it makes sense to precompute
                # the complete covariance operator and its inverse.
                self.massSqrt = _scipy_linalg.sqrtm(self.mass_matrix_sparse.todense())
                self.combined_operator = _spsolve(
                    self.stiffness_matrix_sparse, self.massSqrt
                )
                self.combined_operator_inv = _numpy_linalg.inv(self.combined_operator)
            else:
                # If we allow lumping of the mass matrix, we can make the
                # generation of samples sparse, and therefore scale much better
                # with mesh refinements.
                self.massSqrt = _diags(
                    _numpy.asarray(
                        _numpy.sqrt(self.mass_matrix_sparse.sum(axis=1))
                    ).squeeze(),
                    format="csc",
                )

        rnd = _numpy.random.randn(self.massSqrt.shape[0], n_samples)

        if self.superlu and self.stiffness_ludecomp is None:
            self.stiffness_ludecomp = _splu(self.stiffness_matrix_sparse)

        if self.superlu:
            samples = (
                self.stiffness_ludecomp.solve(self.massSqrt @ rnd)
                + self.mean_discrete()
            )
        elif not self.dense:
            # There is a bug in SciPy where the input of spsolve shaped as
            # (d, 1) is returned as (d, ). See:
            # https://github.com/scipy/scipy/pull/8773 and
            # https://github.com/scipy/scipy/issues/8772 ,
            # this behaviour should be fixed in 1.4.0
            if n_samples == 1:
                samples = (
                    _spsolve(self.stiffness_matrix_sparse, self.massSqrt @ rnd)[
                        :, _numpy.newaxis
                    ]
                    + self.mean_discrete()
                )
            else:
                samples = (
                    _spsolve(self.stiffness_matrix_sparse, self.massSqrt @ rnd)
                    + self.mean_discrete()
                )
        else:
            samples = self.combined_operator @ rnd + self.mean_discrete()
        return samples

    def generate(self) -> _numpy.ndarray:
        return self.generate_n(1)

    @staticmethod
    def _example_mean_ricker():
        return """(
            1.0
            * (1.0 - 2.0 * pi * pi * (x[0]*x[0]))
            * exp(-pi * pi * (x[0]*x[0]))
        )"""

    @staticmethod
    def _example_mean_ricker_2d():
        return """(
            5.0
            * (1.0 - 2.0 * pi * pi * (x[0]*x[0]+x[1]*x[1]))
            * exp(-pi * pi * (x[0]*x[0]+x[1]*x[1]))
        )"""

    @staticmethod
    def _example_mean_erf():
        return "1 + erf(10 * (x[0] - 0.5))"

    def _example_operator_smooth(self):
        return self._biharmonic_operator(4, 0.1)

    def _example_operator_rough(self):
        return self._biharmonic_operator(10, 1e-3)

    def _biharmonic_operator(self, alpha, beta):
        alpha = _fenics.Constant(alpha)
        beta = _fenics.Constant(beta)
        bilinear_component = (
            alpha
            * (
                _fenics.dot(
                    beta * _fenics.grad(self._trial_function),
                    _fenics.grad(self._test_function),
                )
                + self._trial_function * self._test_function
            )
            * _fenics.dx
        )
        linear_component = _fenics.Constant(0) * self._test_function * _fenics.dx
        return bilinear_component, linear_component

    def _example_operator_independent(self):
        alpha = _fenics.Constant(4)

        bilinear_component = (
            alpha * self._trial_function * self._test_function * _fenics.dx
        )

        linear_component = _fenics.Constant(0) * self._test_function * _fenics.dx
        return bilinear_component, linear_component

    def sample_to_function(self, sample):
        solution = _fenics.Function(self._function_space)
        solution.vector().set_local(sample)
        return solution

    def check_lumped_inverse_mass(self):
        if self.lump_for_misfit and (self.mass_lumped_inverse is None):
            # Computing the inverse of the lumped mass matrix. Lumping is done
            # using basic row summation.
            self.mass_lumped_inverse = _diags(
                _numpy.asarray(1.0 / (self.mass_matrix_sparse.sum(axis=1))).squeeze()
            )


def mesh2triangulation(mesh):
    import matplotlib.tri as tri

    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())


def plot_2d_solution(obj):
    _plt.gca().set_aspect("equal")
    if isinstance(obj, _dolfin.Function):
        mesh = obj.function_space().mesh()
        seismic_cmap = _plt.get_cmap("seismic")
        if mesh.geometry().dim() != 2:
            raise AttributeError
        if obj.vector().size() == mesh.num_cells():
            C = obj.vector().array()
            extremum = _numpy.max(_numpy.abs(C))
            _plt.tripcolor(
                mesh2triangulation(mesh),
                C,
                vmin=-extremum,
                vmax=extremum,
                cmap=seismic_cmap,
            )
            _plt.colorbar()
        else:
            C = obj.compute_vertex_values(mesh)
            extremum = _numpy.max(_numpy.abs(C))
            _plt.tripcolor(
                mesh2triangulation(mesh),
                C,
                vmin=-extremum,
                vmax=extremum,
                cmap=seismic_cmap,
                shading="gouraud",
            )
            _plt.colorbar()
    elif isinstance(obj, _dolfin.Mesh):
        if obj.geometry().dim() != 2:
            raise AttributeError
        _plt.triplot(mesh2triangulation(obj), color="k")
