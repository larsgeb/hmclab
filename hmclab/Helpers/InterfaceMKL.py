"""This file is taken from https://github.com/larsgeb/scipy_mkl_gemv and distributed
under the GPL-3.0 license.

Tiny wrapper for Intel's MKL sparse BLAS to communicate with SciPy CSR matrices.

Observed speedup approximately 5x in favourable conditions on a 6-core laptop.

Make sure that the libraries for MKL are in your path. On a typical install that would
mean running the following commands before executing anything in Python:

```
export INTEL_COMPILERS_AND_LIBS=/opt/intel/compilers_and_libraries/linux
source $INTEL_COMPILERS_AND_LIBS/mkl/bin/mklvars.sh intel64
```

To-do list:
 - mkl.mkl_sparse_d_mv  # alternative mkl functions (not deprecated)
 - mkl.mkl_sparse_s_mv  # alternative mkl functions (not deprecated)
 - mkl.mkl_sparse_spmm  # Sparse times sparse matrix, with single op
 - mkl.mkl_sparse_sp2m  # Sparse times sparse matrix, with additional op

These are harder to implement because MKL requires 4-array CSRs, as well as passing the
objects as MKL types.

"""
from ctypes import POINTER, byref, c_double, c_float, c_int, cdll

import numpy as np
import scipy.sparse as sparse

# Load DLL
try:
    mkl = cdll.LoadLibrary("libmkl_rt.so")
except OSError:
    mkl = None


def sparse_gemv(A: sparse.csr_matrix, x: np.ndarray) -> np.ndarray:
    """Delegator function for double or single precision gemv.

    Parameters
    ==========
    A : scipy.sparse.csr_matrix
        A (sparse) scipy csr matrix of dimensions (i, j) with datatype float32 or
        float64. The datatype of A will determine operating precision.
    x : numpy.ndarray
        A (dense) numpy ndarray of dimension (j, k) with datatype float32 or
        float64. If x's datatype does not match A's datatype, x is converted.

    Returns
    =======
    y : numpy.ndarray
        A (dense) numpy ndarray of dimension (i, k) with datatype float32 or
        float64, depending on A.
    """
    if not sparse.isspmatrix_csr(A):
        raise ValueError("Matrix must be in csr format")

    if A.dtype == "float64":
        return double_sparse_gemv_via_MKL(A, x)
    elif A.dtype == "float32":
        return single_sparse_gemv_via_MKL(A, x)
    else:
        raise ValueError("Data type of the matrix not understood")


def double_sparse_gemv_via_MKL(A, x):
    """Double precision sparse GEMV, M being sparse, V dense."""
    # Check input types
    if not sparse.isspmatrix_csr(A):
        raise ValueError("Matrix must be in csr format.")
    if A.dtype != "float64":
        raise ValueError("Matrix must be of datatype float64.")
    if x.dtype.type is not np.double:
        x = x.astype(np.double, copy=True)

    # Matrix dimensions
    (i, j) = A.shape

    # Explanation: https://stackoverflow.com/a/52299730/6848887
    # The data of the matrix
    # data is an array containing all the non zero elements of the sparse matrix.
    # indices is an array mapping each element in data to its column in the sparse
    # matrix.
    # indptr then maps the elements of data and indices to the rows of the sparse
    # matrix.
    data = A.data.ctypes.data_as(POINTER(c_double))
    indptr = A.indptr.ctypes.data_as(POINTER(c_int))
    indices = A.indices.ctypes.data_as(POINTER(c_int))

    # Creating the output array y
    k = 1
    if x.ndim == 1:
        # Shape is hopefully (j, ), we can work with this
        y = np.empty(i, dtype=np.double, order="F")

    elif x.shape[1] == 1:
        # Shape is hopefully (j, 1)
        y = np.empty((i, 1), dtype=np.double, order="F")

    else:
        # Shape is hopefully (j, k)
        k = x.shape[1]
        y = np.empty((i, k), dtype=np.double, order="F")

    # Assert that first dimension matches
    if x.shape[0] != j:
        raise ValueError(f"Vector x must have j entries. x.size is {x.size}, j is {j}")

    # Put vector x in column-major order
    if not x.flags["F_CONTIGUOUS"]:
        x = x.copy(order="F")

    # If x is a vector, perform the operation once
    if k == 1:
        pointer_x = x.ctypes.data_as(POINTER(c_double))
        pointer_y = y.ctypes.data_as(POINTER(c_double))

        mkl.mkl_cspblas_dcsrgemv(
            "N", byref(c_int(i)), data, indptr, indices, pointer_x, pointer_y
        )
    else:
        for columns in range(k):
            xx = x[:, columns]
            yy = y[:, columns]
            pointer_x = xx.ctypes.data_as(POINTER(c_double))
            pointer_y = yy.ctypes.data_as(POINTER(c_double))
            mkl.mkl_cspblas_dcsrgemv(
                "N", byref(c_int(i)), data, indptr, indices, pointer_x, pointer_y
            )

    return y


def single_sparse_gemv_via_MKL(A, x):
    """Single precision sparse GEMV, M being sparse, V dense."""
    # Check input types
    if not sparse.isspmatrix_csr(A):
        raise ValueError("Matrix must be in csr format.")
    if A.dtype != "float32":
        raise ValueError("Matrix must be of datatype float32.")
    if x.dtype.type is not np.single:
        x = x.astype(np.single, copy=True)

    # Matrix dimensions
    (i, j) = A.shape

    # Explanation: https://stackoverflow.com/a/52299730/6848887
    # The data of the matrix
    # data is an array containing all the non zero elements of the sparse matrix.
    # indices is an array mapping each element in data to its column in the sparse
    # matrix.
    # indptr then maps the elements of data and indices to the rows of the sparse
    # matrix.
    data = A.data.ctypes.data_as(POINTER(c_float))
    indptr = A.indptr.ctypes.data_as(POINTER(c_int))
    indices = A.indices.ctypes.data_as(POINTER(c_int))

    # Creating the output array y
    k = 1
    if x.ndim == 1:
        # Shape is hopefully (j, ), we can work with this
        y = np.empty(i, dtype=np.single, order="F")

    elif x.shape[1] == 1:
        # Shape is hopefully (j, 1)
        y = np.empty((i, 1), dtype=np.single, order="F")

    else:
        # Shape is hopefully (j, k)
        k = x.shape[1]
        y = np.empty((i, k), dtype=np.single, order="F")

    # Assert that first dimension matches
    if x.shape[0] != j:
        raise ValueError(f"Vector x must have j entries. x.size is {x.size}, j is {j}")

    # Put vector x in column-major order
    if not x.flags["F_CONTIGUOUS"]:
        x = x.copy(order="F")

    # If x is a vector, perform the operation once
    if k == 1:
        pointer_x = x.ctypes.data_as(POINTER(c_float))
        pointer_y = y.ctypes.data_as(POINTER(c_float))

        mkl.mkl_cspblas_scsrgemv(
            "N", byref(c_int(i)), data, indptr, indices, pointer_x, pointer_y
        )
    else:
        for columns in range(k):
            xx = x[:, columns]
            yy = y[:, columns]
            pointer_x = xx.ctypes.data_as(POINTER(c_float))
            pointer_y = yy.ctypes.data_as(POINTER(c_float))
            mkl.mkl_cspblas_scsrgemv(
                "N", byref(c_int(i)), data, indptr, indices, pointer_x, pointer_y
            )

    return y
