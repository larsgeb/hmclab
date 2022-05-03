==========
Optimizers
==========

.. automodule:: hmclab.Optimizers
    :no-members:

Available methods within `Optimizers`:

.. autosummary:: 
   :toctree: _autosummary
   :nosignatures:
   :recursive:

    gradient_descent
   
   
   

Using SciPy's optimization routines
***********************************

It would be a waste to not use the great array of optimization routines available
through ``scipy.optimize.minimize``. To run these, SciPy needs to be supplied with 2 
function handles:
- a function that computes the objective 
- a function that computes the gradient of the objective

The input of these functions should be ``(n, )`` arrays, not the column arrays
``(n, 1)`` used throughout hmclab. Similarly, SciPy requires the return of the
gradient to be an array of shape ``(n, )``, so we'll have to adapt the inputs to both
functions and the gradient returned from the gradient function implemented for
hmclab.

This is an easy way to extract the methods and convert the arguments needed to run
SciPy's routines, given that you constructed a target distribution:

.. code-block:: python

   def misfit(m):
       # Converts the input column vector (n, 1) to (n, ) vector
       return posterior.misfit(m[:, None])
   
   
   def gradient(m):
       # Converts the input column vector (n, 1) to (n, ) vector and
       # creates a column vector (n, 1) from the returned (n, ) vector
       return posterior.gradient(m[:, None])[:, 0]

These functions can now be used in e.g. SciPy's L-BFGS algorithm:

.. code-block:: python

   from scipy.optimize import minimize

   options = {"ftol": 1e-4, "disp": True, "maxiter": 100, "maxcor": 20}
   
   result = minimize(
       misfit,
       starting_model,
       method="L-BFGS-B",
       jac=gradient,
       options=options,
   )
