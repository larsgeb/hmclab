.. *****
.. Tests
.. *****

.. Our package contains a testing module which is not imported by default. To
.. access it, manual importing is required::

..     import hmc_tomography.Tests as Tests

.. All of our testing is automated with ``pytest``. However, one can still run a
.. single test from any script. This might be useful when you want to test one of
.. your own implemented classes.

.. The full testing reference can be found here:

.. .. toctree::
..     :maxdepth: 1

..     tests.reference

.. Running tests using pytest
.. **************************

.. When ``pytest`` is called on the project, it looks for files with the pattern
.. ``test_*.py``. Subsequently every method starting with ``test_`` is executed.
.. The ``pytest`` decorators (the bits before the method that start with an `@`)
.. make sure that we execute the function multiple times with different arguments,
.. as well as allowing ``NotImplementedErrors`` (why?
.. `to allow for unimplemented classes`_).

.. Let's take a look at a specific test, the generation of momenta:

.. .. literalinclude:: ../../hmc_tomography/Tests/test_mass_matrices.py
..     :language: python
..     :caption: hmc_tomography/Tests/test_mass_matrices.py [lines 25-53]
..     :lines: 25-53
..     :linenos:

.. Now to run the test from anywhere on your system after installing the package
.. with ``pip``, make sure that you are in the correct Python environment (which
.. in my case is the Conda environment ``hmc-tom``), and execute
.. ``pytest --pyargs hmc_tomography``::

..     ~ via 🅒 hmc-tom
..     ➜ pytest --pyargs hmc_tomography
..     ================================= test session starts ==================================
..     platform linux -- Python 3.7.3, pytest-5.2.0, py-1.8.0, pluggy-0.13.0
..     rootdir: /home/larsgebraad
..     collected 32 items

..     [...]/hmc_tomography/Tests/test_mass_matrices.py X [  3%]
..     XXxXXXxXXXxXXXxXXXxXXXxXXXxXXXx                                                  [100%]

..     =================================== warnings summary ===================================
..     [...]/hmc_tomography/Tests/test_mass_matrices.py::test_creation[1-LBFGS]
..     [...]/hmc_tomography/Tests/test_mass_matrices.py::test_creation[10-LBFGS]
..     [...]/hmc_tomography/Tests/test_mass_matrices.py::test_creation[100-LBFGS]
..     [...]/hmc_tomography/Tests/test_mass_matrices.py::test_creation[1000-LBFGS]
..     [...]/hmc_tomography/Tests/test_mass_matrices.py::test_generate[1-LBFGS]
..     [...]/hmc_tomography/Tests/test_mass_matrices.py::test_generate[10-LBFGS]
..     [...]/hmc_tomography/Tests/test_mass_matrices.py::test_generate[100-LBFGS]
..     [...]/hmc_tomography/Tests/test_mass_matrices.py::test_generate[1000-LBFGS]
..     [...]/hmc_tomography/MassMatrices.py:182: Warning: The LBFGS-style mass matrix did
..         either not receive a starting coordinate or a starting gradient. We will use a
..         random initial point and gradient.
..         Warning,

..     -- Docs: https://docs.pytest.org/en/latest/warnings.html
..     ====================== 8 xfailed, 24 xpassed, 8 warnings in 0.43s ======================

.. The tests can also be executed without installing the package. Just make sure
.. you have a local copy of the repository, and your terminal is opened in the
.. root directory of the project. Ensure again that you are in the correct
.. environment Then simply execute ``pytest`` without arguments::

..     /your/local/path/to/the/project via 🅒 hmc-tom
..     ➜ pytest
..     [similar output ...]

.. What are xfails, xpassed, warnings and fails?
.. """""""""""""""""""""""""""""""""""""""""""""

.. Sometimes in testing we know something will go wrong. To deal with expected
.. failures of the code, we use ``pytest``'s xfail mechanism. We decorate a
.. function with::

..     @_pytest.mark.xfail(raises=SomeException)

.. If this test fail with the error ``SomeException``, the test as a whole
.. doesn't fail. The test gets registered as an xfail, and testing merrily
.. continues. If the test succeeds, it gets registered as an xpassed.

.. Warnings are simply notifications that won't fail the test, but still
.. require attention if you use the code for something else. Their output
.. gets displayed at the end of test.

.. A fail indicates that something went wrong in the test. If the code generates
.. at least one failure, ``pytest`` will exit with a non-zero exit code (thus
.. also failing a TravisCI build).

.. .. _to allow for unimplemented classes:

.. Testing for unimplemented classes
.. """""""""""""""""""""""""""""""""

.. When we run our test, we tell ``pytest`` that we don't care about
.. ``NotImplementedErrors`` by using the xfail mechanism::

..     @_pytest.mark.xfail(raises=NotImplementedError)

.. However, one might reach a point where we'd like all our included code to work.
.. We can ask pytest to force failures on ``NotImplementedError``'s. This is
.. simply done by appending the argument ``--runxfail`` to ``pytest``. Testing
.. on TravisCI is done without this option.

.. Running tests from a script
.. ***************************

.. But what if we want to test this method outside ``pytest``? What we have to do
.. is simply:

.. 1. import the package (to access components like e.g. the mass matrices);
.. 2. import the test module from the package;
.. 3. execute the test with appropriate arguments.

.. Here is an example run from an ``ipython`` shell:

.. .. ipython::

..     In [1]: import hmc_tomography

..     In [2]: import hmc_tomography.Tests as tests

..     In [3]: mmclass = hmc_tomography.MassMatrices.Unit

..     In [4]: dimensions = 100

..     In [5]: tests.test_mass_matrices.test_generate(mmclass, dimensions)

.. Finding tests
.. *************

.. Without looking at the ``hmc_tomography.Tests`` reference, we typically don't
.. know what tests are available. However, one is able to discover those in an
.. interactive way:

.. Finding test modules ...

.. .. ipython::

..     In [5]: dir(tests)

.. Finding tests in modules ...

.. .. ipython::

..     In [6]: dir(tests.test_mass_matrices)

.. Getting test descriptions ...

.. .. ipython::

..     In [7]: tests.test_mass_matrices.test_generate?
