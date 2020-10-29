from pytest_notebook.nb_regression import (
    NBRegressionFixture,
    NBRegressionError,
    NBRegressionResult,
)

fixture = NBRegressionFixture(
    exec_timeout=10,
    diff_ignore=("/cells/*/outputs/", "/cells/*/execution_count", "/metadata/widgets"),
)
fixture.diff_color_words = False


def test_notebook_01():
    result = fixture.check("examples/notebooks/0.1 - Getting started.ipynb")
    print(result)
    assert False

