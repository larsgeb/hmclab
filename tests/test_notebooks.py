import glob
import nbformat
from black import format_str, FileMode
import pytest
from pytest_notebook.nb_regression import NBRegressionFixture
import warnings

# Setup the fixture for testing notebooks using pytest_notebook
fixture = NBRegressionFixture(
    exec_timeout=600,
    diff_ignore=("/cells/*/outputs/", "/cells/*/execution_count", "/metadata"),
)
fixture.diff_color_words = False

# Find all notebook files
notebooks = glob.glob("notebooks/tutorials/*.ipynb")

@pytest.mark.parametrize("notebook_fh", notebooks)
def test_notebook(notebook_fh):

    # Clean notebook
    notebook = nbformat.read(notebook_fh, as_version=nbformat.NO_CONVERT)

    notebook.metadata = {}
    notebook.cells = [cell for cell in notebook.cells if len(cell["source"]) > 0]

    # Clear cells
    for cell in notebook.cells:
        cell["metadata"] = {}
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

            # Format the cell using black, removing trailing newlines
            cell.source = format_str(
                cell.source.replace("%matplotlib notebook", ""), mode=FileMode()
            ).rstrip()

    # Write to file
    nbformat.write(notebook, notebook_fh)

    # Test notebooks
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        result = fixture.check(notebook_fh)

    if not(notebook_fh == "notebooks/tutorials/1 - Tuning Hamiltonian Monte Carlo.ipynb"):
        # Write out final version to original file if all tests succeeded
        nbformat.write(nb=result.nb_final, fp=notebook_fh)
