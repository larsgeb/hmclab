# content of conftest.py
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--plot", action="store_true", default=False, help="run plot tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "plot: mark test as plot to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--plot"):
        # --plot given in cli: do not skip plot tests
        return
    skip_plot = pytest.mark.skip(reason="need --plot option to run")
    for item in items:
        if "plot" in item.keywords:
            item.add_marker(skip_plot)
