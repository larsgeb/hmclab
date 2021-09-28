import setuptools
import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    name="hmclab",
    author="Lars Gebraad, Andrea Zunino, Andreas Fichtner",
    author_email="lars.gebraad@erdw.ethz.ch",
    description="An example HMC tomography package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/larsgeb/hmclab",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "scipy",
        "termcolor",
        "matplotlib",
        "tqdm",
        "h5py",
        "pyyaml",
        "jax",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "sphinx",
            "nbsphinx",
            "sphinx_rtd_theme",
            "numpydoc",
            "pandoc",
            "sphinx-git",
            "sphinxcontrib-bibtex",
            "versioneer",
            "pandas",
            "pre-commit",
            "codecov",
            "pytest",
            "pytest-harvest",
            "pytest-ordering",
            "pytest_notebook",
            "autoclasstoc",
            "ipywidgets",
            "nbformat",
            "nbconvert < 6.0.0",
            "pybind11",
            "cmake",
            "psvWave",
        ],
    },
    entry_points={"console_scripts": ["hmclab=hmclab.__main__:cli"]},
)
