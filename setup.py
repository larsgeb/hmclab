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
    description="A numerical laboratory for Bayesian Seismology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/larsgeb/hmclab",
    project_urls={
        "Bug Tracker": "https://github.com/larsgeb/hmclab/issues",
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        # "obspy",
        "numpy",
        "scipy",
        "termcolor",
        "matplotlib",
        "tqdm",
        "h5py",
        "pyyaml",
        "ipywidgets",
        "multiprocess",
        "tilemapbase",
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
            "pytest-cov",
            "pytest-harvest",
            "pytest-ordering",
            "pytest_notebook",
            "autoclasstoc",
            "ipywidgets",
            "nbformat",
            "nbconvert < 6.0.0",
        ]
    },
)
