import setuptools
import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    name="hmc-tomography",
    author="Lars Gebraad, Andreas Fichtner, Andrea Zunino",
    author_email="lars.gebraad@erdw.ethz.ch",
    description="An example HMC tomography package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/larsgeb/hmc-tomography",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy",
        "termcolor",
        "matplotlib",
        "tqdm",
        "h5py",
        "pyyaml",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "ipython",
            "sphinx",
            "nbsphinx",
            "sphinx_rtd_theme",
            "numpydoc",
            "pandoc",
            "sphinx-git",
            "versioneer",
            "pandas",
            "pre-commit",
            "codecov",
            "pytest",
            "pytest-harvest",
            "pytest-ordering",
            "pytest_notebook",
        ],
        "testing": [
            "black",
            "pandas",
            "pre-commit",
            "codecov",
            "pytest",
            "pytest-harvest",
            "pytest-ordering",
            "pytest_notebook",
        ],
    },
    entry_points={"console_scripts": ["hmc_tomography=hmc_tomography.__main__:cli"]},
)
