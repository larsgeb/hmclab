import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hmc-tomography-lars-gebraad",
    version="0.0.1",
    author="Lars Gebraad, Andreas Fichtner, Andrea Zunino",
    author_email="lars.gebraad@erdw.ethz.ch",
    description="An example HMC tomography package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/larsgeb/hmc-tomography",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
