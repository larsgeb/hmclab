"""Python setup.py for project_name package"""
import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="hmclab",
    version=read("hmclab", "VERSION"),
    description="project_description",
    url="https://github.com/larsgeb/hmclab/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Lars Gebraad",
    author_email="larsgebraad@gmail.com",
    packages=find_packages(exclude=["dev", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["hmclab = hmclab.__main__:main"]
    },
    extras_require={"dev": read_requirements("requirements-dev.txt")},
    python_requires=">=3.11",
)
