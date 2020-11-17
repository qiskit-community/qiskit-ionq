"""setup module for packaging and distribution"""
import os
import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(".").absolute()
version_path = here / "qiskit" / "providers" / "ionq" / "version.py"

with open(str(here / "README.md"), "r") as f:
    long_description = f.read()

__version__ = None
with open(str(version_path)) as f:
    exec(f.read())

REQUIREMENTS = ["qiskit-terra>=0.10", "requests>=2.24.0"]

setup(
    name="qiskit-ionq-provider",
    version=__version__,
    author="IonQ",
    author_email="info@ionq.com",
    description="Qiskit provider for interacting with IonQ backends",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ionq/qiskit-ionq-provider",
    license="Apache 2.0",
    packages=find_packages(exclude=["test"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=REQUIREMENTS,
    python_requires=">=3.7",
)
