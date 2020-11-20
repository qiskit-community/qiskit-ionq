# -*- coding: utf-8 -*-
# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Copyright 2020 IonQ, Inc. (www.ionq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""setup module for packaging and distribution"""
import pathlib

from setuptools import find_namespace_packages, setup

__version__ = None
here = pathlib.Path(".").absolute()
version_path = here / "qiskit" / "providers" / "ionq" / "version.py"
requirements_path = here / "requirements.txt"

with open(str(here / "README.md"), "r") as _fp:
    long_description = _fp.read()

with open(str(requirements_path)) as _fp:
    REQUIREMENTS = _fp.readlines()

with open(str(version_path)) as _fp:
    exec(_fp.read())


setup(
    name="qiskit-ionq-provider",
    version=__version__,
    author="IonQ",
    author_email="info@ionq.com",
    description="Qiskit provider for interacting with IonQ backends",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qiskit-community/qiskit-ionq-provider",
    license="Apache 2.0",
    packages=find_namespace_packages(exclude=["test"]),
    keywords="qiskit quantum ionq",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    install_requires=REQUIREMENTS,
    python_requires=">=3.6",
    include_package_data=True,
    zip_safe=False,
)
