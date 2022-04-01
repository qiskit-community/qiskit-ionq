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

import os

from setuptools import find_packages, setup

here = os.path.dirname(os.path.realpath(__file__))
readme_path = os.path.join(here, "README.md")
requirements_path = os.path.join(here, "requirements.txt")
test_requirements_path = os.path.join(here, "requirements-test.txt")
version_path = os.path.join(here, "qiskit_ionq", "version.py")

with open(readme_path, "r") as _fp:
    long_description = _fp.read()

with open(requirements_path) as _fp:
    REQUIREMENTS = _fp.readlines()

with open(test_requirements_path) as _fp:
    TEST_REQUIREMENTS = _fp.readlines()

# This is needed to prevent importing any package specific dependencies at
#   stages of the setup.py life-cycle where they may not yet be installed.
__version__ = None
with open(version_path) as _fp:
    exec(_fp.read())  # pylint: disable=exec-used

setup(
    name="qiskit-ionq",
    version=__version__,
    author="IonQ",
    author_email="info@ionq.com",
    packages=find_packages(exclude=["test*"]),
    description="Qiskit provider for IonQ backends",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qiskit-partners/qiskit-ionq",
    license="Apache 2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit sdk quantum",
    python_requires=">=3.7",
    setup_requires=["pytest-runner"],
    install_requires=REQUIREMENTS,
    tests_require=TEST_REQUIREMENTS,
    zip_safe=False,
    include_package_data=True,
    project_urls={
        "Bug Tracker": "https://github.com/qiskit-partners/qiskit-ionq/issues",
        "Source Code": "https://github.com/qiskit-partners/qiskit-ionq",
    },
)
