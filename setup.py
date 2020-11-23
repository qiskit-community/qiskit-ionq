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
from qiskit_ionq_provider.version import get_version_info

here = os.path.dirname(os.path.realpath(__file__))
requirements_path = here + "/requirements.txt"

with open(str(here+"/README.md"), "r") as _fp:
    long_description = _fp.read()

with open(str(requirements_path)) as _fp:
    REQUIREMENTS = _fp.readlines()

REQUIREMENTS = ["qiskit-terra>=0.10", "requests>=2.24.0"]

setup(
    name="qiskit-ionq-provider",
    version = get_version_info(),
    author="IonQ",
    author_email="info@ionq.com",
    packages=find_packages(exclude=["test"]),
    description="Qiskit provider for IonQ backends",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/qiskit-community/qiskit-ionq-provider",
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit sdk quantum",
    install_requires=REQUIREMENTS,
    include_package_data=True,
    python_requires=">=3.7",
    project_urls={
        "Bug Tracker": "https://github.com/qiskit-community/qiskit-ionq-provider/issues",
        "Source Code": "https://github.com/qiskit-community/qiskit-ionq-provider",
    },
    zip_safe=False
)
