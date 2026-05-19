# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
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
"""Test basic provider API methods."""

import pytest

from qiskit_ionq import IonQProvider


def test_provider_autocomplete():
    """Verifies that provider.backends autocomplete works."""
    pro = IonQProvider("123456")

    for backend in pro.backends():
        assert hasattr(pro.backends, backend.name)


def test_provider_getbackend():
    """Verifies that provider.get_backend works."""
    pro = IonQProvider("123456")

    for backend in pro.backends():
        qis = pro.get_backend(backend.name)
        native = pro.get_backend(backend.name, gateset="native")
        assert backend == qis
        assert backend != native


def test_backend_eq():
    """Verifies equality works for various backends"""
    pro = IonQProvider("123456")

    sub1 = pro.get_backend("ionq_qpu.sub-1")
    sub2 = pro.get_backend("ionq_qpu.sub-2")
    also_sub1 = pro.get_backend("ionq_qpu.sub-1")
    simulator = pro.get_backend("ionq_simulator")

    assert sub1 == also_sub1
    assert sub1 != sub2
    assert also_sub1 != sub2
    assert sub1 != simulator


# ---------------------------------------------------------------------------
# Native-gateset two-qubit instruction selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected_2q",
    [
        # ZZ-class
        ("ionq_qpu.forte-1", "zz"),
        ("ionq_qpu.forte-enterprise-1", "zz"),
        ("ionq_qpu.tempo-1", "zz"),
        # MS-class
        ("ionq_qpu.aria-1", "ms"),
        ("ionq_qpu.aria-2", "ms"),
        # Generic / unrecognised falls back to MS, matching pre-existing behaviour
        ("ionq_qpu.unknown-1", "ms"),
    ],
)
def test_native_2q_gate(name, expected_2q):
    """The native-gateset Target should expose ZZ on Forte/Tempo and MS on Aria."""
    pro = IonQProvider("123456")
    backend = pro.get_backend(name, gateset="native")
    instr_names = set(backend.target.operation_names)
    assert expected_2q in instr_names
    # And the *other* 2q native gate must NOT be present, to keep the Target unambiguous.
    other = "ms" if expected_2q == "zz" else "zz"
    assert other not in instr_names
