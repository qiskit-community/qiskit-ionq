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

import warnings

import pytest

from qiskit_ionq import IonQProvider

# Real backend_config captured before the autouse fixture stubs it.
_REAL_BACKEND_CONFIG = IonQProvider.backend_config

_CATALOG = {
    "qpu.aria-1": {"qubits": 25, "supported_native_gates": ["gpi", "gpi2", "ms"]},
    "simulator": {"qubits": 29},
}


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


def test_backend_config_lookup():
    """backend_config resolves local names against the cached catalog, silently."""
    pro = IonQProvider("123456")
    pro._catalog = _CATALOG
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _REAL_BACKEND_CONFIG(pro, "ionq_qpu.aria-1") == _CATALOG["qpu.aria-1"]
        assert _REAL_BACKEND_CONFIG(pro, "ionq_simulator") == _CATALOG["simulator"]


def test_backend_config_unknown_warns():
    """An id missing from a populated catalog warns and returns {}."""
    pro = IonQProvider("123456")
    pro._catalog = _CATALOG
    with pytest.warns(UserWarning, match="not in the IonQ catalog"):
        assert _REAL_BACKEND_CONFIG(pro, "ionq_qpu.nope-1") == {}


@pytest.mark.parametrize(
    "name, catalog",
    [
        ("ionq_qpu", _CATALOG),  # generic template: no real device behind it
        ("ionq_qpu.aria-1", {}),  # offline: the failed fetch already warned
    ],
)
def test_backend_config_silent_defaults(name, catalog):
    """The generic qpu template and an empty (offline) catalog resolve to {}
    without re-warning."""
    pro = IonQProvider("123456")
    pro._catalog = catalog
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _REAL_BACKEND_CONFIG(pro, name) == {}
