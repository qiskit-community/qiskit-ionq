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

from qiskit_ionq import IonQProvider, ionq_provider

# Real get_backend_config captured before the autouse fixture stubs it.
_REAL_GET_BACKEND_CONFIG = IonQProvider.get_backend_config

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
    """get_backend_config resolves local names against the cached catalog, silently."""
    pro = IonQProvider("123456")
    pro._catalog = _CATALOG
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (
            _REAL_GET_BACKEND_CONFIG(pro, "ionq_qpu.aria-1") == _CATALOG["qpu.aria-1"]
        )
        assert _REAL_GET_BACKEND_CONFIG(pro, "ionq_simulator") == _CATALOG["simulator"]


def test_backend_config_unknown_warns():
    """An id missing from a populated catalog warns and returns {}."""
    pro = IonQProvider("123456")
    pro._catalog = _CATALOG
    with pytest.warns(UserWarning, match="not in the IonQ catalog"):
        assert _REAL_GET_BACKEND_CONFIG(pro, "ionq_qpu.nope-1") == {}


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
        assert _REAL_GET_BACKEND_CONFIG(pro, name) == {}


def test_catalog_fetched_on_get_backend_not_init(monkeypatch):
    """The catalog is fetched on first get_backend, not at construction,
    and cached."""
    calls = []

    def fake_get_backends(*_args, **_kwargs):
        calls.append(1)
        return _CATALOG

    monkeypatch.setattr(ionq_provider, "get_backends", fake_get_backends)
    monkeypatch.setattr(
        ionq_provider.IonQProvider, "get_backend_config", _REAL_GET_BACKEND_CONFIG
    )
    pro = IonQProvider("123456")
    assert not calls  # construction is network-free
    pro.get_backend("ionq_qpu.aria-1")
    assert len(calls) == 1
    pro.get_backend("ionq_simulator")
    assert len(calls) == 1  # cached; no refetch


def test_catalog_fetch_failure_warns_and_caches(monkeypatch):
    """A failed fetch warns once and caches an empty catalog."""

    def raise_offline(*_args, **_kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(ionq_provider, "get_backends", raise_offline)
    pro = IonQProvider("123456")
    with pytest.warns(UserWarning, match="Failed to fetch backends catalog"):
        assert _REAL_GET_BACKEND_CONFIG(pro, "ionq_qpu.aria-1") == {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _REAL_GET_BACKEND_CONFIG(pro, "ionq_qpu.aria-1") == {}
