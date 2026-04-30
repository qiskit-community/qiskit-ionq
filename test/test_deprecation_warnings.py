# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Copyright 2026 IonQ, Inc. (www.ionq.com)
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

"""Assert the 1.1.0 deprecation warnings fire for symbols that are removed in 2.0."""

import warnings
from unittest.mock import MagicMock

import pytest

from qiskit_ionq import IonQProvider, Session


def test_session_init_warns():
    backend = MagicMock()
    backend.name = "ionq_qpu.aria-1"
    backend.client = MagicMock()
    backend.client.post.return_value = {"id": "sess-1"}
    backend.client.make_path.side_effect = lambda *p: "/".join(p)
    with pytest.warns(DeprecationWarning, match="qiskit_ionq.Session is deprecated"):
        Session(backend=backend, max_jobs=1)


def test_provider_custom_headers_warns():
    with pytest.warns(DeprecationWarning, match="custom_headers=.*is deprecated"):
        IonQProvider(token="t", custom_headers={"X-Foo": "bar"})


def test_provider_no_custom_headers_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        IonQProvider(token="t")


@pytest.mark.parametrize(
    "kwarg",
    ["extra_query_params", "extra_metadata", "sampler_seed"],
)
def test_backend_run_deprecated_kwargs_warn(kwarg, requests_mock):
    """Each removed run() kwarg fires its own deprecation."""
    provider = IonQProvider(token="t", url="https://api.example.invalid/v0.4")
    backend = provider.get_backend("ionq_simulator")
    requests_mock.post(
        "https://api.example.invalid/v0.4/jobs",
        json={"id": "job-1", "status": "submitted"},
        status_code=200,
    )
    requests_mock.get(
        "https://api.example.invalid/v0.4/jobs/job-1",
        json={"id": "job-1", "status": "submitted"},
        status_code=200,
    )

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    payload = {kwarg: 42 if kwarg == "sampler_seed" else {"k": "v"}}
    with pytest.warns(
        DeprecationWarning, match=f"backend.run\\({kwarg}=\\.\\.\\.\\) is deprecated"
    ):
        backend.run(qc, shots=1, **payload)


def test_backend_calibration_warns(requests_mock):
    provider = IonQProvider(token="t", url="https://api.example.invalid/v0.4")
    backend = provider.get_backend("ionq_qpu.aria-1")
    requests_mock.get(
        "https://api.example.invalid/v0.4/backends/qpu.aria-1/characterizations",
        json={
            "characterizations": [
                {
                    "id": "c-1",
                    "backend": "qpu.aria-1",
                    "status": "available",
                    "date": "2026-01-01T00:00:00Z",
                    "qubits": 25,
                }
            ]
        },
        status_code=200,
    )
    with pytest.warns(
        DeprecationWarning, match="IonQBackend.calibration\\(\\) is deprecated"
    ):
        backend.calibration()


def test_backend_status_warns(requests_mock):
    provider = IonQProvider(token="t", url="https://api.example.invalid/v0.4")
    backend = provider.get_backend("ionq_qpu.aria-1")
    requests_mock.get(
        "https://api.example.invalid/v0.4/backends/qpu.aria-1/characterizations",
        json={
            "characterizations": [
                {
                    "id": "c-1",
                    "backend": "qpu.aria-1",
                    "status": "available",
                    "date": "2026-01-01T00:00:00Z",
                    "qubits": 25,
                }
            ]
        },
        status_code=200,
    )
    with pytest.warns(
        DeprecationWarning, match="IonQBackend.status\\(\\) is deprecated"
    ):
        backend.status()
