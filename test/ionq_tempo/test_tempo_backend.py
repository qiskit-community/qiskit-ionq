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

"""Backend-side tests for Tempo: name resolution, native target, verbatim gate set."""

import pytest
from qiskit import QuantumCircuit

from qiskit_ionq import GPI2Gate, GPIGate, MSGate, ZZGate
from qiskit_ionq.exceptions import IonQGateError
from qiskit_ionq.helpers import qiskit_to_ionq


# ---------------------------------------------------------------------------
# Provider name resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["ionq_qpu.tempo-1", "qpu.tempo-1", "ionq_qpu.tempo-2"]
)
def test_tempo_backend_resolves(provider, name):
    """Caller can address tempo-N by either the local or API name form."""
    backend = provider.get_backend(name, gateset="native")
    assert "tempo" in backend.name.lower()
    # API form drops the local `ionq_qpu` prefix.
    assert backend._api_backend_name.startswith("qpu.tempo")  # pylint: disable=protected-access


def test_native_target_uses_zz(provider):
    """The native target for a tempo backend exposes ZZ rather than MS."""
    backend = provider.get_backend("ionq_qpu.tempo-1", gateset="native")
    ops = set(backend.target.operation_names)
    assert "zz" in ops
    assert "ms" not in ops


def test_tempo_qis_target_unchanged(provider):
    """The QIS-gateset target is the same superset on tempo as on aria/forte."""
    tempo = provider.get_backend("ionq_qpu.tempo-1", gateset="qis")
    forte = provider.get_backend("ionq_qpu.forte-1", gateset="qis")
    assert set(tempo.target.operation_names) == set(forte.target.operation_names)


# ---------------------------------------------------------------------------
# Verbatim gate-set validation
# ---------------------------------------------------------------------------


def _native_qc():
    """Native-gate circuit valid for verbatim submission."""
    qc = QuantumCircuit(2, 2)
    qc.append(GPI2Gate(0.5), [0])
    qc.append(GPIGate(0.25), [1])
    qc.append(ZZGate(0.25), [0, 1])
    qc.measure([0, 1], [0, 1])
    return qc


def test_verbatim_accepts_zz_native(provider):
    """A circuit using gpi/gpi2/zz/measure passes verbatim validation."""
    backend = provider.get_backend("ionq_qpu.tempo-1", gateset="native")
    # Should not raise.
    qiskit_to_ionq(
        _native_qc(), backend, passed_args={"shots": 100, "verbatim": True}
    )


def test_verbatim_accepts_ms_native(provider):
    """MS is in the verbatim gate set too (Aria-class native), and is accepted."""
    backend = provider.get_backend("ionq_qpu.tempo-1", gateset="native")
    qc = QuantumCircuit(2, 2)
    qc.append(GPI2Gate(0.0), [0])
    qc.append(MSGate(0.0, 0.0, 0.25), [0, 1])
    qc.measure([0, 1], [0, 1])
    qiskit_to_ionq(qc, backend, passed_args={"shots": 100, "verbatim": True})


def test_verbatim_rejects_h_gate(provider):
    """Common QIS gates (h, cx, rx, etc.) cannot ride through verbatim."""
    backend = provider.get_backend("ionq_qpu.tempo-1", gateset="native")
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    with pytest.raises(IonQGateError) as exc:
        qiskit_to_ionq(qc, backend, passed_args={"shots": 100, "verbatim": True})
    assert exc.value.gate_name == "h"


def test_verbatim_rejects_cx_gate(provider):
    """CX is not native on IonQ — verbatim must reject it before the wire."""
    backend = provider.get_backend("ionq_qpu.tempo-1", gateset="native")
    qc = QuantumCircuit(2, 2)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    with pytest.raises(IonQGateError) as exc:
        qiskit_to_ionq(qc, backend, passed_args={"shots": 100, "verbatim": True})
    assert exc.value.gate_name == "cx"


def test_verbatim_rejects_arrange(provider):
    """ARRANGE is compiler-only; users cannot submit it (per the v2 RFC).

    A custom-named operation surfaces here as ``IonQGateError`` because it is
    not in the verbatim allow-list. This is the client-side guard; the server
    would also reject it but failing fast saves a round-trip + queue wait.
    """
    backend = provider.get_backend("ionq_qpu.tempo-1", gateset="native")
    qc = QuantumCircuit(2, 2)
    qc.barrier()  # placeholder; barrier is allowed
    # Build a fake "arrange" instruction with an unrecognised name.
    qc.unitary(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        [0, 1],
        label="arrange",
    )
    qc.measure([0, 1], [0, 1])
    with pytest.raises(IonQGateError):
        qiskit_to_ionq(qc, backend, passed_args={"shots": 100, "verbatim": True})


def test_no_verbatim_no_validation(provider):
    """Without verbatim=True, QIS gates ride through unmolested for compiler synthesis."""
    backend = provider.get_backend("ionq_qpu.tempo-1", gateset="qis")
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    # Should not raise even though h() is not in the verbatim allow-list.
    qiskit_to_ionq(qc, backend, passed_args={"shots": 100})
