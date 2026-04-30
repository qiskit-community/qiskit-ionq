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

"""Build Qiskit Target from IonQ backend and characterization data."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from ionq_core.api.characterizations import get_characterization
from ionq_core.models.backend import Backend
from ionq_core.types import Unset
from qiskit.circuit import Measure, Parameter
from qiskit.circuit.library import (
    CXGate,
    HGate,
    RXGate,
    RXXGate,
    RYGate,
    RYYGate,
    RZGate,
    RZZGate,
    SdgGate,
    SGate,
    SwapGate,
    SXdgGate,
    SXGate,
    TdgGate,
    TGate,
    XGate,
    YGate,
    ZGate,
)
from qiskit.transpiler import InstructionProperties, Target

from .gates import GPI2Gate, GPIGate, MSGate, ZZGate

if TYPE_CHECKING:
    from ionq_core.client import AuthenticatedClient

_t, _p0, _p1, _a = (
    Parameter("theta"),
    Parameter("phi0"),
    Parameter("phi1"),
    Parameter("angle"),
)

_GATESETS = {
    "qis": (
        [
            HGate(),
            XGate(),
            YGate(),
            ZGate(),
            SGate(),
            SdgGate(),
            TGate(),
            TdgGate(),
            SXGate(),
            SXdgGate(),
            RXGate(_t),
            RYGate(_t),
            RZGate(_t),
        ],
        [CXGate(), SwapGate(), RXXGate(_t), RYYGate(_t), RZZGate(_t)],
    ),
    "native": (
        [GPIGate(_t), GPI2Gate(_t)],
        [MSGate(_p0, _p1, _a), ZZGate(_t)],
    ),
}


def _fetch_characterization(backend_info: Backend, client: AuthenticatedClient):
    char_id = backend_info.characterization_id
    if isinstance(char_id, Unset) or char_id is None:
        return None
    return get_characterization.sync(
        backend=backend_info.backend, uuid=UUID(char_id), client=client
    )  # ty: ignore[invalid-argument-type]


def _get_error(char, key: str) -> float | None:
    if char is None or isinstance(char.fidelity, Unset) or char.fidelity is None:
        return None
    if key == "spam":
        fidelity = char.fidelity.spam.median
    else:
        entry = char.fidelity.additional_properties.get(key)
        fidelity = entry.get("mean") if isinstance(entry, dict) else None
    return (1 - fidelity) if fidelity is not None else None


def build_target(
    backend_info: Backend, client: AuthenticatedClient, gateset: str = "qis"
) -> Target:
    nq = backend_info.qubits
    gates_1q, gates_2q = _GATESETS[gateset]
    target = Target(num_qubits=nq)

    if backend_info.backend == "simulator":
        for gate in [*gates_1q, *gates_2q, Measure()]:
            target.add_instruction(gate, {None: None})
    else:
        char = _fetch_characterization(backend_info, client)
        error_1q, error_2q = _get_error(char, "1q"), _get_error(char, "2q")
        props_1q = {(q,): InstructionProperties(error=error_1q) for q in range(nq)}
        for gate in gates_1q:
            target.add_instruction(gate, props_1q)
        props_2q = {
            (i, j): InstructionProperties(error=error_2q)
            for i in range(nq)
            for j in range(nq)
            if i != j
        }
        for gate in gates_2q:
            target.add_instruction(gate, props_2q)
        target.add_instruction(
            Measure(),
            {
                (q,): InstructionProperties(error=_get_error(char, "spam"))
                for q in range(nq)
            },
        )

    return target
