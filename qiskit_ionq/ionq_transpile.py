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
"""
The IonQ transpile function to make using the native gate optimizer easier.
"""

from __future__ import annotations
from typing import Sequence, Union

from qiskit import QuantumCircuit
from .ionq_optimizer_plugin import TrappedIonOptimizerPlugin
from .ionq_backend import IonQBackend


def _unpad_to_inputs(circ: QuantumCircuit) -> QuantumCircuit:
    """
    Remove ancilla padding based on the circuit's finalized layout, if present.
    Falls back to returning the circuit unchanged if no layout info exists.
    """
    layout = getattr(circ, "layout", None)
    if layout is None or not hasattr(layout, "final_index_layout"):
        return circ

    pos = layout.final_index_layout(filter_ancillas=True)
    if pos is None:
        return circ

    qmap = {circ.qubits[phys]: i for i, phys in enumerate(pos)}
    cmap = {c: i for i, c in enumerate(circ.clbits)}
    out = QuantumCircuit(len(pos), circ.num_clbits)
    for inst in circ.data:
        if any(q not in qmap for q in inst.qubits):
            # Skip ops touching ancillas that we’re dropping.
            continue
        out.append(
            inst.operation,
            [out.qubits[qmap[q]] for q in inst.qubits],
            [out.clbits[cmap[c]] for c in inst.clbits],
        )
    return out


def ionq_transpile(
    circuits: Union[QuantumCircuit, Sequence[QuantumCircuit]],
    backend: IonQBackend | None = None,
    *,
    optimization_level: int = 3,
    drop_idle_qubits: bool = True,
):
    """
    Transpile circuit(s) for IonQ backends using Qiskit's preset pipeline
    followed by IonQ-native rewrite passes from TrappedIonOptimizerPlugin.

    Args:
        circuits: A circuit or list of circuits.
        backend: IonQ backend (simulator, aria, forte) or any Qiskit backend.
        optimization_level: 0-3; forwarded into the preset + IonQ pipelines.
        drop_idle_qubits: If True, remove ancillas based on the final layout.

    Returns:
        Transpiled circuit (or list of circuits).
    """

    # Minimal shim that provides .backend (and .target if available) to the plugin.
    class _PMConfig:
        def __init__(self, backend):
            self.backend = backend
            self.target = getattr(backend, "target", None)

    single = isinstance(circuits, QuantumCircuit)
    circ_list = [circuits] if single else list(circuits)

    # Build the combined pass manager (preset pipeline + IonQ-native passes).
    ionq_pm = TrappedIonOptimizerPlugin().pass_manager(
        _PMConfig(backend), optimization_level=optimization_level
    )

    # Run once through the full pipeline.
    out_list = ionq_pm.run(circ_list)

    # Optionally drop ancillas introduced by layout/translation stages.
    if drop_idle_qubits:
        out_list = [_unpad_to_inputs(c) for c in out_list]

    return out_list[0] if single else out_list
