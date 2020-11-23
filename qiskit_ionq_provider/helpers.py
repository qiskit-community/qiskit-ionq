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

"""
Helper methods for mapping a qobj (:mod:`Qiskit Quantum Objects <qiskit.qobj>`)
to IonQ REST API compatible values.
"""

import json

from qiskit import qobj as qqobj

from . import exceptions


def build_output_map(qobj: qqobj.QasmQobj):
    """
    IonQ's API does not allow ad-hoc remapping of classical to quantum
    registers, instead always returning quantum[i] as classical[i] in the
    return bitstring.

    The output map is created from the measure instructions in the program so
    that arbitrary remapping may be done later, based on desired output mapping.

    Args:
        qobj (:class:`QasmQobj <qiskit.qobj.QasmQobj>`): A qiskit quantum job.

    Raises:
        IonQGateError: If a measurement in the experiment was found
            before the end of the last non-measurement instruction.
        ValueError: If the instructions contained zero measurements.

    Returns:
        dict: an output map dict to be used by the caller.
    """
    output_map = {}
    measurements = 0
    for instruction in qobj.experiments[0].instructions:
        if instruction.name == "measure":
            output_map[instruction.qubits[0]] = instruction.memory[0]
            measurements += 1
        else:
            if measurements > 0:
                raise exceptions.IonQGateError(
                    "Measurements must occur at the end of the circuit."
                )

    if measurements == 0:
        raise ValueError("Circuit must have at least one measurement")

    return output_map


def qobj_to_ionq(qobj: qqobj.QasmQobj):
    """Convert a job to a JSON object compatible with the IonQ REST API.

    Args:
        qobj (:class:`QasmQobj <qiskit.qobj.QasmQobj>`): A qiskit quantum job.

    Raises:
        IonQJobError: If ``qobj`` has more than one experiment.

    Returns:
        dict: A dict with IonQ API compatible values.
    """
    if len(qobj.experiments) > 1:
        raise exceptions.IonQJobError(
            "IonQ backends do not support multi-experiment jobs."
        )

    ionq_json = {
        "lang": "json",
        "target": qobj.header.backend_name[5:],
        "shots": qobj.config.shots,
        "body": {
            "qubits": qobj.experiments[0].config.n_qubits,
            "circuit": build_circuit(qobj.experiments[0].instructions),
        },
        # store a couple of things we'll need later for result formatting
        "metadata": {
            "shots": str(qobj.config.shots),
            "qobj_id": str(qobj.qobj_id),
            "output_length": str(qobj.experiments[0].header.memory_slots),
            "output_map": json.dumps(build_output_map(qobj)),
            "header": json.dumps(qobj.experiments[0].header.to_dict()),
        },
    }
    return json.dumps(ionq_json)


def build_circuit(instructions):
    """Build a circuit in IonQ's instruction format from qiskit instructions.

    .. ATTENTION:: This function ignores the following compiler directives:

       * ``barrier``

    .. ATTENTION::
       The following instructions are currently **unsupported**:

       * ``reset``
       * ``u1``
       * ``u2``
       * ``u3``
       * ``cu1``
       * ``cu2``
       * ``cu3``

    Args:
        instructions (list[:class:`QasmQobjInstruction <qiskit.qobj.QasmQobjInstruction>`]):
            A list of quantum circuit instructions.

    Raises:
        IonQGateError: If an unsupported instruction is supplied.

    Returns:
        list[dict]: A list of instructions in a converted dict format.
    """
    compiler_directives = ["barrier"]
    invalid_instructions = [
        "reset",
        "u1",
        "u2",
        "u3",
        "cu1",
        "cu2",
        "cu3",
    ]
    circuit = []
    for instruction in instructions:
        # Don't process compiler directives.
        if instruction.name in compiler_directives:
            continue

        # Don't process measurement instructions.
        if instruction.name == "measure":
            continue

        # Raise out for instructions we don't support.
        if instruction.name in invalid_instructions:
            raise exceptions.IonQGateError(instruction.name)

        # Process the instruction and convert.
        rotation = {}
        if hasattr(instruction, "params"):
            rotation = {"rotation": instruction.params[0]}

        # Default conversion is simple.
        converted = {
            "gate": instruction.name,
            "target": instruction.qubits[0],
            **rotation,
        }

        # If this is a `c` instruction, do some extra work.
        if instruction.name[0] == "c":
            is_double_control = instruction.name[1] == "c"
            gate = instruction.name[1:]
            controls = [instruction.qubits[0]]
            target = instruction.qubits[1]
            if is_double_control:
                gate = instruction.name[2:]
                controls = [instruction.qubits[0], instruction.qubits[1]]
                target = instruction.qubits[2]
            converted = {
                "gate": gate,
                "controls": controls,
                "target": target,
                **rotation,
            }

        # Finally, add the converted instruction to our circuit.
        circuit.append(converted)

    return circuit


__all__ = ["build_circuit", "build_output_map", "qobj_to_ionq"]
