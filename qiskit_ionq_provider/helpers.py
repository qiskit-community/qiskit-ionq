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

from . import exceptions

__all__ = ["qiskit_to_ionq", "qiskit_circ_to_ionq_circ"]


def qiskit_circ_to_ionq_circ(circ):
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

    Parameters:
        circ (:class:`QuantumCircuit <qiskit.QuantumCircuit>`): A quantum circuit.

    Raises:
        IonQGateError: If an unsupported instruction is supplied.

    Returns:
        list[dict]: A list of instructions in a converted dict format.
        int: The number of measurements.
        dict: The measurement map from qubit number to classical bit number.
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
    num_meas = 0
    meas_map = {}
    for instruction in circ.data:
        # Don't process compiler directives.
        if instruction[0].name in compiler_directives:
            continue

        # Don't process measurement instructions.
        if instruction[0].name == "measure":
            meas_map[int(instruction[1][0].index)] = instruction[2][0].index
            num_meas += 1
            continue

        # Raise out for instructions we don't support.
        if instruction[0].name in invalid_instructions:
            raise exceptions.IonQGateError(instruction.name)

        # Process the instruction and convert.
        rotation = {}
        if any(instruction[0].params):
            rotation = {"rotation": instruction[0].params}

        # Default conversion is simple.
        converted = {
            "gate": instruction[0].name,
            "target": instruction[1][0].index,
            **rotation,
        }

        # If this is a `c` instruction, do some extra work.
        if instruction[0].name[0] == "c":
            is_double_control = instruction[0].name[1] == "c"
            gate = instruction[0].name[1:]
            controls = [instruction[1][0].index]
            target = instruction[1][1].index
            if is_double_control:
                gate = instruction[0].name[2:]
                controls = [instruction[1][0].index, instruction[1][1].index]
                target = instruction[1][2].index
            converted = {
                "gate": gate,
                "controls": controls,
                "target": target,
                **rotation,
            }

        # Finally, add the converted instruction to our circuit.
        circuit.append(converted)

    return circuit, num_meas, meas_map

def qiskit_to_ionq(circuit, backend_name, shots=1024):
    """Convert a job to a JSON object compatible with the IonQ REST API.

    Parameters:
        circuit (:class:`QuantumCircuit <qiskit.QuantumCircuit>`): A qiskit quantum circuit.
        shots (int, optional): Number of shots to take. Defaults to 1024.

    Returns:
        dict: A dict with IonQ API compatible values.
    """     
    ionq_circ, num_meas, meas_map = qiskit_circ_to_ionq_circ(circuit)

    ionq_json = {
        "lang": "json",
        "target": backend_name[5:],
        "shots": shots,
        "body": {
            "qubits": circuit.num_qubits,
            "circuit": ionq_circ,
        },
        # store a couple of things we'll need later for result formatting
        "metadata": {
            "shots": str(shots),
            "qobj_id": '123456',
            "output_length": str(num_meas),
            "output_map": json.dumps(meas_map),
            "header": "{}",
        },
    }
    return json.dumps(ionq_json)
