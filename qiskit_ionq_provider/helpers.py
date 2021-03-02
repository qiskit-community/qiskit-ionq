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
import gzip
import base64

from qiskit.circuit import controlledgate as q_cgates
from qiskit.circuit.library import standard_gates as q_gates
from qiskit.qobj import QasmQobj
from qiskit.assembler import disassemble

from . import exceptions

ionq_basis_gates = [
    "x",
    "y",
    "z",
    "rx",
    "ry",
    "rz",
    "h",
    "not",
    "cnot",
    "cx",
    "s",
    "si",
    "t",
    "ti",
    "v",
    "vi",
    "xx",
    "yy",
    "zz",
    "swap",
]

# gates that we don't take by name but can serialze correctly to our IR
acceptable_aliases = [
    "cx",
    "cy",
    "cz",
    "crx",
    "crz",
    "cry",
    "ch",
    "ccx",
    "cswap",
]


def qiskit_circ_to_ionq_circ(input_circuit):
    """Build a circuit in IonQ's instruction format from qiskit instructions.

    .. ATTENTION:: This function ignores the following compiler directives:
       * ``barrier``

    Parameters:
        circ (:class:`QuantumCircuit <qiskit.circuit.QuantumCircuit>`): A quantum circuit.

    Raises:
        IonQGateError: If an unsupported instruction is supplied.

    Returns:
        list[dict]: A list of instructions in a converted dict format.
        int: The number of measurements.
        dict: The measurement map from qubit number to classical bit number.
    """
    compiler_directives = ["barrier"]
    output_circuit = []
    num_meas = 0
    meas_map = [None] * len(input_circuit.clbits)
    for instruction, qargs, cargs in input_circuit.data:
        # Don't process compiler directives.
        instruction_name = instruction.name
        if instruction_name in compiler_directives:
            continue

        # Don't process measurement instructions.
        if instruction_name == "measure":
            meas_map[int(cargs[0].index)] = int(qargs[0].index)
            num_meas += 1
            continue

        # Raise out for instructions we don't support.
        if instruction_name not in ionq_basis_gates and instruction_name not in acceptable_aliases:
            raise exceptions.IonQGateError(instruction_name)

        # Process the instruction and convert.
        rotation = {}
        if any(instruction.params):
            # The float is here to cast Qiskit ParameterExpressions to numbers
            rotation = {"rotation": float(instruction.params[0])}

        # Default conversion is simple, just gate & target.
        converted = {"gate": instruction_name, "targets": [qargs[0].index]}

        # Make sure swap uses all qargs.
        if isinstance(instruction, q_gates.SwapGate):
            converted["targets"] = [qargs[0].index, qargs[1].index]
        # If this is a controlled gate, make sure to set control qubits.
        elif isinstance(instruction, q_cgates.ControlledGate):
            gate = instruction_name[1:]
            controls = [qargs[0].index]
            targets = [qargs[1].index]
            # If this is a "double" control, we use two control qubits.
            if gate[0] == "c":
                gate = gate[1:]
                controls = [qargs[0].index, qargs[1].index]
                targets = [qargs[2].index]
            elif gate == "swap":
                # If this is a cswap, we have two targets:
                targets = [qargs[-2].index, qargs[-1].index]

            # Update converted gate values.
            converted.update(
                {
                    "gate": gate,
                    "controls": controls,
                    "targets": targets,
                }
            )

        # if there's a valid instruction after a measurement,
        if num_meas > 0:
            # see if any of the involved qubits have been measured,
            # and raise if so — no mid-circuit measurement!
            controls_and_targets = converted.get("targets", []) + converted.get("controls", [])
            if any(i in meas_map for i in controls_and_targets):
                raise exceptions.IonQMidCircuitMeasurementError(qargs[0].index, instruction_name)

        output_circuit.append({**converted, **rotation})

    return output_circuit, num_meas, meas_map


def get_register_sizes_and_labels(register):
    sizes = []
    labels = []

    for bit in register:
        size = [bit.register.name, bit.register.size]
        label = [bit.register.name, bit.index]

        if size not in sizes:
            sizes.append(size)

        labels.append(label)

    return sizes, labels


# slightly goofy workaround to account for the fact that IonQ's "arbitrary" metadata field
# only takes string KV pairs with value max length 400
# so we try and pack it down into a more-compressed string format
# and raise if it's still too long
# TODO: make this behavior a little nicer (dict metadata) on IonQ side; fix here when we do
def compress_dict_to_metadata_string(metadata_dict):
    """Convert a dict to a compact string format (dumped, gzipped, base64 encoded) for storing in IonQ API metadata

    Parameters:
        metadata_dict: a dict with metadata relevant to building the results object on a returned job.

    Returns:
        str: encoded string
    """
    serialized = json.dumps(metadata_dict)
    compressed = gzip.compress(serialized.encode("utf-8"))
    encoded = base64.b64encode(compressed)
    encoded_string = encoded.decode()
    encoded_string_length = len(encoded_string)
    if encoded_string_length > 400:  # 400 char is an IonQ API limitation
        raise exceptions.IonQMetadataStringError(encoded_string_length)
    return encoded_string


def decompress_metadata_string_to_dict(input_string):
    """Convert compact string format (dumped, gzipped, base64 encoded) from IonQ API metadata back into a dict
    relevant to building the results object on a returned job.

    Parameters:
        input_string: compressed string format of metadata dict

    Returns:
        dict: decompressed metadata dict
    """
    if input_string is None:
        return None
    encoded = input_string.encode()
    decoded = base64.b64decode(encoded)
    decompressed = gzip.decompress(decoded)
    return json.loads(decompressed)


def qiskit_to_ionq(circuit, backend_name, passed_args=None):
    """Convert a Qiskit circuit to a IonQ compatible dict.

    Parameters:
        circuit (:class:`qiskit.circuit.QuantumCircuit`): A Qiskit quantum circuit.
        backend_name (str): Backend name.
        passed_args (dict): Dictionary containing additional passed arguments, eg. shots.

    Returns:
        dict: A dict with IonQ API compatible values.
    """
    passed_args = passed_args or {}
    ionq_circ, num_meas, meas_map = qiskit_circ_to_ionq_circ(circuit)
    creg_sizes, clbit_labels = get_register_sizes_and_labels(circuit.clbits)
    qreg_sizes, qubit_labels = get_register_sizes_and_labels(circuit.qubits)
    qiskit_header = compress_dict_to_metadata_string(
        {
            "memory_slots": circuit.num_clbits,  # int
            "global_phase": circuit.global_phase,  # float
            "n_qubits": circuit.num_qubits,  # int
            "name": circuit.name,  # str
            "creg_sizes": creg_sizes,  # list of [str, int] tuples cardinality memory_slots
            "clbit_labels": clbit_labels,  # list of [str, int] tuples cardinality memory_slots
            "qreg_sizes": qreg_sizes,  # list of [str, int] tuples cardinality num_qubits
            "qubit_labels": qubit_labels,  # list of [str, int] tuples cardinality num_qubits
        }
    )

    ionq_json = {
        "lang": "json",
        "target": backend_name[5:],
        "shots": passed_args["shots"],
        "body": {
            "qubits": circuit.num_qubits,
            "circuit": ionq_circ,
        },
        # store a couple of things we'll need later for result formatting
        "metadata": {
            "shots": str(passed_args["shots"]),
            "output_map": json.dumps(meas_map),
            "qiskit_header": qiskit_header,
        },
    }
    return json.dumps(ionq_json)


__all__ = [
    "qiskit_to_ionq",
    "qiskit_circ_to_ionq_circ",
    "compress_dict_to_metadata_string",
    "decompress_metadata_string_to_dict",
]
