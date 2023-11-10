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
Helper methods for mapping Qiskit classes
to IonQ REST API compatible values.
"""

import json
import gzip
import base64
import platform
import warnings

from qiskit import __version__ as qiskit_terra_version
from qiskit.circuit import controlledgate as q_cgates
from qiskit.circuit.library import standard_gates as q_gates

# Use this to get version instead of __version__ to avoid circular dependency.
from importlib_metadata import version
from qiskit_ionq.constants import ErrorMitigation
from . import exceptions

# the qiskit gates that the IonQ backend can serialize to our IR
# not the actual hardware basis gates for the system — we do our own transpilation pass.
# also not an exact/complete list of the gates IonQ's backend takes
#   by name — please refer to IonQ docs for that.
ionq_basis_gates = [
    "ccx",
    "ch",
    "cnot",
    "cp",
    "crx",
    "cry",
    "crz",
    "csx",
    "cx",
    "cy",
    "cz",
    "h",
    "i",
    "id",
    "mcp",
    "mcphase",
    "mct",
    "mcx",
    "mcx_gray",
    "measure",
    "p",
    "rx",
    "rxx",
    "ry",
    "ryy",
    "rz",
    "rzz",
    "s",
    "sdg",
    "swap",
    "sx",
    "sxdg",
    "t",
    "tdg",
    "toffoli",
    "x",
    "y",
    "z",
]

ionq_api_aliases = {  # todo fix alias bug
    "cp": "cz",
    "csx": "cv",
    "mcphase": "cz",
    "ccx": "cx",  # just one C for all mcx
    "mcx": "cx",  # just one C for all mcx
    "mcx_gray": "cx",  # just one C for all mcx
    "tdg": "ti",
    "p": "z",
    "rxx": "xx",
    "ryy": "yy",
    "rzz": "zz",
    "sdg": "si",
    "sx": "v",
    "sxdg": "vi",
}

multi_target_uncontrolled_gates = (
    q_gates.SwapGate,
    q_gates.RXXGate,
    q_gates.RYYGate,
    q_gates.RZZGate,
)

# https://ionq.com/docs/getting-started-with-native-gates
ionq_native_basis_gates = [
    "gpi",  # TODO All single qubit gates can transpile into GPI/GPI2
    "gpi2",
    "ms",  # Pairwise MS gate
    "zz",  # ZZ gate
]

# Each language corresponds to a different set of basis gates.
GATESET_MAP = {
    "qis": ionq_basis_gates,
    "native": ionq_native_basis_gates,
}


def qiskit_circ_to_ionq_circ(input_circuit, gateset="qis"):
    """Build a circuit in IonQ's instruction format from qiskit instructions.

    .. ATTENTION:: This function ignores the following compiler directives:
       * ``barrier``

    Parameters:
        input_circuit (:class:`qiskit.circuit.QuantumCircuit`): A Qiskit quantum circuit.
        gateset (string): Set of gates to target. It can be QIS (required transpilation pass in
          IonQ backend, which is sent standard gates) or native (only IonQ native gates are
          allowed, in the future we may provide transpilation to these gates in Qiskit).

    Raises:
        IonQGateError: If an unsupported instruction is supplied.
        IonQMidCircuitMeasurementError: If a mid-circuit measurement is detected.

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
            meas_map[input_circuit.clbits.index(cargs[0])] = input_circuit.qubits.index(
                qargs[0]
            )
            num_meas += 1
            continue

        # serialized identity gate is a no-op
        if instruction_name == "id":
            continue

        # Raise out for instructions we don't support.
        if instruction_name not in GATESET_MAP[gateset]:
            raise exceptions.IonQGateError(instruction_name, gateset)

        # Process the instruction and convert.
        rotation = {}
        if len(instruction.params) > 0:
            if gateset == "qis" or (
                len(instruction.params) == 1 and instruction_name != "zz"
            ):
                # The float is here to cast Qiskit ParameterExpressions to numbers
                rotation = {
                    ("rotation" if gateset == "qis" else "phase"): float(
                        instruction.params[0]
                    )
                }
            elif instruction_name in {"zz"}:
                rotation = {"angle": instruction.params[0]}
            else:
                rotation = {
                    "phases": [float(t) for t in instruction.params[:2]],
                    "angle": instruction.params[2],
                }

        # Default conversion is simple, just gate & target(s).
        targets = [input_circuit.qubits.index(qargs[0])]
        if instruction_name in {"ms", "zz"}:
            targets.append(input_circuit.qubits.index(qargs[1]))

        converted = (
            {"gate": instruction_name, "targets": targets}
            if instruction_name not in {"gpi", "gpi2"}
            else {
                "gate": instruction_name,
                "target": targets[0],
            }
        )

        # re-alias certain names
        if instruction_name in ionq_api_aliases:
            instruction_name = ionq_api_aliases[instruction_name]
            converted["gate"] = instruction_name

        # Make sure uncontrolled multi-targets use all qargs.
        if isinstance(instruction, multi_target_uncontrolled_gates):
            converted["targets"] = [
                input_circuit.qubits.index(qargs[0]),
                input_circuit.qubits.index(qargs[1]),
            ]

        # If this is a controlled gate, make sure to set control qubits.
        if isinstance(instruction, q_cgates.ControlledGate):
            gate = instruction_name[1:]  # trim the leading c
            controls = [input_circuit.qubits.index(qargs[0])]
            targets = [input_circuit.qubits.index(qargs[1])]
            # If this is a multi-control, use more than one qubit.
            if instruction.num_ctrl_qubits > 1:
                controls = [
                    input_circuit.qubits.index(qargs[i])
                    for i in range(instruction.num_ctrl_qubits)
                ]
                targets = [
                    input_circuit.qubits.index(qargs[instruction.num_ctrl_qubits])
                ]
            if gate == "swap":
                # If this is a cswap, we have two targets:
                targets = [
                    input_circuit.qubits.index(qargs[-2]),
                    input_circuit.qubits.index(qargs[-1]),
                ]

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
            controls_and_targets = converted.get("targets", []) + converted.get(
                "controls", []
            )
            if any(i in meas_map for i in controls_and_targets):
                raise exceptions.IonQMidCircuitMeasurementError(
                    input_circuit.qubits.index(qargs[0]), instruction_name
                )

        output_circuit.append({**converted, **rotation})

    return output_circuit, num_meas, meas_map


def get_register_sizes_and_labels(registers):
    """Returns a tuple of sizes and labels in for a given register

    Args:
        registers (list): A list of of qiskit registers.

    Returns:
        tuple: A list of sizes and labels for the provided list of registers.
    """
    sizes = []
    labels = []

    for register in registers:
        for index, _ in enumerate(register):
            # we actually don't need to know anything about the bit itself, just its position
            size = [register.name, register.size]
            label = [register.name, index]

            if size not in sizes:
                sizes.append(size)

            labels.append(label)

    return sizes, labels


# slightly goofy workaround to account for the fact that IonQ's "arbitrary" metadata field
# only takes string KV pairs with value max length 400
# so we try and pack it down into a more-compressed string format
# and raise if it's still too long
# TODO: make this behavior a little nicer (dict metadata) on IonQ side; fix here when we do
def compress_dict_to_metadata_string(metadata_dict):  # pylint: disable=invalid-name
    """
    Convert a dict to a compact string format (dumped, gzipped, base64 encoded)
    for storing in IonQ API metadata

    Parameters:
        metadata_dict (dict): a dict with metadata relevant to building the results
            object on a returned job.

    Returns:
        str: encoded string

    Raises:
        IonQMetadataStringError: If the base64 encoded `json.dumps`'d dict is
            greater than 400 characters.
    """
    serialized = json.dumps(metadata_dict)
    compressed = gzip.compress(serialized.encode("utf-8"))
    encoded = base64.b64encode(compressed)
    encoded_string = encoded.decode()
    encoded_string_length = len(encoded_string)
    if encoded_string_length > 400:  # 400 char is an IonQ API limitation
        raise exceptions.IonQMetadataStringError(encoded_string_length)
    return encoded_string


def decompress_metadata_string_to_dict(input_string):  # pylint: disable=invalid-name
    """
    Convert compact string format (dumped, gzipped, base64 encoded) from
    IonQ API metadata back into a dict relevant to building the results object
    on a returned job.

    Parameters:
        input_string (str): compressed string format of metadata dict

    Returns:
        dict: decompressed metadata dict
    """
    if input_string is None:
        return None
    encoded = input_string.encode()
    decoded = base64.b64decode(encoded)
    decompressed = gzip.decompress(decoded)
    return json.loads(decompressed)


def qiskit_to_ionq(
    circuit, backend, passed_args=None, extra_query_params=None, extra_metadata=None
):
    """Convert a Qiskit circuit to a IonQ compatible dict.

    Parameters:
        circuit (:class:`qiskit.circuit.QuantumCircuit`): A Qiskit quantum circuit.
        backend (:class:`qiskit_ionq.IonQBackend`): The IonQ backend.
        passed_args (dict): Dictionary containing additional passed arguments, eg. shots.
        extra_query_params (dict): Specify any parameters to include in the request
        extra_metadata (dict): Specify any additional metadata to include.

    Returns:
        str: A string / JSON-serialized dictionary with IonQ API compatible values.
    """
    passed_args = passed_args or {}
    extra_query_params = extra_query_params or {}
    extra_metadata = extra_metadata or {}
    ionq_circ, _, meas_map = qiskit_circ_to_ionq_circ(circuit, backend.gateset())
    creg_sizes, clbit_labels = get_register_sizes_and_labels(circuit.cregs)
    qreg_sizes, qubit_labels = get_register_sizes_and_labels(circuit.qregs)
    qiskit_header = compress_dict_to_metadata_string(
        {
            "memory_slots": circuit.num_clbits,  # int
            "global_phase": circuit.global_phase,  # float
            "n_qubits": circuit.num_qubits,  # int
            "name": circuit.name,  # str
            # list of [str, int] tuples cardinality memory_slots
            "creg_sizes": creg_sizes,
            # list of [str, int] tuples cardinality memory_slots
            "clbit_labels": clbit_labels,
            # list of [str, int] tuples cardinality num_qubits
            "qreg_sizes": qreg_sizes,
            # list of [str, int] tuples cardinality num_qubits
            "qubit_labels": qubit_labels,
        }
    )

    target = backend.name()[5:] if backend.name().startswith("ionq") else backend.name()
    if target == "qpu":
        target = "qpu.harmony"  # todo default to cheapest available option
        warnings.warn(
            "The ionq_qpu backend is deprecated. Defaulting to ionq_qpu.harmony.",
            DeprecationWarning,
            stacklevel=2,
        )
    ionq_json = {
        "target": target,
        "shots": passed_args.get("shots"),
        "name": circuit.name,
        "input": {
            "format": "ionq.circuit.v0",
            "gateset": backend.gateset(),
            "qubits": circuit.num_qubits,
            "circuit": ionq_circ,
        },
        "registers": {"meas_mapped": meas_map} if meas_map else {},
        # store a couple of things we'll need later for result formatting
        "metadata": {
            "shots": str(passed_args.get("shots")),
            "sampler_seed": str(passed_args.get("sampler_seed")),
            "qiskit_header": qiskit_header,
        },
    }
    if target == "simulator":
        ionq_json["noise"] = {
            "model": passed_args.get("noise_model") or backend.options.noise_model,
            "seed": backend.options.sampler_seed,
        }
    ionq_json.update(extra_query_params)
    # merge circuit and extra metadata
    ionq_json["metadata"].update(extra_metadata)
    settings = passed_args.get("job_settings") or None
    if settings is not None:
        ionq_json["settings"] = settings
    error_mitigation = passed_args.get("error_mitigation")
    if error_mitigation and isinstance(error_mitigation, ErrorMitigation):
        ionq_json["error_mitigation"] = error_mitigation.value
    return json.dumps(ionq_json)


def get_user_agent():
    """Generates the user agent string which is helpful in identifying
    different tools in the internet. Valid user-agent ionq_client header that
    indicates the request is from qiskit_ionq along with the system, os,
    python,libraries details.

    Returns:
        str: A string of generated user agent.
    """
    # from qiskit_ionq import __version__ as qiskit_ionq_version

    os_string = f"os/{platform.system()}"
    provider_version_string = f"qiskit-ionq/{version('qiskit_ionq')}"
    qiskit_terra_version_string = f"qiskit-terra/{qiskit_terra_version}"
    python_version_string = f"python/{platform.python_version()}"
    return (
        f"{provider_version_string} "
        f"({qiskit_terra_version_string}) {os_string} "
        f"({python_version_string})"
    )


__all__ = [
    "qiskit_to_ionq",
    "qiskit_circ_to_ionq_circ",
    "compress_dict_to_metadata_string",
    "decompress_metadata_string_to_dict",
    "get_user_agent",
]
