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

from __future__ import annotations

import json
import gzip
import base64
import platform
import warnings
import os
from typing import Literal, Any
import functools
import time
import random
import requests
from dotenv import dotenv_values

from qiskit import __version__ as qiskit_terra_version
from qiskit.circuit import (
    controlledgate as q_cgates,
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
)

# Use this to get version instead of __version__ to avoid circular dependency.
from importlib_metadata import version
from qiskit_ionq.constants import ErrorMitigation
from . import exceptions as ionq_exceptions

# the qiskit gates that the IonQ backend can serialize to our IR
# not the actual hardware basis gates for the system — we do our own transpilation pass.
# also not an exact/complete list of the gates IonQ's backend takes
#   by name — please refer to IonQ docs for that.
#
# Some of these gates may be deprecated or removed in qiskit 1.0
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
    "PauliEvolution",
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
    "PauliEvolution": "pauliexp",
    "rxx": "xx",
    "ryy": "yy",
    "rzz": "zz",
    "sdg": "si",
    "sx": "v",
    "sxdg": "vi",
}

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


def qiskit_circ_to_ionq_circ(
    input_circuit: QuantumCircuit,
    gateset: Literal["qis", "native"] = "qis",
    ionq_compiler_synthesis: bool = False,
):
    """Build a circuit in IonQ's instruction format from qiskit instructions.

    .. ATTENTION:: This function ignores the following compiler directives:
       * ``barrier``

    Parameters:
        input_circuit (:class:`qiskit.circuit.QuantumCircuit`): A Qiskit quantum circuit.
        gateset (string): Set of gates to target. It can be QIS (required transpilation pass in
          IonQ backend, which is sent standard gates) or native (only IonQ native gates are
          allowed, in the future we may provide transpilation to these gates in Qiskit).
        ionq_compiler_synthesis (bool): Whether to opt-in to IonQ compiler's intelligent
          trotterization.

    Raises:
        IonQGateError: If an unsupported instruction is supplied.
        IonQMidCircuitMeasurementError: If a mid-circuit measurement is detected.
        IonQPauliExponentialError: If non-commuting PauliExponentials are found without
          the appropriate flag.

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
            raise ionq_exceptions.IonQGateError(instruction_name, gateset)

        # Process the instruction and convert.
        rotation: dict[str, Any] = {}
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
                if instruction_name == "PauliEvolution":
                    # rename rotation to time
                    rotation["time"] = rotation.pop("rotation")
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
        if instruction.num_qubits > 1 and not hasattr(instruction, "num_ctrl_qubits"):
            converted["targets"] = [
                input_circuit.qubits.index(qargs[i])
                for i in range(instruction.num_qubits)
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

        if instruction_name == "pauliexp":
            imag_coeff = any(coeff.imag for coeff in instruction.operator.coeffs)
            assert not imag_coeff, (
                "PauliEvolution gate must have real coefficients, "
                f"but got {imag_coeff}"
            )
            terms = [term[0] for term in instruction.operator.to_list()]
            if not ionq_compiler_synthesis and not paulis_commute(terms):
                raise ionq_exceptions.IonQPauliExponentialError(
                    f"You have included a PauliEvolutionGate with non-commuting terms: {terms}."
                    "To decompose it with IonQ hardware-aware synthesis, resubmit with the "
                    "IONQ_COMPILER_SYNTHESIS flag."
                )
            targets = [
                input_circuit.qubits.index(qargs[i])
                for i in range(instruction.num_qubits)
            ]
            coefficients = [coeff.real for coeff in instruction.operator.coeffs]
            gate = {
                "gate": instruction_name,
                "targets": targets,
                "terms": terms,
                "coefficients": coefficients,
            }
            converted.update(gate)

        # if there's a valid instruction after a measurement,
        if num_meas > 0:
            # see if any of the involved qubits have been measured,
            # and raise if so — no mid-circuit measurement!
            controls_and_targets = converted.get("targets", []) + converted.get(
                "controls", []
            )
            if any(i in meas_map for i in controls_and_targets):
                raise ionq_exceptions.IonQMidCircuitMeasurementError(
                    input_circuit.qubits.index(qargs[0]), instruction_name
                )

        output_circuit.append({**converted, **rotation})

    return output_circuit, num_meas, meas_map


def paulis_commute(pauli_terms: list[str]) -> bool:
    """Check if a list of Pauli terms commute.

    Args:
        pauli_terms (list): A list of Pauli terms.

    Returns:
        bool: Whether the Pauli terms commute.
    """
    for i, term in enumerate(pauli_terms):
        for other_term in pauli_terms[i:]:
            assert len(term) == len(other_term)
            anticommutation_parity = 0
            for index, char in enumerate(term):
                other_char = other_term[index]
                if "I" not in (char, other_char):
                    if char != other_char:
                        anticommutation_parity += 1
            if anticommutation_parity % 2 == 1:
                return False
    return True


def get_register_sizes_and_labels(
    registers: list[QuantumRegister | ClassicalRegister],
) -> tuple[list, list]:
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


def compress_to_metadata_string(
    metadata: dict | list,
) -> str:  # pylint: disable=invalid-name
    """
    Convert a metadata object to a compact string format (dumped, gzipped, base64 encoded)
    for storing in IonQ API metadata

    Parameters:
        metadata (dict or list): a dict or list of dicts with metadata relevant
            to building the results object on a returned job.

    Returns:
        str: encoded string

    """
    serialized = json.dumps(metadata)
    compressed = gzip.compress(serialized.encode("utf-8"))
    encoded = base64.b64encode(compressed)
    return encoded.decode()


def decompress_metadata_string(
    input_string: str,
) -> dict | list:  # pylint: disable=invalid-name
    """
    Convert compact string format (dumped, gzipped, base64 encoded) from
    IonQ API metadata back into a dict or list of dicts relevant to building
    the results object on a returned job.

    Parameters:
        input_string (str): compressed string format of metadata dict

    Returns:
        dict or list: decompressed metadata dict or list of dicts
    """
    if input_string is None:
        return None
    encoded = input_string.encode()
    decoded = base64.b64decode(encoded)
    decompressed = gzip.decompress(decoded)
    return json.loads(decompressed)


def qiskit_to_ionq(
    circuit, backend, passed_args=None, extra_query_params=None, extra_metadata=None
) -> str:
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
    ionq_circs = []
    multi_circuit = False
    if isinstance(circuit, (list, tuple)):
        multi_circuit = True
        for circ in circuit:
            ionq_circ, _, meas_map = qiskit_circ_to_ionq_circ(
                circ,
                backend.gateset(),
                extra_metadata.get("ionq_compiler_synthesis", False),
            )
            ionq_circs.append((ionq_circ, meas_map, circ.name))
    else:
        ionq_circs, _, meas_map = qiskit_circ_to_ionq_circ(
            circuit,
            backend.gateset(),
            extra_metadata.get("ionq_compiler_synthesis", False),
        )
        circuit = [circuit]
    circuit: list[QuantumCircuit] | tuple[QuantumCircuit, ...]  # type: ignore[no-redef]
    metadata_list = [
        {
            "memory_slots": circ.num_clbits,  # int
            "global_phase": circ.global_phase,  # float
            "n_qubits": circ.num_qubits,  # int
            "name": circ.name,  # str
            # list of [str, int] tuples cardinality memory_slots
            "creg_sizes": get_register_sizes_and_labels(circ.cregs)[0],
            # list of [str, int] tuples cardinality memory_slots
            "clbit_labels": get_register_sizes_and_labels(circ.cregs)[1],
            # list of [str, int] tuples cardinality num_qubits
            "qreg_sizes": get_register_sizes_and_labels(circ.qregs)[0],
            # list of [str, int] tuples cardinality num_qubits
            "qubit_labels": get_register_sizes_and_labels(circ.qregs)[1],
            # custom metadata from the circuits
            **({"metadata": circ.metadata} if circ.metadata else {}),
        }
        for circ in circuit
    ]

    qiskit_header = compress_to_metadata_string(
        metadata_list if multi_circuit else metadata_list[0]
    )

    # Common fields
    target = backend.name()[5:] if backend.name().startswith("ionq") else backend.name()
    name = passed_args.get("name") or (
        f"{len(circuit)} circuits" if multi_circuit else circuit[0].name
    )
    error_mitigation = passed_args.get("error_mitigation") or backend.options.get(
        "error_mitigation"
    )

    # Settings block
    settings = {
        "compilation": {
            "opt": extra_metadata.get("compilation", {}).get("opt", "0"),
            "precision": extra_metadata.get("compilation", {}).get(
                "precision", "1E-3"
            ),
            "gate_basis": extra_metadata.get("compilation", {}).get(
                "gate_basis", "ZZ"
            ),
        },
        "error_mitigation": {
            "debiasing": {
                "method": (
                    error_mitigation.value
                    if isinstance(error_mitigation, ErrorMitigation)
                    else (error_mitigation or "none")
                )
            }
        },
    }

    # Flatten IonQ circuit instructions into MVP‐style list of dicts
    def serialize_instructions(ionq_circ):
        insts = []
        for op in ionq_circ:
            gate = op.get("gate")
            # figure out the single target
            if "target" in op:
                tgt = op["target"]
            elif "targets" in op and op["targets"]:
                tgt = op["targets"][0]
            elif "qubits" in op and op["qubits"]:
                tgt = op["qubits"][0]
            else:
                raise KeyError(f"No target found in operation {op}")

            inst = {"gate": gate, "target": tgt}

            # optional control
            if "control" in op:
                inst["control"] = op["control"]
            elif "controls" in op and op["controls"]:
                inst["control"] = op["controls"][0]

            # optional rotation/angle
            if "rotation" in op:
                inst["rotation"] = op["rotation"]
            elif "angle" in op:
                inst["rotation"] = op["angle"]

            insts.append(inst)
        return insts

    # Input block
    input_block = {
        "format": "ionq.circuit.v0",
        "gateset": backend.gateset(),
        "qubits": max(c.num_qubits for c in circuit),
        "circuits": [{"circuit": []}],
    }
    if multi_circuit:
        for ionq_circ, _, name in ionq_circs:
            input_block["circuits"][0]["circuit"].append(serialize_instructions(ionq_circ))
    else:
        input_block["circuits"][0]["circuit"] = serialize_instructions(ionq_circs)

    # Final MVP payload
    job_payload = {
        "type": "ionq.circuit.v1",
        "name": name,
        "shots": passed_args.get("shots", None),
        "dry_run": passed_args.get("dry_run", False),
        "input": input_block,
        "backend": target,
    }
    if extra_query_params.get("metadata") is not None:
        job_payload["metadata"] = extra_query_params.get("metadata")
    if extra_metadata.get("compilation") is None:
        del settings["compilation"]

    return json.dumps(job_payload, cls=SafeEncoder)


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


class SafeEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles non-JSON-safe objects by converting them to strings.
    """

    def default(self, o):
        funcs = [
            lambda: super().default(o),
            lambda: str(o),
            lambda: repr(o),
        ]

        for func in funcs:
            try:
                return func()
            except Exception as exception:  # pylint: disable=broad-except
                warnings.warn(
                    f"Unable to encode {o} using {func.__name__}: {exception}"
                )

        return "unknown"


def resolve_credentials(token: str | None = None, url: str | None = None) -> dict:
    """Resolve credentials for use in IonQ API calls.

    If the provided ``token`` and ``url`` are both ``None``, then these values
    are loaded from the ``IONQ_API_TOKEN`` and ``IONQ_API_URL``
    environment variables, respectively.

    If no url is discovered, then ``https://api.ionq.co/v0.4`` is used.

    Args:
        token (str): IonQ API access token.
        url (str, optional): IonQ API url. Defaults to ``None``.

    Returns:
        dict[str]: A dict with "token" and "url" keys, for use by a client.
    """
    env_values = dotenv_values()
    env_token = (
        env_values.get("QISKIT_IONQ_API_TOKEN")
        or env_values.get("IONQ_API_KEY")
        or env_values.get("IONQ_API_TOKEN")
        or os.getenv("QISKIT_IONQ_API_TOKEN")
        or os.getenv("IONQ_API_KEY")
        or os.getenv("IONQ_API_TOKEN")
    )
    env_url = (
        env_values.get("QISKIT_IONQ_API_URL")
        or env_values.get("IONQ_API_URL")
        or os.getenv("QISKIT_IONQ_API_URL")
        or os.getenv("IONQ_API_URL")
    )
    return {
        "token": token or env_token,
        "url": url or env_url or "https://api.ionq.co/v0.4",
    }


def get_n_qubits(backend: str, fallback: int = 100) -> int:
    """Get the number of qubits for a given backend.

    Args:
        backend (str): The name of the backend.
        fallback (int): Fallback number of qubits if API call fails.

    Returns:
        int: The number of qubits for the backend.
    """
    creds = resolve_credentials()
    url = creds.get("url")

    target = backend.split("ionq_")[-1] if backend.startswith("ionq_qpu.") else backend

    try:
        response = requests.get(url=f"{url}/backends", timeout=5)
        response.raise_for_status()  # Ensure we catch any HTTP errors
        return next(
            (item["qubits"] for item in response.json() if item["backend"] == target),
            fallback,
        )  # Default to fallback if no backend found
    except Exception as exception:  # pylint: disable=broad-except
        warnings.warn(
            f"Unable to get qubit count for {backend}: {exception}. Defaulting to {fallback}."
        )
        return fallback


def retry(
    exceptions: Any,
    tries: int = -1,
    delay: float = 0,
    max_delay: float = float("inf"),
    backoff: float = 1,
    jitter: float = 0,
    enable_logging: bool = True,
):  # pylint: disable=too-many-positional-arguments
    """Retry decorator with exponential backoff.

    Args:
        exceptions: The exception(s) to catch. Can be a tuple of exceptions.
        tries: Number of attempts before giving up. -1 means infinite tries.
        delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries.
        backoff: Multiplier applied to delay after each retry.
        jitter: Maximum random jitter added to delay.
        enable_logging: Whether to log failures.
    """

    def deco_retry(func):
        @functools.wraps(func)
        def f_retry(*args, **kwargs):
            _tries, _delay = tries, delay
            while _tries != 0:
                try:
                    return func(*args, **kwargs)
                except exceptions as exception:
                    _tries -= 1
                    if _tries == 0:
                        raise
                    if enable_logging:
                        warnings.warn(
                            f"Retrying {func.__name__} "
                            f"{f'{_tries} more time(s) ' if _tries > 0 else ''}"
                            f"after {exception}"
                        )
                    sleep = _delay + (random.uniform(0, jitter) if jitter else 0)
                    time.sleep(sleep)
                    _delay *= backoff
                    if _delay > max_delay:
                        _delay = min(_delay, max_delay)

            return None

        return f_retry

    return deco_retry


__all__ = [
    "qiskit_to_ionq",
    "qiskit_circ_to_ionq_circ",
    "compress_to_metadata_string",
    "decompress_metadata_string",
    "get_user_agent",
    "resolve_credentials",
    "get_n_qubits",
    "retry",
]
