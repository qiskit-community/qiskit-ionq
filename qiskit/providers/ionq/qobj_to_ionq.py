import json

from .exceptions import *


def qobj_to_ionq(qobj):
    """map a job and its instructions to a JSON object for IonQ's API"""
    if len(qobj.experiments) > 1:
        raise IonQJobError("IonQ backends do not support multi-experiment jobs.")
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


def build_output_map(qobj):
    """IonQ's API does not allow ad-hoc remapping of classical to quantum registers,
    instead always returning quantum[i] as classical[i] in the return bitstring.
    We create this output map from the measure instructions in the program so we can
    do an arbitrary remapping later based on the user's desired output mapping.
    """
    output_map = {}
    measurements = 0
    for instruction in qobj.experiments[0].instructions:
        if instruction.name == "measure":
            output_map[instruction.qubits[0]] = instruction.memory[0]
            measurements += 1
        else:
            if measurements < 0:
                raise IonQGateError("all measurements must occur at the end of the circuit")
    if measurements == 0:
        raise ValueError("circuit must have at least one measurement")
    return output_map


def build_circuit(instructions):
    """build a circuit in IonQ's instruction format from qiskit instructions"""
    circuit = []
    invalid_instructions = [
        "barrier",
        "reset",
        "u1",
        "u2",
        "u3",
        "cu1",
        "cu2",
        "cu3",
    ]
    for instruction in instructions:

        if instruction.name in invalid_instructions:
            raise IonQGateError(instruction.name)

        if instruction.name != "measure":
            rotation = {}
            if hasattr(instruction, "params"):
                rotation = {"rotation": instruction.params[0]}
            if instruction.name[0] == "c":
                is_double_control = instruction.name[1] == "c"
                gate = instruction.name[1:]
                controls = [instruction.qubits[0]]
                target = instruction.qubits[1]
                if is_double_control:
                    gate = instruction.name[2:]
                    controls = [instruction.qubits[0], instruction.qubits[1]]
                    target = instruction.qubits[2]
                circuit.append(
                    {
                        "gate": gate,
                        "controls": controls,
                        "target": target,
                        **rotation,
                    }
                )
            else:
                circuit.append(
                    {
                        "gate": instruction.name,
                        "target": instruction.qubits[0],
                        **rotation,
                    }
                )
    return circuit
