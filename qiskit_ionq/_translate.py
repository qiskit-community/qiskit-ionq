"""Translate Qiskit QuantumCircuit to ionq-core gate models."""

from __future__ import annotations

from ionq_core.models.gate_native_gate import GateNativeGate
from ionq_core.models.gate_qis_gate import GateQisGate
from ionq_core.models.qis_gate import QisGate
from ionq_core.types import UNSET
from qiskit.circuit import QuantumCircuit

_SKIP = frozenset({"measure", "barrier", "delay"})

_QIS_NAMES: dict[str, QisGate] = {
    "h": "h",
    "x": "x",
    "y": "y",
    "z": "z",
    "s": "s",
    "sdg": "si",
    "t": "t",
    "tdg": "ti",
    "sx": "v",
    "sxdg": "vi",
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "cx": "cnot",
    "swap": "swap",
    "rxx": "xx",
    "ryy": "yy",
    "rzz": "zz",
}
_PARAMETERIZED = frozenset({"rx", "ry", "rz", "xx", "yy", "zz"})
_NATIVE = frozenset({"gpi", "gpi2", "ms", "zz"})


def translate_qis_gates(circuit: QuantumCircuit) -> list[GateQisGate]:
    gates: list[GateQisGate] = []
    for inst in circuit.data:
        name = inst.operation.name
        if name in _SKIP:
            continue
        ionq = _QIS_NAMES.get(name)
        if ionq is None:
            raise ValueError(f"Unsupported gate for IonQ QIS translation: {name!r}")
        qubits = [circuit.find_bit(q).index for q in inst.qubits]
        params = [float(p) for p in inst.operation.params]
        rotation = params[0] if ionq in _PARAMETERIZED and params else UNSET
        if ionq == "cnot":
            gates.append(
                GateQisGate(
                    gate=ionq, control=qubits[0], target=qubits[1], rotation=rotation
                )
            )
        elif len(qubits) == 2:
            gates.append(GateQisGate(gate=ionq, targets=qubits, rotation=rotation))
        else:
            gates.append(GateQisGate(gate=ionq, target=qubits[0], rotation=rotation))
    return gates


def translate_native_gates(circuit: QuantumCircuit) -> list[GateNativeGate]:
    gates: list[GateNativeGate] = []
    for inst in circuit.data:
        name = inst.operation.name
        if name in _SKIP:
            continue
        if name not in _NATIVE:
            raise ValueError(f"Unsupported gate for IonQ native translation: {name!r}")
        qubits = [circuit.find_bit(q).index for q in inst.qubits]
        params = [float(p) for p in inst.operation.params]
        if name in ("gpi", "gpi2"):
            gates.append(GateNativeGate(gate=name, target=qubits[0], phase=params[0]))
        elif name == "ms":
            gates.append(
                GateNativeGate(
                    gate="ms", targets=qubits, phases=params[:2], angle=params[2]
                )
            )
        else:
            gates.append(GateNativeGate(gate="zz", targets=qubits, phase=params[0]))
    return gates
