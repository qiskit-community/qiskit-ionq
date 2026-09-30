"""Qiskit core workflow example for the IonQ provider.

This example keeps cloud submission opt-in. By default it builds a GHZ
circuit, evaluates a simple local observable with qiskit.quantum_info, and
transpiles the measured circuit against an IonQ backend target.

Run locally without submitting:

    python example/qiskit_core_workflow.py

Submit only when you intentionally pass --submit and have an IonQ API token.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector

# Make the example runnable from a source checkout before qiskit-ionq is installed.
REPO_ROOT = Path(__file__).resolve().parents[1]
if (REPO_ROOT / "qiskit_ionq").exists():
    sys.path.insert(0, str(REPO_ROOT))

from qiskit_ionq import IonQProvider


def build_ghz_state_prep(num_qubits: int = 3) -> QuantumCircuit:
    """Return an unmeasured GHZ state-preparation circuit."""
    if num_qubits < 2:
        raise ValueError("GHZ examples need at least two qubits.")

    circuit = QuantumCircuit(num_qubits, name=f"ghz_{num_qubits}")
    circuit.h(0)
    for target in range(1, num_qubits):
        circuit.cx(0, target)
    return circuit


def build_ghz_measurement_circuit(num_qubits: int = 3) -> QuantumCircuit:
    """Return a measured GHZ circuit suitable for backend.run(...)."""
    circuit = QuantumCircuit(num_qubits, num_qubits, name=f"ghz_{num_qubits}_meas")
    circuit.compose(build_ghz_state_prep(num_qubits), inplace=True)
    circuit.measure(range(num_qubits), range(num_qubits))
    return circuit


def local_pair_expectation(circuit: QuantumCircuit) -> float:
    """Compute <ZZI...> locally for the first two GHZ qubits."""
    pauli = "ZZ" + ("I" * (circuit.num_qubits - 2))
    observable = SparsePauliOp.from_list([(pauli, 1.0)])
    state = Statevector.from_instruction(circuit)
    return float(state.expectation_value(observable).real)


def transpile_for_backend(circuit: QuantumCircuit, backend) -> QuantumCircuit:
    """Use Qiskit's transpiler with an IonQ backend target."""
    return transpile(circuit, backend=backend, optimization_level=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="ionq_simulator")
    parser.add_argument("--gateset", choices=("qis", "native"), default="qis")
    parser.add_argument("--qubits", type=int, default=3)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--token", default=os.getenv("IONQ_API_TOKEN"))
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit to IonQ Cloud. Omitted by default to avoid cloud usage.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compile on IonQ Cloud without executing shots. Requires --submit.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    provider = IonQProvider(args.token)
    backend = provider.get_backend(args.backend, gateset=args.gateset)

    state_prep = build_ghz_state_prep(args.qubits)
    measured = build_ghz_measurement_circuit(args.qubits)
    transpiled = transpile_for_backend(measured, backend)

    print(f"Local <ZZI...> expectation: {local_pair_expectation(state_prep):.1f}")
    print(f"Backend target: {backend.name} ({args.gateset})")
    print(f"Original ops: {dict(measured.count_ops())}")
    print(f"Transpiled ops: {dict(transpiled.count_ops())}")

    if not args.submit:
        print("Not submitting. Pass --submit to run or dry-run on IonQ Cloud.")
        return

    if not args.token:
        raise SystemExit(
            "Set IONQ_API_TOKEN or pass --token before using --submit."
        )

    job = backend.run(transpiled, shots=args.shots, dry_run=args.dry_run)
    print(f"Submitted job: {job.job_id()}")
    job.wait_for_final_state()

    if args.dry_run:
        print(job.compiled_circuit(lang="qasm3"))
    else:
        print(job.result().get_counts())


if __name__ == "__main__":
    main()
