"""Convert ionq-core job responses to Qiskit Result and BitArray."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ionq_core.api.default import get_job_probabilities
from qiskit.primitives import BitArray
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData

if TYPE_CHECKING:
    from ionq_core.client import AuthenticatedClient


def _probs_to_counts(probs: dict[int, float], shots: int) -> dict[int, int]:
    """Convert probabilities to integer counts that sum exactly to shots."""
    if not probs:
        return {}
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    counts = {}
    remaining = shots
    for state, prob in sorted_items[:-1]:
        count = round(prob * shots)
        counts[state] = count
        remaining -= count
    counts[sorted_items[-1][0]] = remaining
    return {k: v for k, v in counts.items() if v > 0}


def _fetch_counts(client: AuthenticatedClient, job_id: str, shots: int) -> dict[int, int]:
    resp = get_job_probabilities.sync(uuid=job_id, client=client)
    probs = {int(k): float(v) for k, v in resp.additional_properties.items()} if resp else {}
    return _probs_to_counts(probs, shots)


def to_qiskit_result(client: AuthenticatedClient, job_id: str, num_qubits: int, shots: int) -> Result:
    int_counts = _fetch_counts(client, job_id, shots)
    bin_counts = {format(state, f"0{num_qubits}b"): count for state, count in int_counts.items()}
    return Result(
        backend_name="ionq",
        backend_version="1.0.0",
        job_id=job_id,
        success=True,
        results=[ExperimentResult(shots=shots, success=True, data=ExperimentResultData(counts=bin_counts))],
    )


def to_bitarray(client: AuthenticatedClient, job_id: str, num_qubits: int, shots: int) -> BitArray:
    counts = _fetch_counts(client, job_id, shots)
    return BitArray.from_counts(counts or {0: shots}, num_bits=num_qubits)  # ty: ignore[invalid-argument-type]
