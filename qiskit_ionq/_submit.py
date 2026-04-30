"""Shared submission layer: transpile, translate, submit to IonQ."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ionq_core.api.default import create_job
from ionq_core.models import json_multi_circuit_input as mci
from ionq_core.models import json_multi_circuit_job as mcj
from ionq_core.models import json_multi_circuit_job_settings as mcjs
from ionq_core.models import json_multi_circuit_job_settings_error_mitigation as mcjem
from ionq_core.models.native_circuit import NativeCircuit
from ionq_core.models.noise import Noise
from ionq_core.models.noise_model import NoiseModel
from ionq_core.models.qis_circuit import QISCircuit
from ionq_core.types import UNSET
from qiskit import transpile
from qiskit.circuit import QuantumCircuit

from ._translate import translate_native_gates, translate_qis_gates

if TYPE_CHECKING:
    from ionq_core.client import AuthenticatedClient
    from qiskit.transpiler import Target


def _translate_circuits(
    circuits: list[QuantumCircuit], gateset: str
) -> list[NativeCircuit | QISCircuit]:
    if gateset == "native":
        return [
            NativeCircuit(circuit=translate_native_gates(c), qubits=c.num_qubits)
            for c in circuits
        ]
    return [
        QISCircuit(circuit=translate_qis_gates(c), qubits=c.num_qubits)
        for c in circuits
    ]


def submit(
    *,
    client: AuthenticatedClient,
    backend: str,
    circuits: list[QuantumCircuit],
    shots: int,
    gateset: Literal["qis", "native"],
    target: Target,
    session_id: str | None = None,
    error_mitigation: dict | None = None,
    noise_model: NoiseModel | None = None,
    noise_seed: int | None = None,
) -> str:
    transpiled = [transpile(c, target=target) for c in circuits]
    body = mcj.JSONMultiCircuitJob(
        backend=backend,
        type_="ionq.multi-circuit.v1",
        input_=mci.JsonMultiCircuitInput(
            gateset=gateset, circuits=_translate_circuits(transpiled, gateset)
        ),
        shots=shots,
        session_id=session_id or UNSET,
        settings=mcjs.JSONMultiCircuitJobSettings(
            error_mitigation=mcjem.JSONMultiCircuitJobSettingsErrorMitigation(
                **error_mitigation
            ),
        )
        if error_mitigation is not None
        else UNSET,
        noise=Noise(
            model=noise_model, seed=noise_seed if noise_seed is not None else UNSET
        )
        if noise_model is not None
        else UNSET,
    )
    resp = create_job.sync(client=client, body=body)
    if resp is None:
        raise RuntimeError("Job creation failed: no response from API")
    return resp.id
