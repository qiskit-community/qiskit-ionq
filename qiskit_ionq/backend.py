"""IonQ backend for Qiskit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target

from ._submit import submit
from ._target import build_target
from .job import IonQJob

if TYPE_CHECKING:
    from ionq_core.client import AuthenticatedClient
    from ionq_core.models.backend import Backend


class IonQBackend(BackendV2):
    def __init__(
        self,
        provider,
        backend_info: Backend,
        client: AuthenticatedClient,
        gateset: Literal["qis", "native"] = "qis",
    ):
        super().__init__(name=backend_info.backend, provider=provider)
        self._client = client
        self._backend_info = backend_info
        self._gateset = gateset
        self._target = build_target(backend_info, client, gateset)

    @property
    def target(self) -> Target:
        return self._target

    @property
    def max_circuits(self) -> int | None:
        return None

    @property
    def gateset(self) -> str:
        return self._gateset

    @classmethod
    def _default_options(cls) -> Options:
        return Options(shots=1024, error_mitigation=None, noise_model=None, noise_seed=None)

    def run(self, run_input, **options) -> IonQJob:
        circuits = [run_input] if isinstance(run_input, QuantumCircuit) else list(run_input)
        if len(circuits) > 1 and len({c.num_qubits for c in circuits}) > 1:
            raise ValueError(
                "All circuits must have the same number of qubits. Use IonQSampler for mixed-width batching."
            )
        keys = ("shots", "error_mitigation", "noise_model", "noise_seed")
        opt = {k: options.get(k, getattr(self.options, k)) for k in keys}
        job_id = submit(
            client=self._client,
            backend=self._backend_info.backend,
            circuits=circuits,
            gateset=self._gateset,
            target=self._target,
            **opt,
        )
        return IonQJob(
            backend=self,
            job_id=job_id,
            client=self._client,
            num_qubits=circuits[0].num_qubits,
            shots=opt["shots"],
        )
