"""IonQ Sampler primitive for Qiskit."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ionq_core import wait_for_job
from ionq_core.api.default import cancel_job, get_job
from ionq_core.types import Unset
from qiskit.primitives import BasePrimitiveJob, BaseSamplerV2, DataBin, PrimitiveResult, SamplerPubResult
from qiskit.primitives.containers.sampler_pub import SamplerPub

from ._result import to_bitarray
from ._submit import submit

if TYPE_CHECKING:
    from .backend import IonQBackend
    from .session import IonQSession


class IonQSamplerJob(BasePrimitiveJob):
    def __init__(self, job_id: str, client, num_qubits_list: list[int], shots_list: list[int]):
        super().__init__(job_id)
        self._client = client
        self._num_qubits_list = num_qubits_list
        self._shots_list = shots_list
        self._result = None

    def result(self) -> PrimitiveResult[SamplerPubResult]:
        if self._result is not None:
            return self._result
        parent = wait_for_job(self._client, self.job_id())
        child_ids = getattr(parent, "child_job_ids", None)
        job_ids = [self.job_id()] if isinstance(child_ids, Unset) or child_ids is None else child_ids
        pub_results = []
        for i, jid in enumerate(job_ids):
            ba = to_bitarray(self._client, jid, self._num_qubits_list[i], self._shots_list[i])
            pub_results.append(SamplerPubResult(data=DataBin(meas=ba, shape=ba.shape), metadata={}))
        self._result = PrimitiveResult(pub_results=pub_results, metadata={})
        return self._result

    def status(self) -> str:
        resp = get_job.sync(uuid=self.job_id(), client=self._client)
        if resp is None:
            raise RuntimeError("Failed to fetch job status")
        return resp.status

    def done(self) -> bool:
        return self.status() == "completed"

    def running(self) -> bool:
        return self.status() == "started"

    def cancelled(self) -> bool:
        return self.status() == "canceled"

    def in_final_state(self) -> bool:
        return self.status() in {"completed", "failed", "canceled"}

    def cancel(self):
        cancel_job.sync(uuid=self.job_id(), client=self._client)


class IonQSampler(BaseSamplerV2):
    def __init__(self, backend: IonQBackend, *, session: IonQSession | None = None):
        self._backend = backend
        self._client = backend._client
        self._session = session

    def run(self, pubs, *, shots: int | None = None) -> IonQSamplerJob:
        coerced = [SamplerPub.coerce(p, shots or self._backend.options.shots) for p in pubs]
        if not coerced:
            raise ValueError("At least one PUB is required")

        circuits = [pub.circuit.assign_parameters(pub.parameter_values.as_array().flatten()) for pub in coerced]
        num_qubits_list = [c.num_qubits for c in circuits]
        shots_list = [pub.shots or self._backend.options.shots for pub in coerced]

        if len(circuits) > 1 and len(set(shots_list)) != 1:
            raise ValueError(
                "All PUBs must have the same shot count for multi-circuit batching. "
                "Pass a uniform `shots` kwarg to sampler.run()."
            )

        job_id = submit(
            client=self._client,
            backend=self._backend._backend_info.backend,
            circuits=circuits,
            shots=shots_list[0],
            gateset=self._backend._gateset,
            target=self._backend.target,
            session_id=self._session.session_id if self._session else None,
        )
        return IonQSamplerJob(
            job_id=job_id, client=self._client, num_qubits_list=num_qubits_list, shots_list=shots_list
        )
