"""IonQ job wrapper for Qiskit."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ionq_core import wait_for_job
from ionq_core.api.default import cancel_job, get_job
from qiskit.providers import JobStatus, JobV1
from qiskit.result import Result

from ._result import to_qiskit_result

if TYPE_CHECKING:
    from ionq_core.client import AuthenticatedClient
    from qiskit.providers import BackendV2

_STATUS_MAP = {
    "submitted": JobStatus.QUEUED,
    "ready": JobStatus.QUEUED,
    "started": JobStatus.RUNNING,
    "completed": JobStatus.DONE,
    "failed": JobStatus.ERROR,
    "canceled": JobStatus.CANCELLED,
}


class IonQJob(JobV1):
    def __init__(
        self,
        backend: BackendV2 | None,
        job_id: str,
        client: AuthenticatedClient,
        num_qubits: int,
        shots: int,
    ):
        super().__init__(backend, job_id)
        self._client = client
        self._num_qubits = num_qubits
        self._shots = shots
        self._result: Result | None = None

    def submit(self):
        raise NotImplementedError("IonQJob is submitted at creation time via backend.run()")

    def status(self) -> JobStatus:
        resp = get_job.sync(uuid=self.job_id(), client=self._client)
        if resp is None:
            raise RuntimeError("Failed to fetch job status")
        mapped = _STATUS_MAP.get(resp.status)
        if mapped is None:
            raise ValueError(f"Unknown IonQ job status: {resp.status!r}")
        return mapped

    def result(self, timeout: float = 300.0) -> Result:
        if self._result is not None:
            return self._result
        wait_for_job(self._client, self.job_id(), timeout=timeout)
        self._result = to_qiskit_result(self._client, self.job_id(), self._num_qubits, self._shots)
        return self._result

    def cancel(self):
        cancel_job.sync(uuid=self.job_id(), client=self._client)
