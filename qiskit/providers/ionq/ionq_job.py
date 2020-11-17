import json
import time
from datetime import datetime, timezone

from qiskit.providers import BaseJob
from qiskit.qobj import validate_qobj_against_schema
from qiskit.result import Result

from .constants import JOB_FINAL_STATES, APIJobStatus, JobStatus
from .exceptions import *


class IonQJob(BaseJob):
    """Representation of a Job that will run on an IonQ backend.

    IonQ backends do not support multi-experiment jobs. Attempting to submit a multi-
    experiment job will raise an exception.

    It's recommended that you don't create Job instances directly, but rather use
    the run() and retrieve() methods on the IonQ backends to create and retrieve
    jobs. These will both return a job instance.
    """

    def __init__(self, backend, job_id, client, qobj=None):
        super().__init__(backend, job_id)
        self._client = client

        self._result = None
        self._status = None

        if qobj is not None:
            if job_id:
                raise IonQJobError(
                    "incompatible options. please only provide a Job ID for a previously-created job. for a new job, an ID will be created automatically"
                )
            validate_qobj_against_schema(qobj)
            qobj.header.backend_name = backend.name()
            self.qobj = qobj
            self._status = JobStatus.INITIALIZING

        else:  # retrieve existing job
            self.qobj = None
            self._status = JobStatus.INITIALIZING
            self._job_id = job_id
            self.status()

    def submit(self):
        """submit the job to be run on the backend"""
        if not self.qobj:
            raise IonQJobError("no qobj found. please provide instructions to run.")

        response = self._client.submit_job(job=self)
        self._job_id = response["id"]

    def result(self):
        """retrieve job results"""
        if self._result is not None:
            return self._result

        response = self._client.retrieve_job(self._job_id)

        self._status = JobStatus[APIJobStatus(response["status"]).name]
        if not self._status in JOB_FINAL_STATES:
            raise IonQJobError(
                "Job will only have results when its status is final (COMPLETED, CANCELED, FAILED).",
                "current status is {}".format(self._status),
            )
        self._result = self._format_result(response)

        if not self._result:
            raise IonQJobError("Something went wrong. No result was generated.")

        return self._result

    def cancel(self):
        """cancel job"""
        return self._client.cancel_job(self._job_id)

    def status(self):
        """retrieve the status of a job"""
        if self._job_id is None or self._status in JOB_FINAL_STATES:
            return self._status

        response = self._client.retrieve_job(self._job_id)
        if not response["status"]:
            raise IonQBackendError()
        self._status = JobStatus[APIJobStatus(response["status"]).name]

        # if done, also put the result on the job obj
        # so we don't have to make an API call again if user wants results
        if self._status == JobStatus.COMPLETED:
            self._result = self._format_result(response)

        return self._status

    def _format_result(self, result):
        # TODO: if result is failure, cancelled, this might have a bad time
        """map IonQ's result format onto a qiskit Result instance"""
        results = [
            {
                "success": self._status == JobStatus.COMPLETED,
                "shots": result.get("metadata", {}).get("shots", 1),
                "data": {"counts": self._format_counts(result)},
                "header": json.loads(result.get("metadata", {}).get("header", "{}")),
            }
        ]
        return Result.from_dict(
            {
                "results": results,
                "backend_name": self.backend().name(),
                "backend_version": self._backend._configuration.backend_version,
                "qobj_id": result.get("metadata", {}).get("qobj_id", None),
                "success": self._status == JobStatus.COMPLETED,
                "job_id": self._job_id,
            }
        )

    def _format_counts(self, result):
        """map IonQ's counts onto qiskit's counts model"""
        counts = {}
        metadata = result.get("metadata") or {}
        shots = int(metadata.get("shots", 1024))
        histogram = (result.get("data") or {}).get("histogram") or {}
        output_map = json.loads(metadata.get("output_map") or {})
        output_length = int(metadata.get("output_length", result["qubits"]))
        for bitstring in histogram:
            string_as_hex = self._remap_bitstring(bitstring, output_map, output_length)
            counts[string_as_hex] = round(histogram[bitstring] * shots)
        return counts

    def _remap_bitstring(self, bitstring, output_map, output_length):
        """IonQ's API does not allow ad-hoc remapping of classical to quantum registers,
        instead always returning quantum[i] as classical[i] in the return bitstring.
        This uses an output map created at submission from the measure instructions in the
        instruction list to map to the expected classical bitstring.
        """
        bin_output = list("0" * output_length)
        bin_input = list(bin(int(bitstring))[2:].rjust(output_length, "0"))
        bin_input.reverse()

        for quantum, classical in output_map.items():
            bin_output[int(classical)] = bin_input[int(quantum)]
        bin_output.reverse()
        return hex(int("".join(bin_output), 2))

    def job_id(self):
        return self._job_id
