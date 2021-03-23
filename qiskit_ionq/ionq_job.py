# -*- coding: utf-8 -*-
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

"""IonQ's Job implementation.

.. NOTE::
   IonQ job status names are slightly different than those of the standard
   :class:`JobStatus <qiskit.providers.JobStatus>` enum values.

   As such, the :meth:`IonQJob.status` method on the IonQJob class attempts to
   perform a mapping between these status values for compatibility with
   :class:`BaseJob <qiskit.providers.BaseJob>`.
"""

import json

from qiskit.providers import JobV1, jobstatus
from qiskit.providers.exceptions import JobTimeoutError
from qiskit.result import Result
from .helpers import decompress_metadata_string_to_dict
from . import constants, exceptions


def _build_counts(result, retain_probabilities=False):
    """Map IonQ's ``counts`` onto qiskit's ``counts`` model.

    Args:
        result (dict): A REST API response.
        retain_probabilities (bool): Retain probabilities from the response instead
            of converting them to counts.

    Returns:
        dict[str, float]: A dict of qiskit compatible ``counts``.

    Raises:
        IonQJobError: In the event that ``result`` has missing or invalid job properties.
    """
    # Short circuit when we don't have all the information we need.
    if not result:
        raise exceptions.IonQJobError("Cannot remap counts without an API response!")

    # Check for required result dict keys:
    if "qubits" not in result:
        raise exceptions.IonQJobError("Cannot remap counts without qubits!")
    if "metadata" not in result:
        raise exceptions.IonQJobError("Cannot remap counts without metadata!")
    if "data" not in result:
        raise exceptions.IonQJobError("Cannot remap counts without result data!")

    # Pull metadata, histogram, and num_qubits to perform the mapping.
    metadata = result["metadata"]
    num_qubits = result["qubits"]

    # Get shot count.
    shots = metadata.get("shots")
    shots = int(shots) if shots is not None else 1024  # We do this in cas>e shots was 0.

    # Grab the mapped output from response.
    output_probs = result["data"].get("registers", {}).get("meas_mapped", {})

    # Build counts.
    counts = {}
    for key, val in output_probs.items():
        bits = bin(int(key))[2:].rjust(num_qubits, "0")
        hex_bits = hex(int(bits, 2))
        count = val if retain_probabilities else round(val * shots)
        counts[hex_bits] = count
    return counts


class IonQJob(JobV1):
    """Representation of a Job that will run on an IonQ backend.

    .. IMPORTANT::
       IonQ backends do not support multi-experiment jobs.  Attempting to
       submit a multi-experiment job will raise an exception.

    It is not recommended to create Job instances directly, but rather use the
    :meth:`run <IonQBackend.run>` and :meth:`retrieve_job <IonQBackend.retrieve_job>`
    methods on sub-classe instances of IonQBackend to create and retrieve jobs
    (both methods return a job instance).

    Attributes:
        circuit(:mod:`QuantumCircuit <qiskit.QuantumCircuit>`): A possibly ``None``
            Qiskit quantum circuit.
        _result(:class:`Result <qiskit.result.Result>`):
            The actual Qiskit Result of this job.
            This attribute is only populated when :meth:`status` is called and
            the job has reached a one of the status values in ``JOB_FINAL_STATES``
            from ``qiskit.providers.jobstatus``
    """

    def __init__(self, backend, job_id, client=None, circuit=None, passed_args=None):
        super().__init__(backend, job_id)
        self._client = client or backend.client
        self._passed_args = passed_args or {"shots": 1024}
        self._result = None
        self._status = None

        if circuit is not None:
            self.circuit = circuit
            self._status = jobstatus.JobStatus.INITIALIZING
        else:  # retrieve existing job
            self.circuit = None
            self._status = jobstatus.JobStatus.INITIALIZING
            self._job_id = job_id
            self.status()

    def cancel(self):
        """Cancel this job."""
        self._client.cancel_job(self._job_id)

    def submit(self):
        """Submit a job to the IonQ API.

        Raises:
            IonQJobError: If this instance's :attr:`qobj` was `None`.
        """
        if self.circuit is None:
            raise exceptions.IonQJobError(
                "Cannot submit a job without a circuit. "
                "Please create a job with a circuit and try again."
            )

        response = self._client.submit_job(job=self)
        self._job_id = response["id"]

    def get_counts(self, circuit=None):
        """Return the counts for the job.

        .. ATTENTION::

            Result counts for jobs processed by
            :class:`IonQSimulatorBackend <qiskit_ionq.ionq_backend.IonQSimulatorBackend>`
            are expressed as probabilites, rather than a multiple of shots.

        Args:
             circuit (str or QuantumCircuit or int or None): Optional. The index of the experiment.

        Returns:
            dict: A dictionary of counts.
        """
        return self.result().get_counts(circuit)

    def result(self):
        """Retrieve job result data.

        .. NOTE::
           :attr:`_result` is populated by :meth:`status`, when the job
           status has reached a "final" state.

        This method calls the
        :meth:`wait_for_final_state <qiskit.providers.BaseJob.wait_for_final_state>`
        method to poll for a completed job.

        Raises:
            IonQJobTimeoutError: If after the default wait period in
                :meth:`wait_for_final_state <qiskit.providers.BaseJob.wait_for_final_state>`
                elapses and the job has not reached a "final" state.
            IonQJobError: If the job has reached a final state but
                the job itself was never converted to a
                :class:`Result <qiskit.result.Result>`.

        Returns:
            Result: A Qiskit :class:`Result <qiskit.result.Result>` representation of this job.
        """
        # Short-circuit if we have already cached the result for this job.
        if self._result is not None:
            return self._result

        # Wait for the job to complete.
        try:
            self.wait_for_final_state()
        except JobTimeoutError as ex:
            raise exceptions.IonQJobTimeoutError("Timed out waiting for job to complete.") from ex

        return self._result

    def status(self):
        """Retrieve the status of a job

        This will also populate :attr:`_result` with a :class:`Result <qiskit.result.Result>`
        object, if the job's status has reached a "final" state.

        Raises:
            IonQJobError: If the IonQ job status was unknown or otherwise
                unmappable to a qiskit job status.

        Returns:
            JobStatus: An enum value from Qiskit's :class:`JobStatus <qiskit.providers.JobStatus>`.
        """
        # Return early if we have no job id yet.
        if self._job_id is None:
            return self._status

        # Return early if the job is already done.
        if self._status in jobstatus.JOB_FINAL_STATES:
            return self._status

        # Otherwise, look up a status enum from the response.
        response = self._client.retrieve_job(self._job_id)
        api_response_status = response["status"]
        try:
            status_enum = constants.APIJobStatus(api_response_status)
        except ValueError as ex:
            raise exceptions.IonQJobError(f"Unknown job status {api_response_status}") from ex

        # Map it to a qiskit JobStatus key
        try:
            status_enum = constants.JobStatusMap[status_enum.name]
        except ValueError as ex:
            raise exceptions.IonQJobError(
                f"Job status {status_enum} has no qiskit status mapping!"
            ) from ex

        # Get a qiskit status enum.
        try:
            self._status = jobstatus.JobStatus[status_enum.value]
        except KeyError as ex:
            raise exceptions.IonQJobError(f"Qiskit has no JobStatus named '{status_enum}'") from ex

        # if done, also put the result on the job obj
        # so we don't have to make an API call again if user wants results
        if self._status in jobstatus.JOB_FINAL_STATES:
            self._result = self._format_result(response)

        return self._status

    def _format_result(self, result):
        """Translate IonQ's result format into a qiskit Result instance.

        TODO: If result is (failure, cancelled), this method may fail.

        Args:
            result (dict): A JSON body response from a REST API call.

        Returns:
            Result: A Qiskit :class:`Result <qiskit.result.Result>` representation of this job.
        """

        # Different backends can have differing result data:
        backend = self.backend()
        backend_name = backend.name()
        backend_version = backend.configuration().backend_version
        is_simulator = backend_name == "ionq_simulator"

        # Format the inner result payload.
        success = self._status == jobstatus.JobStatus.DONE
        metadata = result.get("metadata") or {}
        qiskit_header = decompress_metadata_string_to_dict(metadata.get("qiskit_header", None))
        job_result = {
            "data": {},
            "shots": metadata.get("shots", 1),
            "header": qiskit_header or {},
            "success": success,
        }
        if self._status == jobstatus.JobStatus.DONE:
            job_result["data"] = {
                "counts": _build_counts(result, retain_probabilities=is_simulator)
            }
        elif self._status == jobstatus.JobStatus.ERROR:
            failure = result.get("failure") or {}
            job_result["status"] = failure.get("error")
        elif self._status == jobstatus.JobStatus.CANCELLED:
            job_result["status"] = "Job was cancelled"

        # Create a qiskit result to express the IonQ job result data.
        backend = self.backend()
        return Result.from_dict(
            {
                "results": [job_result],
                "job_id": self.job_id(),
                "backend_name": backend_name,
                "backend_version": backend_version,
                "qobj_id": metadata.get("qobj_id"),
                "success": success,
                "time_taken": result.get("execution_time") / 1000,
            }
        )


__all__ = ["IonQJob"]
