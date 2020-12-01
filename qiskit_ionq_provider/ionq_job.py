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

from qiskit.providers import BaseJob, jobstatus
from qiskit.providers.exceptions import JobTimeoutError
from qiskit.result import Result

from . import constants, exceptions, ionq_client


def _remap_bitstring(bitstring, output_map, output_length):
    """IonQ's API does not allow ad-hoc remapping of classical to quantum
    registers, instead always returning quantum[i] as classical[i] in the return
    bitstring.

    This function uses an output map created at submission from the measure
    instructions in the instruction list to map to the expected classical bitstring.

    Args:
        bitstring (str): A bitstring to remap.
        output_map (dict): An output mapping to from quantum <-> classical.
        output_length (int): Output length.

    Returns:
        str: A hexadecimal bit string.
    """
    bin_output = list("0" * output_length)
    bin_input = list(bin(int(bitstring))[2:].rjust(output_length, "0"))
    bin_input.reverse()
    for quantum, classical in output_map.items():
        bin_output[int(classical)] = bin_input[int(quantum)]
    bin_output.reverse()
    return hex(int("".join(bin_output), 2))


def _format_counts(result):
    """Map IonQ's ``counts`` onto qiskit's ``counts`` model.

    Args:
        result (dict): A REST API response.

    Returns:
        dict[str, float]: A dict of qiskit compatible ``counts``.
    """
    # Short circuit with no results.
    if not result:
        return {}
    metadata = result.get("metadata") or {}
    num_qubits = result["qubits"]
    shots = int(metadata.get("shots", 1024))
    histogram = (result.get("data") or {}).get("histogram") or {}
    output_map = json.loads(metadata.get("output_map") or {})
    output_length = len(output_map) if output_map else num_qubits
    offset = num_qubits - 1
    counts = {}
    for key, val in histogram.items():
        bits = bin(int(key))[2:].rjust(num_qubits, "0")
        red_bits = ['0']*output_length
        for qbit, cbit in output_map.items():
            red_bits[cbit] = str(bits[offset-int(qbit)])

        red_bitstring = "".join(red_bits)[::-1]
        if red_bitstring in counts:
            counts[red_bitstring] += round(val * shots)
        else:
            counts[red_bitstring] = round(val * shots)
    return counts


class IonQJob(BaseJob):
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

    def __init__(self, backend, job_id, client=None, circuit=None,
                 passed_args=None):
        super().__init__(backend, job_id)
        self._client = client or ionq_client.IonQClient(backend.create_client())
        self._result = None
        self._status = None
        self._passed_args = passed_args if passed_args else {'shots': 1024}

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
        if not self.circuit:
            raise exceptions.IonQJobError(
                "No `qobj` found! Please provide instructions and try again."
            )

        response = self._client.submit_job(job=self)
        self._job_id = response["id"]

    def get_counts(self, circuit=None):
        """Return the counts for the job.

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
            raise exceptions.IonQJobTimeoutError(
                "Timed out waiting for job to complete."
            ) from ex

        if not self._result:
            raise exceptions.IonQJobError(
                "Job reached final state but no result was stored."
            )

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
        if self._job_id is None or self._status in jobstatus.JOB_FINAL_STATES:
            return self._status

        response = self._client.retrieve_job(self._job_id)
        if not response["status"]:
            raise exceptions.IonQJobError("Could not determine job status!")

        # Look up a status enum from the response.
        api_response_status = response["status"]
        try:
            status_enum = constants.APIJobStatus(api_response_status)
        except ValueError as ex:
            raise exceptions.IonQJobError(
                f"Unknown job status {api_response_status}"
            ) from ex

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
            raise exceptions.IonQJobError(
                f"Qiskit has no JobStatus named '{status_enum}'"
            ) from ex

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
        metadata = result.get("metadata") or {}
        results = [
            {
                "success": self._status == jobstatus.JobStatus.DONE,
                "shots": metadata.get("shots", 1),
                "data": {"counts": _format_counts(result)},
                "header": json.loads(metadata.get("header") or "{}"),
            }
        ]
        return Result.from_dict(
            {
                "results": results,
                "backend_name": self.backend().name(),
                "backend_version": self._backend._configuration.backend_version,
                "qobj_id": metadata.get("qobj_id"),
                "success": self._status == jobstatus.JobStatus.DONE,
                "job_id": self._job_id,
            }
        )


__all__ = ["IonQJob"]
