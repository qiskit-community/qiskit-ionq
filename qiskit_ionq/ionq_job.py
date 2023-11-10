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

import warnings
import numpy as np

from qiskit.providers import JobV1, jobstatus
from qiskit.providers.exceptions import JobTimeoutError
from .ionq_result import IonQResult as Result
from .helpers import decompress_metadata_string_to_dict


from . import constants, exceptions


def map_output(data, clbits, num_qubits):
    """Map histogram according to measured bits"""

    if not clbits:
        return {}

    mapped_output = {}

    def get_bitvalue(bitstring, bit):
        if bit is not None and 0 <= bit < len(bitstring):
            return bitstring[bit]
        return "0"

    for value, probability in data.items():
        bitstring = bin(int(value))[2:].rjust(num_qubits, "0")[::-1]

        outvalue = int(
            "".join([get_bitvalue(bitstring, bit) for bit in clbits])[::-1], 2
        )

        mapped_output[outvalue] = mapped_output.get(outvalue, 0) + probability

    return mapped_output


def _build_counts(
    data, num_qubits, clbits, shots, use_sampler=False, sampler_seed=None
):
    """Map IonQ's ``counts`` onto qiskit's ``counts`` model.

    .. NOTE:: For simulator jobs, this method builds counts using a randomly
        generated sampling of the probabilities returned from the API. Because
        this is a random process, rebuilding the results object (by e.g.
        restarting the kernel and getting the job again) without providing a
        sampler_seed in the run method may result in slightly different counts.

    Args:
        data (dict): histogram as returned by the API.
        num_qubits (int): number of qubits
        clbits (List[int]): array of classical bits for measurements
        shots (int): number of shots
        use_sampler (bool): for counts generation, whether to use
            simple shots * probabilities (for qpu) or a sampler (for simulator)
        sampler_seed (int): ability to provide a seed for the randomness in the
            sampler for repeatable results. passed as
            `np.random.RandomState(seed)`. If none, `np.random` is used

    Returns:
        tuple(dict[str, float], dict[str, float]): A tuple (counts, probabilities),
            respectively a dict of qiskit compatible ``counts`` and a dict of
            the job's probabilities as a``Counts`` object, mostly relevant for
            simulator work.

    Raises:
        IonQJobError: In the event that ``result`` has missing or invalid job
            properties.
    """
    # Short circuit when we don't have all the information we need.
    if not data:
        raise exceptions.IonQJobError("Cannot remap counts without data!")

    # Grab the mapped output from response.
    output_probs = map_output(data, clbits, num_qubits)

    sampled = {}
    if use_sampler:
        rand = np.random.RandomState(sampler_seed)
        outcomes, weights = zip(*output_probs.items())
        weights = np.array(weights).astype(float)
        # just in case the sum isn't exactly 1 — sometimes the API returns
        #  e.g. 0.499999 due to floating point error
        weights /= weights.sum()
        outcomes = np.array(outcomes)

        rand_values = rand.choice(outcomes, shots, p=weights)

        sampled.update(
            {key: np.count_nonzero(rand_values == key) for key in output_probs}
        )

    # Build counts.
    counts = {}
    for key, val in output_probs.items():
        bits = bin(int(key))[2:].rjust(num_qubits, "0")
        hex_bits = hex(int(bits, 2))
        count = sampled[key] if use_sampler else round(val * shots)
        counts[hex_bits] = count
    # build probs
    probabilities = {}
    for key, val in output_probs.items():
        bits = bin(int(key))[2:].rjust(num_qubits, "0")
        hex_bits = hex(int(bits, 2))
        probabilities[hex_bits] = val

    return counts, probabilities


class IonQJob(JobV1):
    """Representation of a Job that will run on an IonQ backend.

    .. IMPORTANT::
       IonQ backends do not support multi-experiment jobs.  Attempting to
       submit a multi-experiment job will raise an exception.

    It is not recommended to create Job instances directly, but rather use the
    :meth:`run <IonQBackend.run>` and :meth:`retrieve_job <IonQBackend.retrieve_job>`
    methods on sub-class instances of IonQBackend to create and retrieve jobs
    (both methods return a job instance).

    Attributes:
        circuit(:mod:`QuantumCircuit <qiskit.QuantumCircuit>`): A possibly ``None``
            Qiskit quantum circuit.
        _result(:class:`Result <qiskit.result.Result>`):
            The actual Qiskit Result of this job when done.
    """

    def __init__(
        self,
        backend,
        job_id,
        client=None,
        circuit=None,
        passed_args=None,
    ):
        super().__init__(backend, job_id)
        self._client = client or backend.client
        self._result = None
        self._status = None
        self._execution_time = None
        self._metadata = {}

        if passed_args is not None:
            self.extra_query_params = passed_args.pop("extra_query_params", {})
            self.extra_metadata = {
                **passed_args.pop("extra_metadata", {}),
                **(circuit.metadata or {}),
            }
            self._passed_args = passed_args
        else:
            self.extra_query_params = {}
            self.extra_metadata = {}
            self._passed_args = {"shots": 1024, "sampler_seed": None}

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
            are returned from the API as probabilities, and are converted to counts via
            simple statistical sampling that occurs on the cient side.

            To obtain the true probabilities, use the get_probabilties() method instead.

        Args:
            circuit (str or QuantumCircuit or int or None): Optional.
                The index of the experiment.

        Returns:
            dict: A dictionary of counts.
        """
        return self.result().get_counts(circuit)

    def get_probabilities(self, circuit=None):  # pylint: disable=unused-argument
        """
        Return the probabilities for the job.

        This is effectively a pass-through to
            :meth:`get_probabilities <qiskit_ionq.ionq_result.IonQResult.get_probabilities>`

        Args:
            circuit (str or QuantumCircuit or int or None): Optional.

        Returns:
            tuple(dict[str, float], dict[str, float]): A tuple counts, probabilities.
        """
        return self.result().get_probabilities()

    def result(self, sharpen: bool = None, extra_query_params: dict = None, **kwargs):
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
        # TODO: cache results by sharpen

        if sharpen is not None and not isinstance(sharpen, bool):
            warnings.warn("Invalid sharpen type")

        # Wait for the job to complete.
        try:
            self.wait_for_final_state(**kwargs)
        except JobTimeoutError as ex:
            raise exceptions.IonQJobTimeoutError(
                "Timed out waiting for job to complete."
            ) from ex

        if self._status is jobstatus.JobStatus.DONE:
            response = self._client.get_results(
                job_id=self._job_id,
                sharpen=sharpen,
                extra_query_params=extra_query_params,
            )
            self._result = self._format_result(response)

        return self._result

    def status(self):
        """Retrieve the status of a job

        Raises:
            IonQJobError: If the IonQ job status was unknown or otherwise
                unmappable to a qiskit job status.
            IonQJobFailureError: If the job fails
            IonQJobStateError: If the job was cancelled

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

        if self._status in jobstatus.JOB_FINAL_STATES:
            self._metadata = response.get("metadata") or {}

        if self._status == jobstatus.JobStatus.DONE:
            self._num_qubits = response.get("qubits")
            default_map = list(range(self._num_qubits))
            self._clbits = (response.get("registers") or {}).get(
                "meas_mapped", default_map
            )
            self._execution_time = response.get("execution_time") / 1000

        if self._status == jobstatus.JobStatus.ERROR:
            failure = response.get("failure") or {}
            failure_type = failure.get("code", "")
            failure_message = failure.get("error", "")
            error_message = (
                f"Unable to retreive result for job {self.job_id()}. "
                f'Failure from IonQ API "{failure_type}: {failure_message}"'
            )
            raise exceptions.IonQJobFailureError(error_message)

        if self._status == jobstatus.JobStatus.CANCELLED:
            error_message = (
                f'Unable to retreive result for job {self.job_id()}. Job was cancelled"'
            )
            raise exceptions.IonQJobStateError(error_message)

        if "warning" in response and "messages" in response["warning"]:
            for warning in response["warning"]["messages"]:
                warnings.warn(warning)

        return self._status

    def _format_result(self, data):
        """Translate IonQ's result format into a qiskit Result instance.

        TODO: If result is (failure, cancelled), this method may fail.

        Args:
            result (dict): A JSON body response from a REST API call.

        Returns:
            Result: A Qiskit :class:`Result <qiskit.result.Result>` representation of this job.

        Raises:
            IonQJobFailureError: If the remote job has an error status.
            IonQJobStateError: If the job was cancelled before this method fetches it.
        """

        # Different backends can have differing result data:
        backend = self.backend()
        backend_name = backend.name()
        backend_version = backend.configuration().backend_version
        is_ideal_simulator = (
            backend_name == "ionq_simulator" and backend.options.noise_model == "ideal"
        )

        # Format the inner result payload.
        success = self._status == jobstatus.JobStatus.DONE
        metadata = self._metadata
        sampler_seed = (
            int(metadata.get("sampler_seed", ""))
            if metadata.get("sampler_seed", "").isdigit()
            else None
        )
        qiskit_header = decompress_metadata_string_to_dict(
            metadata.get("qiskit_header", None)
        )

        shots = int(metadata.get("shots") if metadata.get("shots").isdigit() else 1024)
        job_result = {
            "data": {},
            "shots": shots,
            "header": qiskit_header or {},
            "success": success,
        }
        if self._status == jobstatus.JobStatus.DONE:
            (counts, probabilities) = _build_counts(
                data,
                self._num_qubits,
                self._clbits,
                shots,
                use_sampler=is_ideal_simulator,
                sampler_seed=sampler_seed,
            )
            job_result["data"] = {
                "counts": counts,
                "probabilities": probabilities,
                # Qiskit/experiments relies on this being present in this location in the
                # ExperimentData class.
                "metadata": qiskit_header or {},
            }

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
                "time_taken": self._execution_time,
            }
        )


__all__ = ["IonQJob"]
