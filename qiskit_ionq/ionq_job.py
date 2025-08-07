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

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Optional
import numpy as np

from qiskit import QuantumCircuit
from qiskit.providers import JobV1, jobstatus
from qiskit.providers.exceptions import JobTimeoutError
from .ionq_result import IonQResult as Result
from .helpers import decompress_metadata_string

from . import constants, exceptions

if TYPE_CHECKING:
    from . import ionq_backend
    from . import ionq_client


def map_output(data, clbits, num_qubits):
    """Map histogram according to measured bits."""

    if not clbits:
        return {}

    mapped_output = {}

    def get_bitvalue(bitstring, bit):
        if bit is not None and 0 <= bit < len(bitstring):
            return bitstring[bit]
        return "0"

    for value, probability in data.items():
        bitstring = bin(int(value))[2:].rjust(num_qubits, "0")[::-1]

        outvalue = int("".join(get_bitvalue(bitstring, bit) for bit in clbits)[::-1], 2)

        mapped_output[outvalue] = mapped_output.get(outvalue, 0) + probability

    return mapped_output


def _build_counts(
    data,
    num_qubits,
    clbits,
    shots,
    use_sampler=False,
    sampler_seed=None,
):  # pylint: disable=too-many-positional-arguments
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
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()  # normalize
        sample_counts = np.bincount(
            rand.choice(len(outcomes), shots, p=weights), minlength=len(outcomes)
        )
        sampled = dict(zip(outcomes, sample_counts))

    # Build counts and probabilities
    counts = {}
    probabilities = {}
    for key, val in output_probs.items():
        bits = bin(int(key))[2:].rjust(num_qubits, "0")
        hex_bits = hex(int(bits, 2))
        count = sampled[key] if use_sampler else round(val * shots)
        if count:  # only non-zero counts
            counts[hex_bits] = count
            probabilities[hex_bits] = val

    return counts, probabilities


class IonQJob(JobV1):
    """Representation of a Job that will run on an IonQ backend.

    It is not recommended to create Job instances directly; use
    :meth:`IonQBackend.run` and :meth:`IonQBackend.retrieve_job` instead.

    Attributes:
        circuit(:mod:`QuantumCircuit <qiskit.QuantumCircuit>`): A possibly ``None``
            Qiskit quantum circuit.
        _result(:class:`Result <qiskit.result.Result>`):
            The actual Qiskit Result of this job when done.
    """

    def __init__(
        self,
        backend: ionq_backend.IonQBackend,
        job_id: Optional[str] = None,
        client: Optional[ionq_client.IonQClient] = None,
        circuit: Optional[QuantumCircuit] = None,
        passed_args: Optional[dict] = None,
    ):  # pylint: disable=too-many-positional-arguments
        assert (
            job_id is not None or circuit is not None
        ), "Job must have a job_id or circuit"
        super().__init__(backend, job_id)
        self._client = client or backend.client
        self._result = None
        self._status = None
        self._execution_time = None
        self._metadata: dict[str, Any] = {}

        if passed_args is not None:
            self.extra_query_params = passed_args.pop("extra_query_params", {})
            self.extra_metadata = passed_args.pop("extra_metadata", {})
            self._passed_args = passed_args
        else:
            self.extra_query_params = {}
            self.extra_metadata = {}
            self._passed_args = {"shots": 1024, "sampler_seed": None}

        # Support single or list-of-circuits submissions
        if circuit is not None:
            self.circuit = circuit
            self._status = jobstatus.JobStatus.INITIALIZING
        else:  # retrieve existing job
            self.circuit = None
            self._status = jobstatus.JobStatus.INITIALIZING
            self._job_id = job_id
            self.status()

    @staticmethod
    def _first_of(mapping: dict, *keys, default=None):
        """Return the first present key in `keys` or `default`."""
        for k in keys:
            if k in mapping and mapping[k] is not None:
                return mapping[k]
        return default

    def cancel(self) -> None:
        """Cancel this job."""
        assert self._job_id is not None, "Cannot cancel a job without a job_id."
        self._client.cancel_job(self._job_id)

    def submit(self) -> None:
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

    def get_counts(self, circuit: Optional[QuantumCircuit] = None) -> dict:
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
        """Return the probabilities (for simulators).

        This is effectively a pass-through to
            :meth:`get_probabilities <qiskit_ionq.ionq_result.IonQResult.get_probabilities>`

        Args:
            circuit (str or QuantumCircuit or int or None): Optional.

        Returns:
            tuple(dict[str, float], dict[str, float]): A tuple counts, probabilities.
        """
        return self.result().get_probabilities()

    def result(
        self,
        sharpen: bool | None = None,
        extra_query_params: dict | None = None,
        **kwargs,
    ):
        """Retrieve job result data, blocking until the job is complete.

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
            IonQJobStateError: If the job was cancelled before this method fetches it.

        Returns:
            Result: A Qiskit :class:`Result <qiskit.result.Result>` representation of this job.
        """
        # Validate args
        if sharpen is not None and not isinstance(sharpen, bool):
            warnings.warn("Invalid sharpen type")

        # Wait for the job to complete.
        try:
            self.wait_for_final_state(**kwargs)
        except JobTimeoutError as ex:
            raise exceptions.IonQJobTimeoutError(
                "Timed out waiting for job to complete."
            ) from ex

        if self._status is jobstatus.JobStatus.CANCELLED:
            assert self._job_id is not None
            raise exceptions.IonQJobStateError(
                f"Cannot retrieve result for canceled job {self._job_id}"
            )

        if self._status is jobstatus.JobStatus.DONE:
            assert self._job_id is not None
            response = self._client.get_results(
                results_url=self._results_url,
                sharpen=sharpen,
                extra_query_params=extra_query_params,
            )
            self._result = self._format_result(response)

        return self._result

    def status(self, detailed: bool = False) -> jobstatus.JobStatus | dict:
        """Retrieve the status of a job.

        Args:
            detailed (bool): If True, returns a detailed status of children.

        Returns:
            JobStatus or dict: An enum value from Qiskit's
                :class:`JobStatus <qiskit.providers.JobStatus>` if detailed is False.
                A dictionary containing the detailed status of the children if detailed is True.

        Raises:
            IonQJobError: If the IonQ job status was unknown or otherwise
                unmappable to a qiskit job status.
            IonQJobFailureError: If the job fails
            IonQJobStateError: If the job was cancelled
        """
        # Return early if we have no job id yet.
        if self._job_id is None:
            return self._status

        # Return early if the job is already final.
        if self._status in jobstatus.JOB_FINAL_STATES:
            return self._children_status() if detailed else self._status

        # Otherwise, look up a status enum from the response.
        response = self._client.retrieve_job(self._job_id)
        api_response_status = response.get("status")

        try:
            status_enum = constants.APIJobStatus(api_response_status)
            mapped_status = constants.JobStatusMap[status_enum.name]
            self._status = jobstatus.JobStatus[mapped_status.value]
        except (ValueError, KeyError) as ex:
            raise exceptions.IonQJobError(
                f"Unknown or unmappable job status {api_response_status}"
            ) from ex

        if self._status in jobstatus.JOB_FINAL_STATES:
            self._save_metadata(response)

        if self._status == jobstatus.JobStatus.DONE:
            stats = response.get("stats", {})
            self._children = self._first_of(
                response, "child_job_ids", "children", default=None
            )

            # Circuit count: if we have children, prefer that length
            if self._children:
                self._num_circuits = len(self._children)
            else:
                self._num_circuits = self._first_of(stats, "circuits", default=1)

            self._num_qubits = self._first_of(stats, "qubits", default=0)
            _results_url = self._first_of(
                response, "results", "results_url", default={}
            )
            self._results_url = (
                _results_url
                if isinstance(_results_url, str)
                else _results_url.get("probabilities", {}).get("url")
            )

            # Classical-bit maps per circuit
            def _meas_map_from_header(header_dict, fallback_nq):
                """Return meas_mapped list or a default 0-based map."""
                mmap = header_dict.get("meas_mapped")
                if mmap is None or (
                    isinstance(mmap, list) and all(b is None for b in mmap)
                ):
                    return list(range(header_dict.get("n_qubits", fallback_nq)))
                return mmap

            # Classical-bit maps for every circuit
            header_list = decompress_metadata_string(
                response.get("metadata", {}).get("qiskit_header")
            )
            if not isinstance(header_list, list):
                header_list = [header_list]
            self._clbits = [
                _meas_map_from_header(h, self._num_qubits) for h in header_list
            ]

            # Ensure one map per circuit
            if len(self._clbits) == 1 and self._num_circuits > 1:
                self._clbits *= self._num_circuits

            # Prefer the API-supplied execution time
            self._execution_time = (
                self._first_of(
                    response,
                    "execution_duration_ms",
                    "execution_time",
                    default=float("inf"),
                )
                / 1000
            )

        if self._status == jobstatus.JobStatus.ERROR:
            failure = response.get("failure") or {}
            raise exceptions.IonQJobFailureError(
                f"Unable to retrieve result for job {self._job_id}. "
                f'Failure from IonQ API "{failure.get("code","")}: '
                f'{failure.get("error","")}"'
            )

        if self._status == jobstatus.JobStatus.CANCELLED:
            warnings.warn(
                f"Unable to retrieve result for job {self._job_id}. Job was cancelled"
            )

        # Propagate any warnings returned by the API
        if "warning" in response and "messages" in response["warning"]:
            for msg in response["warning"]["messages"]:
                warnings.warn(msg)

        return self._children_status() if detailed else self._status

    def _children_status(self):
        """Return a dictionary describing the status of any child jobs.

        Raises:
            IonQJobError: If the IonQ job status was unknown or otherwise
                unmappable to a qiskit job status.
            IonQJobFailureError: If the job fails
            IonQJobStateError: If the job was cancelled

        Returns:
            dict: A dictionary containing the detailed status of the children.
        """
        response = self._client.retrieve_job(self._job_id)
        child_ids = self._first_of(response, "child_job_ids", "children", [])
        child_statuses = []

        for child_id in child_ids:
            resp = self._client.retrieve_job(child_id)
            api_status = resp.get("status")

            # Map API status to JobStatus enum
            try:
                status_enum = constants.APIJobStatus(api_status)
                status_enum = constants.JobStatusMap[status_enum.name]
                qiskit_status = jobstatus.JobStatus[status_enum.value]
            except (ValueError, KeyError) as ex:
                raise exceptions.IonQJobError(
                    f"Unknown or unmappable child job status {api_status}"
                ) from ex

            child_statuses.append(qiskit_status)

        total = len(child_statuses)
        completed = child_statuses.count(jobstatus.JobStatus.DONE)
        failed = child_statuses.count(jobstatus.JobStatus.ERROR)

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "percentage_complete": completed / total if total else 0,
            "statuses": child_statuses,
        }

    def _format_result(self, data):
        """Translate IonQ result format into a Qiskit `Result` instance.

        Args:
            result (dict): A JSON body response from a REST API call.

        Returns:
            Result: A Qiskit :class:`Result <qiskit.result.Result>` representation of this job.

        Raises:
            IonQJobFailureError: If the remote job has an error status.
            IonQJobStateError: If the job was cancelled before this method fetches it.
        """
        backend = self.backend()
        backend_name = backend.name()
        backend_version = backend.configuration().backend_version
        is_ideal_sim = (
            backend_name == "ionq_simulator" and backend.options.noise_model == "ideal"
        )

        success = self._status == jobstatus.JobStatus.DONE
        metadata = self._metadata.get("metadata") or {}

        sampler_seed = (
            int(metadata.get("sampler_seed", ""))
            if metadata.get("sampler_seed", "").isdigit()
            else None
        )
        qiskit_header = decompress_metadata_string(metadata.get("qiskit_header"))
        if not isinstance(qiskit_header, list):
            qiskit_header = [qiskit_header]

        shots = (
            int(metadata.get("shots", 1024))
            if str(metadata.get("shots", "1024")).isdigit()
            else 1024
        )

        if isinstance(data, dict):
            looks_like_multi = all(
                isinstance(v, dict) and all(isinstance(p, float) for p in v.values())
                for v in data.values()
            )
            data = list(data.values()) if looks_like_multi else [data]
        elif isinstance(data, list):
            pass
        else:
            raise exceptions.IonQJobError("Unexpected result payload type")

        # pad headers if API dropped them
        while len(qiskit_header) < self._num_circuits:
            qiskit_header.append({})

        job_result = [
            {
                "data": {},
                "shots": shots,
                "header": qiskit_header[i] or {},
                "success": success,
            }
            for i in range(self._num_circuits)
        ]
        if self._status == jobstatus.JobStatus.DONE:
            for i in range(self._num_circuits):
                (counts, probabilities) = _build_counts(
                    data[i],
                    qiskit_header[i].get("n_qubits", self._num_qubits),
                    self._clbits[i],
                    shots,
                    use_sampler=is_ideal_sim,
                    sampler_seed=sampler_seed,
                )
                job_result[i]["data"] = {
                    "counts": counts,
                    "probabilities": probabilities,
                    "metadata": qiskit_header[i] or {},
                }

        # Final Qiskit Result object
        return Result.from_dict(
            {
                "results": job_result,
                "job_id": self.job_id(),
                "backend_name": backend_name,
                "backend_version": backend_version,
                "qobj_id": metadata.get("qobj_id"),
                "success": success,
                "time_taken": self._execution_time,
            }
        )

    def _save_metadata(self, response):
        """Persist metadata from the API response to this instance.

        Args:
            response (dict): A JSON body response from a REST API call.
        """
        self._metadata.update(response)


__all__ = ["IonQJob"]
