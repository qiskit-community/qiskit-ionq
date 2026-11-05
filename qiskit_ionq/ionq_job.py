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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, NamedTuple
import numpy as np

from qiskit import QuantumCircuit
from qiskit.providers import JobV1, jobstatus
from qiskit.providers.exceptions import JobTimeoutError
from .ionq_result import IonQResult as Result
from .helpers import decompress_metadata_string, normalize
from .exceptions import IonQBackendError

from . import constants, exceptions

if TYPE_CHECKING:  # pragma: no cover
    from . import ionq_backend
    from . import ionq_client


# Batch results take a request per child. Cap how many run at once so a big
# batch doesn't send them all in one burst; the client retries 429s with
# backoff, so the exact value only affects latency.
_MAX_CONCURRENT_REQUESTS = 8


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


def _sample_counts(
    probabilities: dict[int, float], shots: int, sampler_seed: int | None
) -> dict[int, int]:
    """Sample shot counts from a probability distribution."""
    rand = np.random.RandomState(sampler_seed)
    outcomes, weights = zip(*probabilities.items())
    sample_counts = np.bincount(
        rand.choice(len(outcomes), shots, p=normalize(weights)),
        minlength=len(outcomes),
    )
    return {outcome: int(count) for outcome, count in zip(outcomes, sample_counts)}


def _build_counts(  # pylint: disable=too-many-positional-arguments
    data,
    num_qubits: int,
    clbits: list[int],
    shots: int,
    use_sampler: bool = False,
    sampler_seed: int | None = None,
    data_is_histogram: bool = False,
) -> tuple[dict[str, int], dict[str, float]]:
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
        data_is_histogram (bool): whether ``data`` contains integer histogram
            counts instead of probabilities.

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
    mapped_output = map_output(data, clbits, num_qubits)

    if data_is_histogram and sum(mapped_output.values()) <= 0:
        raise exceptions.IonQJobError("Cannot normalize an empty histogram")

    sampled_counts = {}
    if use_sampler:
        sampled_counts = _sample_counts(mapped_output, shots, sampler_seed)

    # Build counts and probabilities
    counts = {}
    probabilities = {}
    distribution_total = sum(mapped_output.values())
    for key_int, value in mapped_output.items():
        bitstr = bin(int(key_int))[2:].rjust(
            len(clbits) if clbits else num_qubits, "0"
        )  # e.g. '101'
        if data_is_histogram:
            count = value
            prob = float(value / distribution_total)
        else:
            prob = value
            count = sampled_counts.get(key_int, round(prob * shots))
        if count:  # ignore zero bins
            counts[bitstr] = int(count)
            probabilities[bitstr] = float(prob)

    return counts, probabilities


def _wire_order_to_decimal(bits: str | list[int]) -> int:
    """Read wire-order bits (qubit 0 first) as an integer whose lowest bit is qubit 0."""
    return int("".join(str(bit) for bit in bits)[::-1] or "0", 2)


def _decode_distribution_artifact(
    distribution: dict[str, Any],
) -> dict[str, int | float]:
    """Re-key a probabilities or histogram artifact by decimal outcome.

    Artifacts key outcomes by wire-order bitstring (qubit 0 first), while
    ``IonQJob._result_from_distributions`` takes decimal keys with qubit 0
    as the lowest bit, as the results endpoint returns for older jobs.
    """
    decimal_distribution: dict[str, int | float] = {}
    for bitstring, value in distribution["registers"]["output_all"].items():
        decimal = str(_wire_order_to_decimal(bitstring))
        decimal_distribution[decimal] = decimal_distribution.get(decimal, 0) + value

    return decimal_distribution


def _build_memory(
    raw_shots: list[int | str],
    n_qubits: int,
    clbits: list[int] | None,
    memory_slots: int | None = None,
) -> list[str]:
    """Convert IonQ shot outcomes into Qiskit memory bitstrings.

    The IonQ v0.4 ``CircuitJobResult.shots.url`` endpoint serves an ordered
    list of decimal-encoded outcome integers (same convention as histogram
    keys); this function applies the ``clbits`` mapping and returns one
    MSB-left bitstring per shot.
    """
    if clbits is None:
        clbits = list(range(n_qubits))
    out_width = (
        int(memory_slots) if memory_slots is not None else (len(clbits) or n_qubits)
    )

    def remap_raw_shot(val: int | str) -> str:
        raw = bin(int(val))[2:].rjust(n_qubits, "0")[::-1]
        mapped = "".join(
            raw[b] if b is not None and 0 <= b < n_qubits else "0" for b in clbits
        )[::-1]
        return mapped.rjust(out_width, "0")

    return [remap_raw_shot(s) for s in raw_shots]


def _is_leaked(shot: dict[str, Any]) -> bool:
    """Whether a v2 shot leaked.

    Per the API, any non-empty ``leakage_bits`` marks a leaked shot, even all
    zeros; a missing or empty list is clean.
    """
    return bool(shot.get("leakage_bits"))


class _Shots(NamedTuple):
    """One circuit's shots as legacy decimal outcomes, plus per-shot leak flags.

    ``leaked`` is ``None`` for legacy shots, which carry no flags.
    """

    outcomes: list[int | str]
    leaked: list[bool] | None


@dataclass(frozen=True)
class _ResultLocations:
    """One circuit's result locations, parsed once from its job response.

    Jobs that predate v2 result artifacts have no artifact IDs and are read from
    their legacy result URLs instead.

    Attributes:
        job_id: The job that owns the artifacts (a child, in a multi-circuit job).
        probabilities_id: The default distribution
            (``ionq.result.probabilities.json.v2``).
        shots_id: Per-shot outcomes (``ionq.result.shots.json.v2``).
        legacy_shots_url: Per-shot outcomes for jobs without ``shots_id``.
        aggregation_ids: Each aggregation method's histogram
            (``ionq.result.histogram.json.v2``), on debiased jobs.
    """

    job_id: str
    probabilities_id: str | None
    shots_id: str | None
    legacy_shots_url: str | None
    aggregation_ids: dict[str, str]

    @classmethod
    def from_response(cls, response: dict[str, Any]) -> _ResultLocations:
        """Parse the result locations out of a job response."""
        results = response.get("results") or {}
        probabilities = results.get(constants.ResultFormat.PROBABILITIES_V2)
        shots = results.get(constants.ResultFormat.SHOTS_V2)
        error_mitigation = (response.get("output") or {}).get("error_mitigation")
        aggregations = (error_mitigation or {}).get("aggregations") or {}
        return cls(
            job_id=response["id"],
            probabilities_id=None if probabilities is None else probabilities["id"],
            shots_id=None if shots is None else shots["id"],
            legacy_shots_url=results["shots"]["url"] if "shots" in results else None,
            aggregation_ids={
                method: descriptor["id"] for method, descriptor in aggregations.items()
            },
        )

    def distribution_id(self, aggregation: str | None) -> str | None:
        """Artifact ID of ``aggregation``'s histogram, or of the default if ``None``."""
        if aggregation is None:
            return self.probabilities_id
        return self.aggregation_ids.get(aggregation)


@dataclass(frozen=True)
class _UnretrievedChild:
    """A child of a multi-circuit job that could not be retrieved.

    The batch's default distributions then come from the parent's legacy
    endpoint and this circuit's memory is ``None``. A requested aggregation
    re-raises ``error``, since the legacy endpoint ignores ``aggregation``.

    Attributes:
        job_id: The child job.
        error: Why retrieving it failed.
    """

    job_id: str
    error: exceptions.IonQAPIError


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
        job_id: str | None = None,
        client: ionq_client.IonQClient | None = None,
        circuit: QuantumCircuit | None = None,
        passed_args: dict | None = None,
    ):  # pylint: disable=too-many-positional-arguments
        assert (
            job_id is not None or circuit is not None
        ), "Job must have a job_id or circuit"
        super().__init__(
            backend=backend, job_id=job_id if job_id else ""
        )  # TODO improve handling of None job_id
        self._client = client or backend.client
        self._result = None
        self._status = None
        self._execution_time = None
        self._dry_run: bool = False
        self._results_urls: dict[str, str | None] = {}
        # Set in status() for ionq.qasm3.v1 (mid-circuit measurement) jobs.
        self._is_qasm3: bool = False
        self._metadata: dict[str, Any] = {}

        if passed_args is not None:
            self.extra_query_params = passed_args.pop("extra_query_params", {})
            self.extra_metadata = passed_args.pop("extra_metadata", {})
            self.memory = passed_args.pop("memory", False)
            self._dry_run = bool(passed_args.get("dry_run", False))
            self._passed_args = passed_args
        else:
            self.extra_query_params = {}
            self.extra_metadata = {}
            self.memory = False
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

    @property
    def dry_run(self) -> bool:
        """Whether this job was submitted with ``dry_run=True``.

        Dry-run jobs are compiled by the IonQ Cloud compiler-as-a-service but
        not executed - they produce no measurement results. Use
        :meth:`compiled_circuit` to retrieve the compiled circuit instead of
        :meth:`result`.
        """
        return self._dry_run

    @staticmethod
    def _resolve_compiled_format(lang: str, available: dict) -> str | None:
        """Map ``lang`` (exact key or short name like ``"native"``) to an
        available format key with an artifact ``id``, else ``None``.
        """

        def has_id(key: str) -> bool:
            entry = available.get(key)
            return isinstance(entry, dict) and bool(entry.get("id"))

        if has_id(lang):
            return lang
        return next(
            (k for k in available if has_id(k) and k.split(".")[1:2] == [lang]),
            None,
        )

    def compiled_circuit(self, lang: str = "native") -> dict | list | str | bytes:
        """Fetch the server-compiled circuit, parsed from its published artifact.

        ``lang`` is a short name -- ``"native"`` or ``"ore"`` (QIS jobs),
        ``"mir"`` (OpenQASM 3 jobs) -- or an exact format key, matched against
        ``output.compilation.compiled_circuits``; raises ``IonQJobError`` if
        none match. Returns a ``dict``/``list`` for JSON formats or ``bytes``
        for binary ones (``ionq.mir.v1``).
        """
        self.status()
        if self._status not in jobstatus.JOB_FINAL_STATES:
            raise exceptions.IonQJobStateError(
                "Cannot fetch compiled circuit until the job reaches a final state. "
                "Call wait_for_final_state() first."
            )
        assert self._job_id is not None
        circuits = ((self._metadata.get("output") or {}).get("compilation") or {}).get(
            "compiled_circuits"
        ) or {}
        fmt = self._resolve_compiled_format(lang, circuits)
        if fmt is None:
            available = ", ".join(sorted(circuits)) or "none"
            raise exceptions.IonQJobError(
                f"No compiled circuit matching {lang!r} for job {self._job_id}. "
                f"Available formats: {available}."
            )
        return self._client.get_artifact(self._job_id, circuits[fmt]["id"])

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

    def get_counts(self, circuit: QuantumCircuit | None = None) -> dict:
        """Return the counts for the job.

        .. ATTENTION::

            Result counts for jobs processed by
            :class:`IonQSimulatorBackend <qiskit_ionq.ionq_backend.IonQSimulatorBackend>`
            are returned from the API as probabilities, and are converted to counts via
            simple statistical sampling that occurs on the client side.

            To obtain the true probabilities, use the get_probabilties() method instead.

        Args:
            circuit (str or QuantumCircuit or int or None): Optional.
                The index of the experiment.

        Returns:
            dict: A dictionary of counts.
        """
        return self.result().get_counts(circuit)

    def get_memory(self, circuit=None, *, drop_leaked=False):
        """Return per-shot memory bitstrings (MSB-left), one entry per shot.

        Shorthand for ``job.result().get_memory(...)`` that fails clearly when
        the job ran without memory.

        Args:
            circuit (str or QuantumCircuit or int or None): Optional.
                The index of the experiment.
            drop_leaked (bool): Drop the shots the hardware flagged as leaked.

        Raises:
            IonQBackendError: If the job was submitted with ``memory=False``.

        Returns:
            list[str]: One bitstring per shot.
        """
        if self.memory:
            return self.result().get_memory(circuit, drop_leaked=drop_leaked)

        label = "" if circuit is None else getattr(circuit, "name", circuit)
        raise IonQBackendError(
            f'No memory for experiment "{label}". '
            "Re-run with memory=True to enable per-shot output."
        )

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

    def _result_locations(self) -> list[_ResultLocations | _UnretrievedChild]:
        """Return each circuit's result locations, in circuit order.

        A multi-circuit job keeps per-circuit results on its children (the
        parent has only an aggregate), so each child is retrieved once for both
        distributions and memory. A child that fails to retrieve becomes an
        :class:`_UnretrievedChild` instead of failing the batch.
        """
        if not self._children:
            return [_ResultLocations.from_response(self._metadata)]
        with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_REQUESTS) as pool:
            return list(pool.map(self._retrieve_child, self._children))

    def _retrieve_child(self, child_id: str) -> _ResultLocations | _UnretrievedChild:
        """Retrieve one child's result locations, keeping a failure for later."""
        try:
            response = self._client.retrieve_job(child_id)
        except exceptions.IonQAPIError as err:
            return _UnretrievedChild(child_id, err)
        return _ResultLocations.from_response(response)

    def _fetch_result_artifacts(
        self,
        locations: list[_ResultLocations | _UnretrievedChild],
        aggregation: str | None,
        extra_query_params: dict | None,
    ) -> list[dict[str, int | float]] | None:
        """Fetch each circuit's distribution artifact: the default probabilities,
        or ``aggregation``'s histogram counts (never the default in its place).

        Return ``None`` to fall back to the legacy endpoint for the whole job
        when any circuit lacks that artifact (an older job, or a method the job
        didn't publish) or, for the default, when a child couldn't be retrieved.

        Raises:
            IonQAPIError: If a child could not be retrieved and ``aggregation``
                is set; the legacy endpoint ignores ``aggregation``, so it
                cannot stand in.
        """
        retrieved: list[_ResultLocations] = []
        for location in locations:
            if isinstance(location, _UnretrievedChild):
                if aggregation is not None:
                    raise location.error
                return None
            retrieved.append(location)

        artifact_ids = [location.distribution_id(aggregation) for location in retrieved]
        if None in artifact_ids:
            return None

        get_artifact = partial(
            self._client.get_artifact, extra_query_params=extra_query_params
        )
        job_ids = [location.job_id for location in retrieved]
        with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_REQUESTS) as pool:
            payloads = list(pool.map(get_artifact, job_ids, artifact_ids))
        result_key = "probabilities" if aggregation is None else "histogram"
        return [
            _decode_distribution_artifact(payload[result_key]) for payload in payloads
        ]

    def result(
        self,
        sharpen: bool | None = None,
        aggregation: str | constants.AggregationMethod | None = None,
        timeout: float | None = None,
        wait: float = 5,
        callback: Callable | None = None,
        extra_query_params: dict | None = None,
    ):  # pylint: disable=too-many-positional-arguments
        """Retrieve job result data, blocking until the job is complete.

        .. NOTE::
            :attr:`_result` is populated by :meth:`status`, when the job
            status has reached a "final" state.

        This method calls the
        :meth:`wait_for_final_state <qiskit.providers.BaseJob.wait_for_final_state>`
        method to poll for a completed job.

        For QIS and native-gate jobs, counts and probabilities reflect error
        mitigation and the selected aggregation, while memory holds every
        recorded shot, so the two can disagree. Memory and the default counts
        include leaked shots; ``get_memory(drop_leaked=True)`` drops them from
        memory.

        Args:
            sharpen: Deprecated; use ``aggregation`` instead. ``sharpen=True``
                maps to ``aggregation="voting"``.
            aggregation: How the per-variant results of a debiased job are
                combined. One of ``"average"`` (default), ``"voting"``, or
                ``"dnl"``, or an :class:`AggregationMethod
                <qiskit_ionq.constants.AggregationMethod>` member. Has no
                effect on jobs that ran without debiasing.
            timeout: Seconds to wait for the job to reach a final state;
                ``None`` waits indefinitely.
            wait: Seconds between status polls.
            callback: Callback invoked on each status poll; see
                :meth:`wait_for_final_state
                <qiskit.providers.BaseJob.wait_for_final_state>`.
            extra_query_params: Extra query parameters forwarded on the
                results request.

        Raises:
            ValueError: If ``aggregation`` is not a supported aggregation method.
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
        # Resolve aggregation method, with sharpen as a deprecated alias.
        if sharpen is not None and not isinstance(sharpen, bool):
            warnings.warn("Invalid sharpen type")
            sharpen = None

        if sharpen is not None:
            warnings.warn(
                "The sharpen parameter is deprecated; use aggregation=... instead.",
                DeprecationWarning,
            )
            if sharpen is True and aggregation is None:
                aggregation = constants.AggregationMethod.VOTING

        if isinstance(aggregation, constants.AggregationMethod):
            aggregation = aggregation.value

        valid_aggregations = tuple(
            method.value for method in constants.AggregationMethod
        )
        if aggregation is not None and aggregation not in valid_aggregations:
            expected = ", ".join(repr(method) for method in valid_aggregations)
            raise ValueError(
                f"Unknown aggregation method {aggregation!r}; expected one of {expected}."
            )

        # Wait for the job to complete.
        try:
            self.wait_for_final_state(timeout=timeout, wait=wait, callback=callback)
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
            if self._dry_run:
                raise exceptions.IonQJobError(
                    f"Job {self._job_id} was submitted with dry_run=True; "
                    "no measurement results are produced. Use "
                    "job.compiled_circuit(...) to "
                    "retrieve the compiled circuit instead."
                )
            if self._is_qasm3:
                self._result = self._result_from_register_shots(
                    self._fetch_qasm3_shots(extra_query_params)
                )
            else:
                locations = self._result_locations()
                distributions = self._fetch_result_artifacts(
                    locations, aggregation, extra_query_params
                )
                if distributions is None:
                    data = self._client.get_results(
                        results_url=self._results_urls.get("probabilities", ""),
                        aggregation=aggregation,
                        extra_query_params=extra_query_params,
                    )
                    data_is_histogram = False
                else:
                    data = distributions
                    data_is_histogram = aggregation is not None
                self._result = self._result_from_distributions(
                    data,
                    self._fetch_memory(locations),
                    data_is_histogram=data_is_histogram,
                )

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
        if not self._job_id:
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
            # Track dry-run regardless of source; the API echoes it as a
            # top-level boolean on the job response.
            self._dry_run = bool(response.get("dry_run", False))

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

            # qasm3 jobs carry per-register results in a separate artifact.
            self._is_qasm3 = response.get("type") == "ionq.qasm3.v1"

            # Dry-run jobs have results=null; non-dry-run may also be null mid-rollout.
            if not self._dry_run:
                results = response.get("results") or {}
                self._results_urls = {
                    k: v["url"]
                    for k, v in results.items()
                    if isinstance(v, dict) and isinstance(v.get("url"), str)
                }

            # Classical-bit maps per circuit
            def _meas_map_from_header(header_dict, fallback_nq):
                """Return meas_mapped list or a default 0-based map."""
                # Header may be missing entirely (e.g. for dry-run jobs that
                # skip the qiskit metadata roundtrip); fall back to a 0..N map.
                if not isinstance(header_dict, dict):
                    return list(range(fallback_nq))
                mmap = header_dict.get("meas_mapped")
                if mmap is None or (
                    isinstance(mmap, list) and all(b is None for b in mmap)
                ):
                    return list(range(header_dict.get("n_qubits", fallback_nq)))
                return mmap

            # Classical-bit maps for every circuit
            metadata = response.get("metadata") or {}
            header_list = (
                decompress_metadata_string(metadata.get("qiskit_header")) or {}
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
                f'Failure from IonQ API "{failure.get("code", "")}: '
                f'{failure.get("error", "")}"'
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

    def _fetch_memory(
        self, locations: list[_ResultLocations | _UnretrievedChild]
    ) -> list[_Shots | None]:
        """Fetch each circuit's shots for memory, concurrently for batches.

        Every entry is ``None`` when memory is off or on the ideal simulator,
        which records no shots.
        """
        if not self.memory or self._is_ideal_simulation():
            return [None] * len(locations)
        with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_REQUESTS) as pool:
            return list(pool.map(self._fetch_circuit_shots, locations))

    def _fetch_circuit_shots(
        self, location: _ResultLocations | _UnretrievedChild
    ) -> _Shots | None:
        """Read a circuit's shots as legacy decimal outcomes, with leak flags.

        Only the v2 ``output_all`` register is read; the memory formatter then
        applies the Qiskit measurement map. Jobs without a v2 shots artifact
        read the legacy ``shots.url``, which has no leak flags. If the shots,
        or the child holding them, can't be fetched, warn and return ``None``.
        """
        try:
            if isinstance(location, _UnretrievedChild):
                raise location.error
            if location.shots_id is None:
                if location.legacy_shots_url is None:
                    return None
                return _Shots(
                    self._client.get_results(location.legacy_shots_url), leaked=None
                )
            payload = self._client.get_artifact(location.job_id, location.shots_id)
        except exceptions.IonQAPIError as err:
            warnings.warn(
                f"Failed to fetch per-shot memory for {location.job_id} ({err!r}); "
                "memory will be None.",
                UserWarning,
            )
            return None

        try:
            shots = payload["shots"]
            return _Shots(
                outcomes=[
                    _wire_order_to_decimal(shot["registers"]["output_all"])
                    for shot in shots
                ],
                leaked=[_is_leaked(shot) for shot in shots],
            )
        except (AttributeError, KeyError, TypeError, ValueError) as err:
            raise exceptions.IonQJobError(
                f"Invalid v2 shots artifact for job {location.job_id}: {err}"
            ) from err

    def _fetch_qasm3_shots(self, extra_query_params: dict | None = None) -> list:
        """Fetch the per-register shots array for a qasm3 (MCM) job."""
        shots_id = _ResultLocations.from_response(self._metadata).shots_id
        if shots_id is None:
            raise exceptions.IonQJobError(
                f"Job {self._job_id} produced no per-register shot data. "
                "Mid-circuit measurement results require sampling — run on a "
                'QPU or pass a noise model (e.g. noise_model="aria-1"); the '
                "ideal simulator returns only the aggregate distribution."
            )
        assert self._job_id is not None
        payload = self._client.get_artifact(
            self._job_id,
            shots_id,
            extra_query_params=extra_query_params,
        )
        return payload.get("shots", [])  # artifact is {"shots": [...]}

    def _result_from_register_shots(self, shots: list):
        """Build a Result from per-register shots, folding the declared
        registers via the header's ``clbit_labels``. ``output_all``
        (system-added) is excluded; leaked shots count like any other and are
        flagged alongside memory.
        """
        backend = self.backend()
        success = self._status == jobstatus.JobStatus.DONE
        metadata = self._metadata.get("metadata") or {}
        decoded = decompress_metadata_string(metadata.get("qiskit_header"))
        header = decoded if isinstance(decoded, dict) else {}

        shots = shots or []
        clbit_labels = header.get("clbit_labels") or []
        # Zero-padded binary so get_counts() splits by creg_sizes.
        width = header.get("memory_slots") or len(clbit_labels)

        # Missing registers default to all-zeros. Emit a warning because an
        # omitted register (e.g. compiler-elided measurements) is not the same
        # as a register that was actually measured and found to be all zeros.
        declared = {label[0] for label in clbit_labels}
        first = shots[0] if shots and isinstance(shots[0], dict) else {}
        missing = declared - set(first.get("registers") or {})
        if shots and missing:
            warnings.warn(
                f"Register(s) {sorted(missing)} declared by the circuit are "
                "missing from the job's shots artifact; their bits will read "
                "as 0 in counts/memory.",
                UserWarning,
            )

        counts: dict[str, int] = {}
        memory: list[str] = []
        for shot in shots:
            registers = shot.get("registers", {}) if isinstance(shot, dict) else {}
            value = 0
            for index, label in enumerate(clbit_labels):
                name, bit = label[0], label[1]
                bits = registers.get(name)
                if bits is not None and bit < len(bits) and int(bits[bit]):
                    value |= 1 << index
            key = format(value, f"0{width}b") if width else "0"
            memory.append(key)
            counts[key] = counts.get(key, 0) + 1

        total = len(shots)
        probabilities = {k: v / total for k, v in counts.items()} if total else {}

        data: dict[str, Any] = {
            "counts": counts,
            "memory": memory if self.memory else None,
            "probabilities": probabilities,
            "metadata": header,
        }
        if self.memory:
            data["leaked"] = [_is_leaked(shot) for shot in shots]

        job_result = [
            {
                "data": data,
                "shots": total,
                "header": header,
                "success": success,
            }
        ]

        return Result.from_dict(
            {
                "results": job_result,
                "job_id": self.job_id(),
                "backend_name": backend.name,
                "backend_version": backend.backend_version,
                "qobj_id": metadata.get("qobj_id"),
                "success": success,
                "time_taken": self._execution_time,
            }
        )

    def _is_ideal_simulation(self) -> bool:
        """Whether the job ran on the noiseless simulator.

        Simulator job responses always carry ``noise``; others never do.
        """
        noise = self._metadata.get("noise")
        return noise is not None and noise["model"] == "ideal"

    def _result_from_distributions(
        self,
        data,
        circuit_shots: list[_Shots | None],
        data_is_histogram: bool = False,
    ):
        """Build a Qiskit `Result` from each circuit's distribution and shots.

        Args:
            data: Decoded probability distribution or histogram. Accepted shapes:

                - ``dict[str, int | float]`` -- single circuit; outcome int
                  (as str) to probability or count.
                - ``dict[str, dict[str, int | float]]`` -- multi-circuit,
                  outer keyed by child job id; inner shaped as the
                  single-circuit case.
                - ``list[dict[str, int | float]]`` -- one entry per circuit.

                Probability values may be ``int`` (for example, exact ``0`` or
                ``1`` from a noiseless simulator) or ``float``.
            circuit_shots: Each circuit's shots for memory, or ``None`` for no
                memory.
            data_is_histogram: Whether ``data`` contains histogram counts rather
                than probabilities.

        Returns:
            Result: A Qiskit :class:`Result <qiskit.result.Result>`
                representation of this job.

        Raises:
            IonQJobError: If ``data`` is not one of the shapes above.
        """
        backend = self.backend()
        backend_name = backend.name
        backend_version = backend.backend_version
        is_ideal_sim = self._is_ideal_simulation()

        success = self._status == jobstatus.JobStatus.DONE
        metadata = self._metadata.get("metadata") or {}

        sampler_seed = (
            int(metadata.get("sampler_seed", ""))
            if metadata.get("sampler_seed", "").isdigit()
            else None
        )
        qiskit_header = decompress_metadata_string(metadata.get("qiskit_header")) or {}
        if not isinstance(qiskit_header, list):
            qiskit_header = [qiskit_header]

        shots = (
            int(metadata.get("shots", 1024))
            if str(metadata.get("shots", "1024")).isdigit()
            else 1024
        )

        if isinstance(data, dict):
            looks_like_multi = all(
                isinstance(v, dict)
                and all(isinstance(p, (int, float)) for p in v.values())
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
                header = qiskit_header[i] or {}
                # Infer clbits from result keys when metadata is absent
                # (e.g. job submitted outside qiskit); map_output returns
                # nothing for an empty clbits list.
                clbits = self._clbits[i]
                if not clbits and data[i]:
                    inferred_nq = max(int(k) for k in data[i]).bit_length() or 1
                    clbits = list(range(inferred_nq))
                n_qubits = header.get("n_qubits", len(clbits) or self._num_qubits)

                counts, probabilities = _build_counts(
                    data[i],
                    n_qubits,
                    clbits,
                    shots,
                    use_sampler=is_ideal_sim,
                    sampler_seed=sampler_seed,
                    data_is_histogram=data_is_histogram,
                )
                if data_is_histogram:
                    job_result[i]["shots"] = sum(counts.values())
                raw_shots = circuit_shots[i] if i < len(circuit_shots) else None
                if raw_shots is not None:
                    memory_slots = header.get("memory_slots") or len(clbits or [])
                    memory = _build_memory(
                        raw_shots.outcomes, n_qubits, clbits, memory_slots
                    )
                else:
                    memory = None
                job_result[i]["data"] = {
                    "counts": counts,
                    "memory": memory,
                    "probabilities": probabilities,
                    "metadata": header,
                }
                if raw_shots is not None and raw_shots.leaked is not None:
                    job_result[i]["data"]["leaked"] = raw_shots.leaked

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
