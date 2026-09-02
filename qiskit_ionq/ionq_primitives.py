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

# Copyright 2026 IonQ, Inc. (www.ionq.com)
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

"""IonQ-native V2 primitives (Sampler and Estimator) for Qiskit.

The two key design choices follow the only shipping third-party V2-primitive
precedent (``qiskit-braket-provider`` v0.11 ``BraketSampler``/``BraketEstimator``
/ ``BraketPrimitiveTask``) and the IBM Runtime ``SamplerV2``/``EstimatorV2``/
``RuntimeJobV2`` reference chain:

1. Both primitives **eagerly** submit jobs to the IonQ Cloud at ``run()`` time
   and return an :class:`IonQPrimitiveJob` (a :class:`BasePrimitiveJob`
   subclass) whose :meth:`job_id` is the IonQ-server-assigned UUID. This
   restores the natural round-trip
   ``backend.retrieve_job(sampler.run(...).job_id())`` that the generic
   :class:`qiskit.primitives.BackendSamplerV2` breaks (issue #242).
2. Multi-PUB convention follows IBM Runtime: :meth:`IonQPrimitiveJob.job_id`
   returns a single string (the first IonQ job ID submitted by ``run()``).
   The full list of IonQ job IDs created by one ``run()`` call is available
   via :attr:`IonQPrimitiveJob.ionq_job_ids`, and is also stamped into
   :attr:`PrimitiveResult.metadata` (aggregate) and each
   :attr:`SamplerPubResult.metadata` / :attr:`PubResult.metadata` (per-PUB).

``IonQSampler`` is a direct ``BaseSamplerV2`` subclass and owns the entire
counts-to-:class:`~qiskit.primitives.BitArray` path so it does not depend on
:class:`qiskit.primitives.BackendSamplerV2`'s hardcoded ``memory=True`` shotwise
contract.

``IonQEstimator`` currently subclasses :class:`qiskit.primitives.BackendEstimatorV2`
because the inherited observable-grouping / Pauli-basis-change / expectation-value
post-processing helpers (``_preprocess_pub``, ``_postprocess_pub``,
``_calc_expval_map``) would otherwise need to be re-vendored. The override
surface is just :meth:`IonQEstimator.run`, which collects IonQ job IDs and
returns an :class:`IonQPrimitiveJob`. A future PR may switch to
``BaseEstimatorV2`` for full symmetry with the sampler.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from qiskit.circuit import ClassicalRegister, QuantumCircuit
from qiskit.primitives import BackendEstimatorV2
from qiskit.primitives.backend_estimator_v2 import _prepare_counts
from qiskit.primitives.base import BaseSamplerV2
from qiskit.primitives.containers import (
    BitArray,
    DataBin,
    EstimatorPubLike,
    PrimitiveResult,
    PubResult,
    SamplerPubLike,
    SamplerPubResult,
)
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.primitives.base.base_primitive_job import BasePrimitiveJob
from qiskit.providers import JobStatus

from .ionq_backend import IonQBackend
from .ionq_job import IonQJob


# JobStatus values ordered by precedence when aggregating multiple IonQ jobs:
# the first matching status (left-to-right) wins. This mirrors the natural
# lifecycle - any non-final state on any underlying job means the aggregate
# is not yet final.
_STATUS_PRECEDENCE: tuple[JobStatus, ...] = (
    JobStatus.ERROR,
    JobStatus.CANCELLED,
    JobStatus.INITIALIZING,
    JobStatus.VALIDATING,
    JobStatus.QUEUED,
    JobStatus.RUNNING,
    JobStatus.DONE,
)


class IonQPrimitiveJob(BasePrimitiveJob[PrimitiveResult, JobStatus]):
    """Primitive job that wraps one or more eagerly-submitted IonQ jobs.

    Mirrors the structure of qiskit-braket-provider's ``BraketPrimitiveTask``:
    the underlying IonQ job IDs are known at construction time (submission
    has already happened), :meth:`job_id` returns the first IonQ UUID, and
    :meth:`result` lazily fetches per-job results and hands them to a
    caller-supplied translator that builds the final :class:`PrimitiveResult`.

    Args:
        ionq_jobs: IonQ jobs submitted by the parent primitive. May be empty
            (e.g. ``sampler.run([])``); in that case :meth:`job_id` returns
            the empty string and :meth:`result` calls the translator with an
            empty list.
        result_translator: Callable that consumes the list of finished
            :class:`IonQJob` instances (in submission order) and returns the
            primitive's :class:`PrimitiveResult`. Invoked at most once;
            the result is memoized.
    """

    def __init__(
        self,
        ionq_jobs: list[IonQJob],
        result_translator: Callable[[list[IonQJob]], PrimitiveResult],
    ) -> None:
        first_id = ionq_jobs[0].job_id() if ionq_jobs else ""
        super().__init__(job_id=first_id)
        self._ionq_jobs = list(ionq_jobs)
        self._result_translator = result_translator
        self._cached_result: PrimitiveResult | None = None

    # IonQ-specific surface --------------------------------------------------

    @property
    def ionq_jobs(self) -> list[IonQJob]:
        """The underlying :class:`IonQJob` objects, in submission order."""
        return list(self._ionq_jobs)

    @property
    def ionq_job_ids(self) -> list[str]:
        """IonQ-server UUIDs for every job submitted by this primitive call."""
        return [j.job_id() for j in self._ionq_jobs]

    # BasePrimitiveJob abstract surface -------------------------------------

    def job_id(self) -> str:
        """Return the IonQ-server UUID of the first submitted job.

        For multi-PUB runs, use :attr:`ionq_job_ids` to recover the full list.
        Per-PUB IDs are also stamped onto each
        :attr:`SamplerPubResult.metadata` / :attr:`PubResult.metadata`.
        """
        return self._job_id

    def result(self) -> PrimitiveResult:
        if self._cached_result is None:
            self._cached_result = self._result_translator(self._ionq_jobs)
        return self._cached_result

    def status(self) -> JobStatus:
        return self._aggregate_status()

    def cancel(self) -> None:
        for job in self._ionq_jobs:
            try:
                job.cancel()
            except Exception:  # pylint: disable=broad-except
                # Best-effort: a failure mid-cancel must not prevent
                # cancelling the remaining jobs.
                continue

    def done(self) -> bool:
        return self._aggregate_status() == JobStatus.DONE

    def running(self) -> bool:
        return self._aggregate_status() == JobStatus.RUNNING

    def cancelled(self) -> bool:
        return self._aggregate_status() == JobStatus.CANCELLED

    def in_final_state(self) -> bool:
        return self._aggregate_status() in {
            JobStatus.DONE,
            JobStatus.ERROR,
            JobStatus.CANCELLED,
        }

    # Internals -------------------------------------------------------------

    def _aggregate_status(self) -> JobStatus:
        """Combine per-job statuses according to :data:`_STATUS_PRECEDENCE`."""
        if not self._ionq_jobs:
            return JobStatus.DONE
        observed = {j.status() for j in self._ionq_jobs}
        for candidate in _STATUS_PRECEDENCE:
            if candidate in observed:
                return candidate
        # All statuses are non-canonical; fall back to the first one.
        return next(iter(observed))


class IonQSampler(BaseSamplerV2):
    """SamplerV2 implementation that submits directly to IonQ Cloud.

    Eagerly creates one IonQ job per :class:`~qiskit.primitives.containers.SamplerPub`
    at :meth:`run` time so the returned :class:`IonQPrimitiveJob` carries real
    IonQ-server-assigned UUIDs from the start. Counts are converted to
    :class:`~qiskit.primitives.BitArray` per classical register; this path
    does not depend on per-shot ``memory=True`` output (see issue #242).

    Server-side discoverability: pass ``run_options`` containing any combination
    of ``extra_metadata``, ``name``, ``tags``, ``error_mitigation``,
    ``noise_model``, ``sampler_seed`` etc. - they are forwarded verbatim to
    :meth:`IonQBackend.run` for every PUB. To distinguish primitive submissions
    in the IonQ Cloud UI, this class also stamps a small set of automatic tags
    into ``extra_metadata`` (see :meth:`_build_run_options`).

    Args:
        backend: IonQ backend to submit to (simulator or QPU).
        default_shots: Default shot count when neither the PUB nor :meth:`run`
            specifies one. If ``None``, falls back to ``backend.options.shots``.
        run_options: Extra kwargs forwarded to :meth:`IonQBackend.run` for
            every PUB submission (e.g. ``{"error_mitigation": ...,
            "extra_metadata": {...}, "name": "vqe-iter-12"}``).
    """

    _PRIMITIVE_TAG = "sampler"

    def __init__(
        self,
        *,
        backend: IonQBackend,
        default_shots: int | None = None,
        run_options: dict | None = None,
    ) -> None:
        self._backend = backend
        if default_shots is None:
            default_shots = getattr(backend.options, "shots", 1024)
        self._default_shots = default_shots
        self._run_options = dict(run_options or {})

    # Configuration accessors -----------------------------------------------

    @property
    def backend(self) -> IonQBackend:
        """Backend this sampler submits to."""
        return self._backend

    @property
    def default_shots(self) -> int:
        """Default shots used when :meth:`run` is called with ``shots=None``."""
        return self._default_shots

    # Main SamplerV2 API ----------------------------------------------------

    def run(
        self,
        pubs: Iterable[SamplerPubLike],
        *,
        shots: int | None = None,
    ) -> IonQPrimitiveJob:
        """Eagerly submit one IonQ job per PUB; return an :class:`IonQPrimitiveJob`."""
        if shots is None:
            shots = self._default_shots

        coerced_pubs = [SamplerPub.coerce(pub, shots) for pub in pubs]
        ionq_jobs: list[IonQJob] = []
        bound_lists: list[list[QuantumCircuit]] = []

        for index, pub in enumerate(coerced_pubs):
            bound_circuits = _bind_pub_circuits(pub)
            bound_lists.append(bound_circuits)
            run_opts = self._build_run_options(pub, index, len(coerced_pubs))
            ionq_jobs.append(self._backend.run(bound_circuits, **run_opts))

        def translate(jobs: list[IonQJob]) -> PrimitiveResult[SamplerPubResult]:
            return self._translate_results(coerced_pubs, jobs, bound_lists)

        return IonQPrimitiveJob(ionq_jobs, translate)

    # Internals -------------------------------------------------------------

    def _build_run_options(
        self,
        pub: SamplerPub,
        pub_index: int,
        pub_count: int,
    ) -> dict:
        """Merge ``self._run_options`` with per-PUB shots and primitive tags.

        Per Jon Donovan's IonQ-internal note ("our platform should support
        easy discoverability of what ran from qiskit"), every primitive
        submission is auto-tagged with the primitive kind and PUB position
        so that even users who lose the Qiskit-side ``job_id`` can find their
        runs in the IonQ Cloud UI.
        """
        opts = dict(self._run_options)
        opts.setdefault("shots", pub.shots)

        auto_tags = {
            "qiskit_primitive": self._PRIMITIVE_TAG,
            "qiskit_primitive_pub_index": pub_index,
            "qiskit_primitive_pub_count": pub_count,
        }
        user_metadata = dict(opts.get("extra_metadata") or {})
        # User-supplied keys win over auto-tags so users can override if needed.
        merged = {**auto_tags, **user_metadata}
        opts["extra_metadata"] = merged
        return opts

    def _translate_results(
        self,
        pubs: list[SamplerPub],
        jobs: list[IonQJob],
        bound_lists: list[list[QuantumCircuit]],
    ) -> PrimitiveResult[SamplerPubResult]:
        """Build the per-PUB ``SamplerPubResult``s from finished IonQ jobs."""
        pub_results: list[SamplerPubResult] = []
        for pub, job, bound in zip(pubs, jobs, bound_lists):
            ionq_result = job.result()
            counts_per_bind = [ionq_result.get_counts(idx) for idx in range(len(bound))]
            bit_arrays = _counts_to_bitarrays(pub.circuit.cregs, counts_per_bind)
            data_bin = DataBin(**bit_arrays, shape=pub.shape)
            metadata = {
                "shots": pub.shots,
                "circuit_metadata": pub.circuit.metadata,
                "ionq_job_id": job.job_id(),
            }
            pub_results.append(SamplerPubResult(data_bin, metadata=metadata))

        return PrimitiveResult(
            pub_results,
            metadata={
                "version": 2,
                "ionq_job_ids": [j.job_id() for j in jobs],
            },
        )


class IonQEstimator(BackendEstimatorV2):
    """EstimatorV2 implementation that surfaces IonQ job IDs.

    Currently inherits from :class:`qiskit.primitives.BackendEstimatorV2` to
    reuse its observable-grouping / Pauli-basis / expectation-value helpers
    (``_preprocess_pub``, ``_postprocess_pub``, ``_calc_expval_map``). The
    only override is :meth:`run`, which:

    1. Coerces and validates PUBs (matching ``BackendEstimatorV2.run``).
    2. Groups PUBs by required shot count.
    3. Eagerly submits the measurement circuits via :meth:`IonQBackend.run`,
       chunked by ``backend.max_circuits`` if present, so the IonQ job IDs
       are known immediately.
    4. Returns an :class:`IonQPrimitiveJob` whose lazy translator finishes
       the standard estimator post-processing once the IonQ jobs are done.

    Switching to a pure :class:`~qiskit.primitives.BaseEstimatorV2` subclass
    (for full symmetry with :class:`IonQSampler`) is deferred to a follow-up:
    it requires re-vendoring ~150 lines of private Qiskit estimator helpers
    and isn't necessary to close issue #242.
    """

    _PRIMITIVE_TAG = "estimator"

    def __init__(
        self,
        *,
        backend: IonQBackend,
        options: dict | None = None,
        run_options: dict | None = None,
    ) -> None:
        base_options = dict(options or {})
        # ``default_shots`` is a SamplerV2 option, not an EstimatorV2 option;
        # tolerate it for symmetry with IonQSampler.
        base_options.pop("default_shots", None)
        super().__init__(backend=backend, options=base_options or None)
        self._ionq_run_options = dict(run_options or {})

    def run(
        self,
        pubs: Iterable[EstimatorPubLike],
        *,
        precision: float | None = None,
    ) -> IonQPrimitiveJob:
        if precision is None:
            precision = self._options.default_precision
        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]
        self._validate_pubs(coerced_pubs)

        # Group pubs by required shot count (matches BackendEstimatorV2._run).
        pub_groups: dict[int, list[int]] = defaultdict(list)
        for index, pub in enumerate(coerced_pubs):
            shots = int(math.ceil(1.0 / pub.precision**2))
            pub_groups[shots].append(index)

        plan: list[_EstimatorBatch] = []
        all_ionq_jobs: list[IonQJob] = []

        for shots, indices in pub_groups.items():
            grouped_pubs = [coerced_pubs[i] for i in indices]
            preprocessed = [self._preprocess_pub(p) for p in grouped_pubs]
            flat_circuits: list[QuantumCircuit] = []
            for data in preprocessed:
                flat_circuits.extend(data.circuits)

            jobs, run_metadata = self._submit_circuit_chunks(flat_circuits, shots)
            all_ionq_jobs.extend(jobs)
            plan.append(
                _EstimatorBatch(
                    indices=indices,
                    pubs=grouped_pubs,
                    preprocessed=preprocessed,
                    jobs=jobs,
                    circuit_metadata=run_metadata,
                    shots=shots,
                )
            )

        def translate(_jobs: list[IonQJob]) -> PrimitiveResult[PubResult]:
            return self._translate_results(plan, len(coerced_pubs))

        return IonQPrimitiveJob(all_ionq_jobs, translate)

    # Internals -------------------------------------------------------------

    def _submit_circuit_chunks(
        self,
        circuits: list[QuantumCircuit],
        shots: int,
    ) -> tuple[list[IonQJob], list[dict]]:
        """Submit ``circuits`` to the IonQ backend, chunked by ``max_circuits``.

        Returns the submitted :class:`IonQJob` objects (in chunk order) and
        the per-circuit ``metadata`` snapshot taken before submission - the
        Qiskit estimator pipeline clears it on the originals, so we capture
        a copy for downstream consumers if they ever want it.
        """
        metadata = [dict(c.metadata or {}) for c in circuits]
        for circuit in circuits:
            circuit.metadata = {}

        max_circuits = getattr(self._backend, "max_circuits", None)
        run_opts = dict(self._ionq_run_options)
        run_opts["shots"] = shots
        run_opts["extra_metadata"] = {
            "qiskit_primitive": self._PRIMITIVE_TAG,
            **dict(run_opts.get("extra_metadata") or {}),
        }

        jobs: list[IonQJob] = []
        if max_circuits:
            for pos in range(0, len(circuits), max_circuits):
                chunk = circuits[pos : pos + max_circuits]
                jobs.append(self._backend.run(chunk, **run_opts))
        else:
            jobs.append(self._backend.run(circuits, **run_opts))
        return jobs, metadata

    def _translate_results(
        self,
        plan: list[_EstimatorBatch],
        pub_count: int,
    ) -> PrimitiveResult[PubResult]:
        """Post-process all IonQ jobs into per-PUB ``PubResult``s."""
        results: list[PubResult | None] = [None] * pub_count
        all_job_ids: list[str] = []

        for batch in plan:
            run_result = [job.result() for job in batch.jobs]
            counts = _prepare_counts(run_result)
            start = 0
            for index, pub, data in zip(batch.indices, batch.pubs, batch.preprocessed):
                end = start + len(data.circuits)
                expval_map = self._calc_expval_map(
                    counts[start:end], batch.circuit_metadata[start:end]
                )
                start = end
                pub_result = self._postprocess_pub(pub, expval_map, data, batch.shots)
                # Attach this PUB's IonQ job ID(s). When max_circuits causes a
                # PUB's circuits to span chunks, ``pub.circuits`` per chunk is
                # tracked in ``batch.jobs`` order; surface them all.
                pub_result.metadata["ionq_job_id"] = batch.jobs[0].job_id()
                pub_result.metadata["ionq_job_ids"] = [j.job_id() for j in batch.jobs]
                results[index] = pub_result
            all_job_ids.extend(j.job_id() for j in batch.jobs)

        return PrimitiveResult(
            results,  # type: ignore[arg-type]
            metadata={"version": 2, "ionq_job_ids": all_job_ids},
        )


# Module-level helpers ----------------------------------------------------------


class _EstimatorBatch:
    """Per-shot-group bookkeeping for :meth:`IonQEstimator.run`.

    Plain-attribute container - intentionally not a dataclass to keep the
    public surface of this module minimal.
    """

    __slots__ = (
        "indices",
        "pubs",
        "preprocessed",
        "jobs",
        "circuit_metadata",
        "shots",
    )

    def __init__(  # pylint: disable=too-many-positional-arguments,too-many-arguments
        self,
        indices: list[int],
        pubs: list[EstimatorPub],
        preprocessed: list[Any],
        jobs: list[IonQJob],
        circuit_metadata: list[dict],
        shots: int,
    ) -> None:
        self.indices = indices
        self.pubs = pubs
        self.preprocessed = preprocessed
        self.jobs = jobs
        self.circuit_metadata = circuit_metadata
        self.shots = shots


def _bind_pub_circuits(pub: SamplerPub) -> list[QuantumCircuit]:
    """Bind a SamplerPub's parameter array into a flat list of circuits."""
    bound = pub.parameter_values.bind_all(pub.circuit)
    if isinstance(bound, QuantumCircuit):
        return [bound]
    bound_array = np.asarray(bound, dtype=object)
    return [circuit for _, circuit in np.ndenumerate(bound_array)]


def _counts_to_bitarrays(
    cregs: list[ClassicalRegister],
    counts_list: list[dict[str, int]],
) -> dict[str, BitArray]:
    """Split joined Qiskit bitstrings into one :class:`BitArray` per register.

    Bitstrings from :meth:`IonQResult.get_counts` are MSB-left over all
    classical registers concatenated; this helper carves them up so the
    resulting :class:`DataBin` has one ``BitArray`` per ``ClassicalRegister``,
    matching what :class:`qiskit.primitives.BackendSamplerV2` produces.
    """
    bit_arrays: dict[str, BitArray] = {}
    start_index = 0  # negative-index slice offset, walking from the right

    for creg in cregs:
        creg_counts: list[dict[str, int]] = []
        for counts in counts_list:
            split: dict[str, int] = {}
            for bitstring, value in counts.items():
                key = bitstring[start_index - creg.size : start_index or None]
                # Multiple joined bitstrings can collapse to the same per-register
                # key (e.g. '00' and '10' both have c[0]=0); accumulate.
                split[key] = split.get(key, 0) + value
            creg_counts.append(split)
        bit_arrays[creg.name] = BitArray.from_counts(creg_counts, creg.size)
        start_index -= creg.size

    return bit_arrays
