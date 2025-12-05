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

# Copyright 2024 IonQ, Inc. (www.ionq.com)
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

"""IonQ's Sampler and Estimator primitive implementations."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from qiskit.circuit import ClassicalRegister, QuantumCircuit
from qiskit.primitives import BackendEstimatorV2
from qiskit.primitives.base import BaseSamplerV2
from qiskit.primitives.backend_estimator_v2 import _prepare_counts
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
from qiskit.primitives.primitive_job import PrimitiveJob

from .ionq_backend import IonQBackend


def _job_id_from_job(job: Any) -> str | None:
    """Extraction of a job id from a Qiskit job object."""
    if not hasattr(job, "job_id"):
        return None
    try:
        job_id_val = job.job_id()  # typical BaseJob API
    except TypeError:
        job_id_val = job.job_id  # property-style
    return str(job_id_val) if job_id_val is not None else None


class IonQSampler(BaseSamplerV2):
    """SamplerV2 implementation backed by an IonQBackend."""

    def __init__(
        self,
        *,
        backend: IonQBackend,
        default_shots: int | None = None,
        run_options: dict | None = None,
    ) -> None:
        """
        Args:
            backend: IonQ backend to run on (simulator or QPU).
            default_shots: Default shots if none are specified at run() time.
                If None, uses backend.options.shots.
            run_options: Extra kwargs forwarded to backend.run().
        """
        self._backend = backend
        self._default_shots = (
            backend.options.shots if default_shots is None else default_shots
        )
        self._run_options = dict(run_options or {})

    # required properties

    @property
    def backend(self) -> IonQBackend:
        """Backend that this sampler runs on."""
        return self._backend

    @property
    def default_shots(self) -> int:
        """Default shots used when shots=None."""
        return self._default_shots

    # main SamplerV2 API

    def run(
        self,
        pubs: Iterable[SamplerPubLike],
        *,
        shots: int | None = None,
    ) -> PrimitiveJob[PrimitiveResult[SamplerPubResult]]:
        """Run and collect samples from each provided pub."""
        if shots is None:
            shots = self._default_shots

        coerced_pubs = [SamplerPub.coerce(pub, shots) for pub in pubs]
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    # internal helpers

    def _run(self, pubs: Iterable[SamplerPub]) -> PrimitiveResult[SamplerPubResult]:
        """Execute all pubs and collect their results and IonQ job IDs."""
        results: list[SamplerPubResult] = []
        job_ids: list[str] = []

        for pub in pubs:
            pub_result, job_id = self._run_pub(pub)
            results.append(pub_result)
            if job_id is not None:
                job_ids.append(job_id)

        return PrimitiveResult(
            results,
            metadata={"version": 2, "ionq_job_ids": job_ids},
        )

    def _run_pub(self, pub: SamplerPub) -> tuple[SamplerPubResult, str | None]:
        """Execute a single sampler pub on the IonQ backend."""
        circuit = pub.circuit
        param_values = pub.parameter_values
        shots = pub.shots

        # Bind parameter sets -> flat list[QuantumCircuit]
        bound_circuits = param_values.bind_all(circuit)
        if isinstance(bound_circuits, QuantumCircuit):
            bound_circuits_list = [bound_circuits]
        else:
            # bind_all may return an ND-array (possibly 0-D) of circuits.
            bound_array = np.asarray(bound_circuits, dtype=object)
            bound_circuits_list = [c for _, c in np.ndenumerate(bound_array)]

        # Submit as a (possibly) multi-circuit job to IonQ.
        run_opts = dict(self._run_options)
        run_opts.setdefault("shots", shots)

        ionq_job = self._backend.run(bound_circuits_list, **run_opts)
        ionq_job_id = _job_id_from_job(ionq_job)
        ionq_result = ionq_job.result()

        # Extract counts for each bound instance.
        counts_per_bind = [
            ionq_result.get_counts(idx) for idx in range(len(bound_circuits_list))
        ]

        # Convert counts -> BitArray(s) per classical register.
        bit_arrays = self._counts_to_bitarrays(circuit.cregs, counts_per_bind)

        metadata: dict[str, Any] = {
            "shots": shots,
            "circuit_metadata": circuit.metadata,
        }
        if ionq_job_id is not None:
            metadata["ionq_job_id"] = ionq_job_id

        data_bin = DataBin(**bit_arrays, shape=pub.shape)
        return SamplerPubResult(data_bin, metadata=metadata), ionq_job_id

    @staticmethod
    def _counts_to_bitarrays(
        cregs: list[ClassicalRegister],
        counts_list: list[dict[str, int]],
    ) -> dict[str, BitArray]:
        """Split global bitstrings into per-register BitArray objects.

        This matches the logic used in other SamplerV2 implementations:
        right-most bits correspond to the last classical register, etc.
        """
        bit_arrays: dict[str, BitArray] = {}
        start_index = 0  # we slice from the right using negative indices

        for creg in cregs:
            creg_counts_list: list[dict[str, int]] = []

            for counts in counts_list:
                new_counts: dict[str, int] = {}
                for bitstring, value in counts.items():
                    # Slice from the right, per register size.
                    new_key = bitstring[start_index - creg.size : start_index or None]
                    new_counts[new_key] = value
                creg_counts_list.append(new_counts)

            bit_arrays[creg.name] = BitArray.from_counts(creg_counts_list, creg.size)
            start_index -= creg.size

        return bit_arrays


class IonQEstimator(BackendEstimatorV2):
    """EstimatorV2 wrapper for IonQ backends.

    This is essentially a configured BackendEstimatorV2 that runs on an IonQBackend,
    but also records the underlying IonQ job IDs in the PrimitiveResult metadata.
    """

    def __init__(self, *, backend: IonQBackend, options: dict | None = None) -> None:
        # Copy options so we can safely mutate.
        base_options = dict(options or {})

        # Pull out backend run_options (applied via backend.set_options).
        run_opts = base_options.pop("run_options", {})
        if run_opts:
            backend.set_options(**run_opts)

        # Ignore any options that BackendEstimatorV2.Options doesn't know about
        # (e.g. "default_shots") to avoid TypeError in Options(**...).
        base_options.pop("default_shots", None)

        super().__init__(backend=backend, options=base_options or None)

    # Override run so we can inject IonQ job IDs into the PrimitiveResult metadata.
    def run(
        self,
        pubs: Iterable[EstimatorPubLike],
        *,
        precision: float | None = None,
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        if precision is None:
            precision = self._options.default_precision

        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]
        self._validate_pubs(coerced_pubs)

        job = PrimitiveJob(self._run_with_job_ids, coerced_pubs)
        job._submit()
        return job

    def _run_with_job_ids(
        self,
        pubs: list[EstimatorPub],
    ) -> PrimitiveResult[PubResult]:
        """Like BackendEstimatorV2._run, but also aggregates IonQ job IDs."""
        # Group pubs by required shot count.
        pub_dict: dict[int, list[int]] = defaultdict(list)
        for index, pub in enumerate(pubs):
            shots = int(math.ceil(1.0 / pub.precision**2))
            pub_dict[shots].append(index)

        results: list[PubResult | None] = [None] * len(pubs)
        all_job_ids: list[str] = []

        for shots, indices in pub_dict.items():
            grouped_pubs = [pubs[i] for i in indices]
            grouped_results, job_ids = self._run_pubs_with_job_ids(
                grouped_pubs,
                shots,
            )
            all_job_ids.extend(job_ids)
            for i, pub_result in zip(indices, grouped_results):
                results[i] = pub_result

        # All entries should be filled; type-ignore for static checkers.
        return PrimitiveResult(
            results,  # type: ignore[list-item]
            metadata={"version": 2, "ionq_job_ids": all_job_ids},
        )

    def _run_pubs_with_job_ids(
        self,
        pubs: list[EstimatorPub],
        shots: int,
    ) -> tuple[list[PubResult], list[str]]:
        """Compute results for pubs that share the same shot count, and return
        both their PubResults and the IonQ job IDs used."""
        preprocessed_data: list[Any] = []
        flat_circuits: list[QuantumCircuit] = []

        for pub in pubs:
            data = self._preprocess_pub(pub)
            preprocessed_data.append(data)
            flat_circuits.extend(data.circuits)

        run_result, metadata, job_ids = self._run_circuits_with_job_ids(
            flat_circuits,
            self._backend,
            shots=shots,
            seed_simulator=self._options.seed_simulator,
        )

        counts = _prepare_counts(run_result)

        results: list[PubResult] = []
        start = 0
        for pub, data in zip(pubs, preprocessed_data):
            end = start + len(data.circuits)
            expval_map = self._calc_expval_map(counts[start:end], metadata[start:end])
            start = end
            results.append(self._postprocess_pub(pub, expval_map, data, shots))

        return results, job_ids

    @staticmethod
    def _run_circuits_with_job_ids(
        circuits: QuantumCircuit | list[QuantumCircuit],
        backend: IonQBackend,
        clear_metadata: bool = True,
        **run_options: Any,
    ) -> tuple[list[Any], list[dict], list[str]]:
        """Run circuits on the backend, mirroring Qiskit's _run_circuits helper,
        but also collect the job IDs of the underlying IonQ jobs.
        """
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]

        # Preserve and optionally clear circuit.metadata.
        metadata: list[dict] = []
        for circuit in circuits:
            metadata.append(circuit.metadata)
            if clear_metadata:
                circuit.metadata = {}

        # Chunk circuits according to backend.max_circuits, if present.
        max_circuits = getattr(backend, "max_circuits", None)

        jobs: list[Any] = []
        if max_circuits:
            for pos in range(0, len(circuits), max_circuits):
                jobs.append(
                    backend.run(circuits[pos : pos + max_circuits], **run_options)
                )
        else:
            jobs.append(backend.run(circuits, **run_options))

        results: list[Any] = []
        job_ids: list[str] = []

        for job in jobs:
            job_id = _job_id_from_job(job)
            if job_id is not None:
                job_ids.append(job_id)
            results.append(job.result())

        return results, metadata, job_ids
