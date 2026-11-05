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


"""Qiskit's backend V2 primitives, with IonQ job IDs.

Each ``run`` submits its pubs as one IonQ job before returning, so
``job.job_id()`` is that job's ID (Qiskit's own classes return a random UUID).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

from qiskit.primitives import (
    BackendEstimatorV2,
    BackendSamplerV2,
    BasePrimitiveJob,
    EstimatorPub,
    EstimatorPubLike,
    PrimitiveResult,
    SamplerPub,
    SamplerPubLike,
)
from qiskit.primitives.backend_estimator_v2 import _prepare_counts
from qiskit.primitives.backend_sampler_v2 import _analyze_circuit, _prepare_memory
from qiskit.providers import JobStatus
from qiskit.result import Result

from .ionq_job import IonQJob


class IonQPrimitiveJob(BasePrimitiveJob[PrimitiveResult, JobStatus]):
    """A primitive job whose :meth:`job_id` is its IonQ job's ID."""

    def __init__(
        self, ionq_job: IonQJob, postprocess: Callable[[Result], PrimitiveResult]
    ):
        super().__init__(ionq_job.job_id())
        self._ionq_job = ionq_job
        self._postprocess = postprocess
        self._result: PrimitiveResult | None = None

    def result(self) -> PrimitiveResult:
        if self._result is None:
            self._result = self._postprocess(self._ionq_job.result())
        return self._result

    def status(self) -> JobStatus:
        return self._ionq_job.status()

    def done(self) -> bool:
        return self._ionq_job.done()

    def running(self) -> bool:
        return self._ionq_job.running()

    def cancelled(self) -> bool:
        return self._ionq_job.cancelled()

    def in_final_state(self) -> bool:
        return self._ionq_job.in_final_state()

    def cancel(self) -> None:
        self._ionq_job.cancel()


def _one_job_shots(shots: list[int]) -> int:
    """The shot count of the one IonQ job that runs every pub."""
    if len(set(shots)) != 1:
        raise ValueError(
            "A run() is one IonQ job, so its pubs must share a shot count "
            f"(for the estimator, a precision); got shots {sorted(set(shots))}."
        )
    return shots[0]


class IonQSampler(BackendSamplerV2):
    """:class:`~qiskit.primitives.BackendSamplerV2` returning an :class:`IonQPrimitiveJob`."""

    def run(
        self, pubs: Iterable[SamplerPubLike], *, shots: int | None = None
    ) -> IonQPrimitiveJob:
        if shots is None:
            shots = self.options.default_shots
        coerced_pubs = [SamplerPub.coerce(pub, shots) for pub in pubs]
        self._validate_pubs(coerced_pubs)
        bound = [pub.parameter_values.bind_all(pub.circuit) for pub in coerced_pubs]
        job = self.backend.run(
            [circuit for circuits in bound for circuit in circuits.ravel()],
            shots=_one_job_shots([pub.shots for pub in coerced_pubs]),
            memory=True,
            **(self.options.run_options or {}),
        )

        def postprocess(result: Result) -> PrimitiveResult:
            memory = _prepare_memory([result])
            pub_results = []
            start = 0
            for pub, circuits in zip(coerced_pubs, bound):
                meas_info, max_num_bytes = _analyze_circuit(pub.circuit)
                end = start + circuits.size
                pub_results.append(
                    self._postprocess_pub(
                        memory[start:end],
                        pub.shots,
                        circuits.shape,
                        meas_info,
                        max_num_bytes,
                        pub.circuit.metadata,
                        None,  # IonQ results are level 2 (classified)
                    )
                )
                start = end
            return PrimitiveResult(pub_results, metadata={"version": 2})

        return IonQPrimitiveJob(job, postprocess)


class IonQEstimator(BackendEstimatorV2):
    """:class:`~qiskit.primitives.BackendEstimatorV2` returning an :class:`IonQPrimitiveJob`."""

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> IonQPrimitiveJob:
        if precision is None:
            precision = self.options.default_precision
        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]
        self._validate_pubs(coerced_pubs)
        shots = _one_job_shots(
            [math.ceil(1.0 / pub.precision**2) for pub in coerced_pubs]
        )
        preprocessed = [self._preprocess_pub(pub) for pub in coerced_pubs]
        circuits = [circuit for data in preprocessed for circuit in data.circuits]
        # As BackendEstimatorV2 does, keep the measured Paulis out of the job.
        metadata = [circuit.metadata for circuit in circuits]
        for circuit in circuits:
            circuit.metadata = {}
        job = self.backend.run(circuits, shots=shots)

        def postprocess(result: Result) -> PrimitiveResult:
            counts = _prepare_counts([result])
            pub_results = []
            start = 0
            for pub, data in zip(coerced_pubs, preprocessed):
                end = start + len(data.circuits)
                expval_map = self._calc_expval_map(
                    counts[start:end], metadata[start:end]
                )
                pub_results.append(self._postprocess_pub(pub, expval_map, data, shots))
                start = end
            return PrimitiveResult(pub_results, metadata={"version": 2})

        return IonQPrimitiveJob(job, postprocess)
