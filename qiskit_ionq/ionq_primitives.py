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

from typing import Iterable

import numpy as np

from qiskit.circuit import ClassicalRegister, QuantumCircuit
from qiskit.primitives.base import BaseSamplerV2
from qiskit.primitives import BackendEstimatorV2
from qiskit.primitives.containers import (
    BitArray,
    DataBin,
    PrimitiveResult,
    SamplerPubLike,
    SamplerPubResult,
)
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.primitives.primitive_job import PrimitiveJob

from .ionq_backend import IonQBackend


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
            run_options: Extra kwargs forwarded to backend.run()
        """
        self._backend = backend
        self._default_shots = default_shots or backend.options.shots
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
        """Run and collect samples from each provided pub.

        Args:
            pubs: Iterable of sampler pubs, e.g. circuits or
                  (circuit, parameter_values, [shots]) tuples.
            shots: Default shots for pubs that don't specify their own.
        """
        if shots is None:
            shots = self._default_shots

        # Normalize everything to SamplerPub objects with shots filled in.
        coerced_pubs = [SamplerPub.coerce(pub, shots) for pub in pubs]

        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()  # submit immediately
        return job

    # internal helpers

    def _run(self, pubs: Iterable[SamplerPub]) -> PrimitiveResult[SamplerPubResult]:
        results = [self._run_pub(pub) for pub in pubs]
        return PrimitiveResult(results, metadata={"version": 2})

    def _run_pub(self, pub: SamplerPub) -> SamplerPubResult:
        circuit = pub.circuit
        param_values = pub.parameter_values
        shots = pub.shots

        # Bind parameter sets -> flat list[QuantumCircuit]
        bound_circuits = param_values.bind_all(circuit)

        if isinstance(bound_circuits, QuantumCircuit):
            # No parameters: just a single circuit
            bound_circuits_list = [bound_circuits]
        else:
            # bind_all may return an ND-array (possibly 0-D) of circuits.
            # Convert to object array and flatten in row-major order.
            bound_array = np.asarray(bound_circuits, dtype=object)
            bound_circuits_list = [c for _, c in np.ndenumerate(bound_array)]

        # Submit as a (possibly) multi-circuit job to IonQ.
        run_opts = dict(self._run_options)
        run_opts.setdefault("shots", shots)

        ionq_job = self._backend.run(bound_circuits_list, **run_opts)
        ionq_result = ionq_job.result()

        # Extract counts for each bound instance.
        counts_per_bind: list[dict[str, int]] = []
        for idx in range(len(bound_circuits_list)):
            counts_per_bind.append(ionq_result.get_counts(idx))

        # Convert counts -> BitArray(s) per classical register.
        bit_arrays = self._counts_to_bitarrays(circuit.cregs, counts_per_bind)

        return SamplerPubResult(
            DataBin(**bit_arrays, shape=pub.shape),
            metadata={"shots": shots, "circuit_metadata": circuit.metadata},
        )

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
        start_index = 0

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

    This is essentially a configured BackendEstimatorV2 that runs on an IonQBackend.
    """

    def __init__(self, *, backend: IonQBackend, options: dict | None = None) -> None:
        base_options = dict(options or {})
        run_opts = base_options.pop("run_options", {})
        backend.set_options(**run_opts)
        super().__init__(backend=backend, options=base_options)
