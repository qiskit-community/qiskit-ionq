# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
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

"""Tests for the IonQ primitives (IonQSampler and IonQEstimator)."""

from unittest.mock import MagicMock

import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BaseSamplerV2, BackendEstimatorV2

from qiskit_ionq.ionq_primitives import IonQSampler, IonQEstimator


def test_sampler_requires_keyword_backend(simulator_backend):
    """IonQSampler should require `backend` as a keyword-only argument."""
    # Positional use should fail because the signature is `*, backend`.
    with pytest.raises(TypeError):
        IonQSampler(simulator_backend)  # type: ignore[call-arg]

    # But the keyword form should work.
    sampler = IonQSampler(backend=simulator_backend)
    assert isinstance(sampler, BaseSamplerV2)


def test_sampler_uses_backend_default_shots(simulator_backend):
    """IonQSampler default_shots should default to backend.options.shots."""
    sampler = IonQSampler(backend=simulator_backend)

    # The sampler should expose the backend it is wrapping.
    assert sampler.backend is simulator_backend

    # And it should pick up the backend's default shot count if none is given.
    backend_default = getattr(simulator_backend.options, "shots", None)
    assert sampler.default_shots == backend_default


def test_sampler_explicit_default_shots_override(simulator_backend):
    """An explicit default_shots argument should override the backend default."""
    explicit_shots = 2048
    sampler = IonQSampler(backend=simulator_backend, default_shots=explicit_shots)

    assert sampler.backend is simulator_backend
    assert sampler.default_shots == explicit_shots


def test_sampler_includes_ionq_job_ids_in_metadata():
    """IonQSampler.run should record IonQ job IDs in both pub and top-level metadata."""
    # Fully mock the backend and its run() / job() / result() APIs.
    backend = MagicMock()
    backend.options = MagicMock(shots=100)

    ionq_result = MagicMock()
    # Single circuit, single index -> deterministic counts.
    ionq_result.get_counts.side_effect = lambda idx: {"0": 10, "1": 90}

    ionq_job = MagicMock()
    ionq_job.job_id.return_value = "job-123"
    ionq_job.result.return_value = ionq_result

    backend.run.return_value = ionq_job

    sampler = IonQSampler(backend=backend, default_shots=100)

    qc = QuantumCircuit(1)
    qc.measure_all()

    job = sampler.run([qc], shots=50)
    primitive_result = job.result()

    # Top-level metadata should contain a list of IonQ job IDs.
    assert primitive_result.metadata.get("ionq_job_ids") == ["job-123"]

    # Per-pub metadata should contain the individual ionq_job_id.
    pub_result = primitive_result[0]
    assert pub_result.metadata.get("ionq_job_id") == "job-123"
    assert pub_result.metadata.get("shots") == 50


def test_estimator_is_backend_estimator(simulator_backend):
    """IonQEstimator should be a BackendEstimatorV2."""
    estimator = IonQEstimator(backend=simulator_backend)
    assert isinstance(estimator, BackendEstimatorV2)


def test_estimator_applies_run_options_to_backend(simulator_backend, monkeypatch):
    """IonQEstimator should pull run_options out and pass them to backend.set_options."""
    backend = simulator_backend

    # Spy on set_options so we don't actually mutate backend options for real.
    set_options_spy = MagicMock()
    monkeypatch.setattr(backend, "set_options", set_options_spy)

    run_options = {"shots": 123, "job_settings": {"foo": "bar"}}
    IonQEstimator(backend=backend, options={"run_options": run_options})

    # IonQEstimator.__init__ should have called set_options with the run_options
    # extracted from the user options.
    set_options_spy.assert_called_once_with(**run_options)


def test_estimator_accepts_default_shots_option(simulator_backend):
    """IonQEstimator should ignore 'default_shots' in options (no TypeError)."""
    # This should not raise, even though BackendEstimatorV2.Options
    # does not know about "default_shots".
    estimator = IonQEstimator(
        backend=simulator_backend,
        options={"run_options": {}, "default_shots": 999},
    )
    assert isinstance(estimator, BackendEstimatorV2)


def test_estimator_includes_ionq_job_ids_in_metadata(monkeypatch):
    """IonQEstimator.run should record IonQ job IDs in the top-level PrimitiveResult metadata.

    All backend interaction is mocked via _run_pubs_with_job_ids, so no real requests occur.
    """
    backend = MagicMock()
    estimator = IonQEstimator(backend=backend, options={"run_options": {}})

    def fake_run_pubs_with_job_ids(self, pubs, shots):
        # Return one fake PubResult per pub, plus synthetic job IDs.
        fake_results = [MagicMock(name=f"PubResult-{i}") for i in range(len(pubs))]
        fake_job_ids = [f"job-{shots}-{i}" for i in range(len(pubs))]
        return fake_results, fake_job_ids

    # Patch the helper so we don't talk to a real backend at all.
    monkeypatch.setattr(
        IonQEstimator,
        "_run_pubs_with_job_ids",
        fake_run_pubs_with_job_ids,
    )

    qc = QuantumCircuit(1)
    obs = SparsePauliOp.from_list([("Z", 1.0)])

    # precision=0.1 -> shots = ceil(1 / 0.1^2) = 100 in IonQEstimator._run_with_job_ids
    job = estimator.run([(qc, obs)], precision=0.1)
    primitive_result = job.result()

    assert primitive_result.metadata.get("ionq_job_ids") == ["job-100-0"]
