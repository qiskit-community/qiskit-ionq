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

"""Tests for the IonQSampler / IonQEstimator / IonQPrimitiveJob primitives.

These tests pin the contract that resolves issue #242: the
:meth:`IonQPrimitiveJob.job_id` returned by ``sampler.run(...)`` must be the
IonQ-server-assigned UUID, not a Qiskit-side random UUID, so that
``backend.retrieve_job(job.job_id())`` round-trips.
"""

from unittest.mock import MagicMock

import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseSamplerV2, BackendEstimatorV2
from qiskit.primitives.base.base_primitive_job import BasePrimitiveJob
from qiskit.providers import JobStatus

from qiskit_ionq import IonQSampler, IonQEstimator, IonQPrimitiveJob


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _make_ionq_job_mock(job_id: str, status: JobStatus = JobStatus.DONE) -> MagicMock:
    """An IonQJob double whose .job_id()/.status()/.result() are stubs."""
    job = MagicMock(spec=["job_id", "status", "result", "cancel"])
    job.job_id.return_value = job_id
    job.status.return_value = status
    job.cancel.return_value = None

    result = MagicMock()
    result.get_counts.side_effect = lambda idx=0: {"0": 60, "1": 40}
    job.result.return_value = result
    return job


def _make_backend_mock(jobs: list[MagicMock]) -> MagicMock:
    """A backend double whose .run() returns the supplied jobs in order."""
    backend = MagicMock()
    backend.options = MagicMock(shots=1024)
    backend.max_circuits = None
    backend.run.side_effect = list(jobs)
    return backend


def _measured_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.measure_all()
    return qc


# --------------------------------------------------------------------------- #
# IonQPrimitiveJob: contract surface
# --------------------------------------------------------------------------- #


def test_pjob_is_base_class():
    """IonQPrimitiveJob must extend qiskit's BasePrimitiveJob (not PrimitiveJob)."""
    job = IonQPrimitiveJob([_make_ionq_job_mock("abc")], lambda _: None)
    assert isinstance(job, BasePrimitiveJob)


def test_pjob_id_is_first_ionq_uuid():
    """job_id() returns the first IonQ-server UUID, never a random UUID (#242)."""
    ionq_jobs = [_make_ionq_job_mock("ionq-A"), _make_ionq_job_mock("ionq-B")]
    job = IonQPrimitiveJob(ionq_jobs, lambda _: None)

    assert job.job_id() == "ionq-A"
    assert job.ionq_job_ids == ["ionq-A", "ionq-B"]


def test_pjob_empty_id_is_empty():
    """job_id() for an empty submission is the empty string, not a UUID."""
    job = IonQPrimitiveJob([], lambda _: None)
    assert job.job_id() == ""
    assert job.ionq_job_ids == []
    # An empty job is trivially in a final state.
    assert job.in_final_state()


def test_pjob_result_is_lazy():
    """result() should not call the translator until the first .result() call."""
    translator = MagicMock(return_value="translated")
    job = IonQPrimitiveJob([_make_ionq_job_mock("z")], translator)

    translator.assert_not_called()
    out = job.result()
    assert out == "translated"
    translator.assert_called_once()

    # Second call must be memoized.
    again = job.result()
    assert again == "translated"
    translator.assert_called_once()


def test_pjob_cancel_all_ionq():
    """cancel() must invoke .cancel() on every underlying IonQJob."""
    job_a = _make_ionq_job_mock("a")
    job_b = _make_ionq_job_mock("b")
    job = IonQPrimitiveJob([job_a, job_b], lambda _: None)

    job.cancel()
    job_a.cancel.assert_called_once()
    job_b.cancel.assert_called_once()


def test_pjob_cancel_swallows_err():
    """A failure in one cancel must not skip the remaining cancellations."""
    job_a = _make_ionq_job_mock("a")
    job_a.cancel.side_effect = RuntimeError("network blip")
    job_b = _make_ionq_job_mock("b")
    job = IonQPrimitiveJob([job_a, job_b], lambda _: None)

    # Must not raise; job_b must still be cancelled.
    job.cancel()
    job_b.cancel.assert_called_once()


@pytest.mark.parametrize(
    "statuses,expected",
    [
        ([JobStatus.DONE, JobStatus.DONE], JobStatus.DONE),
        ([JobStatus.DONE, JobStatus.RUNNING], JobStatus.RUNNING),
        ([JobStatus.QUEUED, JobStatus.RUNNING], JobStatus.QUEUED),
        ([JobStatus.DONE, JobStatus.ERROR], JobStatus.ERROR),
        ([JobStatus.CANCELLED, JobStatus.DONE], JobStatus.CANCELLED),
        ([JobStatus.ERROR, JobStatus.CANCELLED], JobStatus.ERROR),
    ],
)
def test_pjob_status_aggregate(statuses, expected):
    """Aggregate status follows the documented precedence order."""
    ionq_jobs = [_make_ionq_job_mock(f"j{i}", status=s) for i, s in enumerate(statuses)]
    job = IonQPrimitiveJob(ionq_jobs, lambda _: None)
    assert job.status() == expected
    assert job.done() is (expected == JobStatus.DONE)
    assert job.running() is (expected == JobStatus.RUNNING)
    assert job.cancelled() is (expected == JobStatus.CANCELLED)


# --------------------------------------------------------------------------- #
# IonQSampler
# --------------------------------------------------------------------------- #


def test_sampler_requires_kwarg():
    """IonQSampler must require ``backend`` as a keyword-only argument."""
    with pytest.raises(TypeError):
        # pylint: disable=too-many-function-args,missing-kwoa
        IonQSampler(MagicMock())  # type: ignore[misc]


def test_sampler_is_base_sampler_v2():
    """IonQSampler must subclass BaseSamplerV2 directly."""
    sampler = IonQSampler(backend=_make_backend_mock([]))
    assert isinstance(sampler, BaseSamplerV2)


def test_sampler_shots_from_backend():
    """default_shots falls back to backend.options.shots when unspecified."""
    backend = _make_backend_mock([])
    backend.options.shots = 4096
    sampler = IonQSampler(backend=backend)
    assert sampler.default_shots == 4096


def test_sampler_returns_ionq_job():
    """sampler.run() must return an IonQPrimitiveJob whose job_id is the IonQ UUID."""
    ionq_job = _make_ionq_job_mock("server-uuid-1")
    backend = _make_backend_mock([ionq_job])
    sampler = IonQSampler(backend=backend, default_shots=100)

    primitive_job = sampler.run([_measured_circuit()])

    assert isinstance(primitive_job, IonQPrimitiveJob)
    # This is the #242 fix: not a random UUID.
    assert primitive_job.job_id() == "server-uuid-1"
    assert primitive_job.ionq_job_ids == ["server-uuid-1"]


def test_sampler_multi_pub_job_ids():
    """N PUBs -> N IonQ jobs; job_id() is the first, full list via ionq_job_ids."""
    jobs = [
        _make_ionq_job_mock("ionq-1"),
        _make_ionq_job_mock("ionq-2"),
        _make_ionq_job_mock("ionq-3"),
    ]
    backend = _make_backend_mock(jobs)
    sampler = IonQSampler(backend=backend, default_shots=100)

    qc = _measured_circuit()
    primitive_job = sampler.run([qc, qc, qc])

    assert primitive_job.job_id() == "ionq-1"
    assert primitive_job.ionq_job_ids == ["ionq-1", "ionq-2", "ionq-3"]


def test_sampler_per_pub_metadata():
    """Each SamplerPubResult must carry its own ionq_job_id in metadata."""
    jobs = [_make_ionq_job_mock("p-A"), _make_ionq_job_mock("p-B")]
    backend = _make_backend_mock(jobs)
    sampler = IonQSampler(backend=backend, default_shots=100)

    qc = _measured_circuit()
    primitive_job = sampler.run([qc, qc])
    primitive_result = primitive_job.result()

    assert primitive_result[0].metadata["ionq_job_id"] == "p-A"
    assert primitive_result[1].metadata["ionq_job_id"] == "p-B"
    assert primitive_result.metadata["ionq_job_ids"] == ["p-A", "p-B"]


def test_sampler_auto_tags_metadata():
    """Each submission must get qiskit_primitive_* auto-tags in extra_metadata."""
    backend = _make_backend_mock([_make_ionq_job_mock("k")])
    sampler = IonQSampler(backend=backend, default_shots=100)

    sampler.run([_measured_circuit()])

    _args, kwargs = backend.run.call_args
    meta = kwargs["extra_metadata"]
    assert meta["qiskit_primitive"] == "sampler"
    assert meta["qiskit_primitive_pub_index"] == 0
    assert meta["qiskit_primitive_pub_count"] == 1


def test_sampler_user_metadata_wins():
    """User-supplied extra_metadata must override the auto-tag defaults."""
    backend = _make_backend_mock([_make_ionq_job_mock("k")])
    sampler = IonQSampler(
        backend=backend,
        default_shots=100,
        run_options={
            "extra_metadata": {
                "qiskit_primitive": "custom",
                "experiment": "vqe-iter-12",
            }
        },
    )

    sampler.run([_measured_circuit()])

    _args, kwargs = backend.run.call_args
    meta = kwargs["extra_metadata"]
    assert meta["qiskit_primitive"] == "custom"
    assert meta["experiment"] == "vqe-iter-12"


def test_sampler_pub_shots_override():
    """When a SamplerPub specifies shots, they take precedence over default."""
    backend = _make_backend_mock([_make_ionq_job_mock("k")])
    sampler = IonQSampler(backend=backend, default_shots=100)

    sampler.run([(_measured_circuit(),)], shots=512)

    _args, kwargs = backend.run.call_args
    assert kwargs["shots"] == 512


def test_sampler_bitarray_per_creg():
    """Joined bitstrings must be carved into one BitArray per ClassicalRegister.

    Per-register slices can collapse multiple joined bitstrings onto the same
    key (e.g. '00' and '10' both have c[0]=0), so counts must sum, not
    overwrite - regression guard against an early bug in _counts_to_bitarrays.
    """
    # pylint: disable=import-outside-toplevel
    from qiskit.circuit import ClassicalRegister

    ionq_job = _make_ionq_job_mock("k")
    ionq_job.result.return_value.get_counts.side_effect = lambda idx=0: {
        "00": 25,
        "01": 25,
        "10": 25,
        "11": 25,
    }
    backend = _make_backend_mock([ionq_job])
    sampler = IonQSampler(backend=backend, default_shots=100)

    qc = QuantumCircuit(2)
    cr_a = ClassicalRegister(1, "a")
    cr_b = ClassicalRegister(1, "b")
    qc.add_register(cr_a)
    qc.add_register(cr_b)
    qc.h(0)
    qc.measure(0, cr_a[0])
    qc.measure(1, cr_b[0])

    data = sampler.run([qc]).result()[0].data
    assert hasattr(data, "a") and hasattr(data, "b")
    # 4 joined counts of 25 each = 100 total shots; both registers see them all.
    assert data.a.num_shots == 100
    assert data.b.num_shots == 100


# --------------------------------------------------------------------------- #
# IonQEstimator
# --------------------------------------------------------------------------- #


def test_estimator_is_backend_v2(simulator_backend):
    """IonQEstimator inherits the BackendEstimatorV2 helpers (transitional shape)."""
    estimator = IonQEstimator(backend=simulator_backend)
    assert isinstance(estimator, BackendEstimatorV2)


def test_estimator_drops_shots_opt(simulator_backend):
    """``default_shots`` (sampler-only) must be tolerated, not raise."""
    estimator = IonQEstimator(
        backend=simulator_backend,
        options={"default_shots": 999},
    )
    assert isinstance(estimator, BackendEstimatorV2)


def test_estimator_returns_ionq_job(monkeypatch, simulator_backend):
    """Estimator.run() must return an IonQPrimitiveJob carrying IonQ job IDs.

    The post-processing pipeline is heavy (Pauli grouping, basis changes); we
    intercept _submit_circuit_chunks so the test just verifies the job_id /
    ionq_job_ids surface, not the numerical correctness of expectation values.
    That is exercised by Qiskit's own BackendEstimatorV2 tests.
    """
    from qiskit.quantum_info import SparsePauliOp

    estimator = IonQEstimator(backend=simulator_backend)

    fake_job = _make_ionq_job_mock("est-uuid-1")

    def fake_submit(self, circuits, shots):  # pylint: disable=unused-argument
        return [fake_job], [{} for _ in circuits]

    # Short-circuit the actual submission and result translation.
    monkeypatch.setattr(IonQEstimator, "_submit_circuit_chunks", fake_submit)
    monkeypatch.setattr(
        IonQEstimator,
        "_translate_results",
        lambda self, plan, n: MagicMock(
            metadata={"ionq_job_ids": [j.job_id() for b in plan for j in b.jobs]}
        ),
    )

    qc = QuantumCircuit(1)
    obs = SparsePauliOp.from_list([("Z", 1.0)])
    primitive_job = estimator.run([(qc, obs)], precision=0.1)

    assert isinstance(primitive_job, IonQPrimitiveJob)
    assert primitive_job.job_id() == "est-uuid-1"
    assert primitive_job.ionq_job_ids == ["est-uuid-1"]
