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

"""V2 primitives on IonQ backends, against a fake IonQ Cloud that simulates the
submitted circuits: Qiskit's generic ones, and IonQSampler / IonQEstimator,
whose job IDs are IonQ job IDs (issue #242)."""

from collections import Counter

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ClassicalRegister, Parameter, QuantumRegister
from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2
from qiskit.providers import JobStatus
from qiskit.quantum_info import SparsePauliOp

from qiskit_ionq import IonQEstimator, IonQSampler


def _two_registers():
    """``alpha`` = (q1, q0) is always '10'; ``beta`` = (q3, q2) is a Bell pair."""
    alpha, beta = ClassicalRegister(2, "alpha"), ClassicalRegister(2, "beta")
    circ = QuantumCircuit(QuantumRegister(4, "q"), alpha, beta)
    circ.x(1)
    circ.h(2)
    circ.cx(2, 3)
    circ.measure([0, 1], alpha)
    circ.measure([2, 3], beta)
    return circ


def _ry():
    """A one-qubit ``ry(theta)`` circuit."""
    circ = QuantumCircuit(1)
    circ.ry(Parameter("theta"), 0)
    return circ


def test_memory_matches_counts(any_backend, fake_cloud):
    """Per-shot memory agrees with counts, on the ideal simulator too."""
    circ = transpile(_two_registers(), any_backend, optimization_level=0)
    result = any_backend.run([circ], shots=100, memory=True).result()
    assert Counter(result.get_memory()) == result.get_counts()
    assert len(fake_cloud.submitted) == 1


def test_sampler_v2(any_backend, fake_cloud):
    """BackendSamplerV2 returns each circuit's outcomes on every IonQ backend."""
    circ = transpile(_two_registers(), any_backend, optimization_level=0)
    rot = _ry()
    rot.measure_all()
    rot = transpile(rot, any_backend, optimization_level=0)
    sampler = BackendSamplerV2(backend=any_backend)
    result = sampler.run([(circ, None, 200), (rot, [[0.0], [np.pi]], 30)]).result()

    assert result[0].data.alpha.get_counts() == {"10": 200}
    beta = result[0].data.beta.get_counts()
    assert set(beta) == {"00", "11"}
    assert min(beta.values()) > 50
    assert result[1].data.meas.get_counts(0) == {"0": 30}
    assert result[1].data.meas.get_counts(1) == {"1": 30}
    assert len(fake_cloud.submitted) == 2  # one job per shot count


def test_estimator_v2(any_backend, fake_cloud):
    """BackendEstimatorV2 returns exact expectation values on every IonQ backend."""
    # q0 = |1>, q1 = |+>, q2 = |+i>: definite in the Z, X and Y bases.
    circ = QuantumCircuit(3)
    circ.x(0)
    circ.h(1)
    circ.h(2)
    circ.s(2)
    circ = transpile(circ, any_backend, optimization_level=0)
    observables = [
        SparsePauliOp(label).apply_layout(circ.layout)
        for label in ("IIZ", "IXI", "YII", "YXZ")
    ]
    rot = transpile(_ry(), any_backend, optimization_level=0)
    z_obs = SparsePauliOp("Z").apply_layout(rot.layout)
    estimator = BackendEstimatorV2(backend=any_backend)
    pubs = [(circ, observables), (rot, z_obs, [[0.0], [np.pi]], 0.1)]
    result = estimator.run(pubs, precision=0.05).result()

    np.testing.assert_allclose(result[0].data.evs, [-1, 1, 1, -1], atol=1e-9)
    np.testing.assert_allclose(result[1].data.evs, [1, -1], atol=1e-9)
    assert len(fake_cloud.submitted) == 2  # one job per shot count


def test_sampler_job_id_is_retrievable(any_backend, fake_cloud):
    """The sampler's job ID is the IonQ job that ran all its pubs."""
    circ = QuantumCircuit(2, 2)
    circ.x(0)
    circ.measure([0, 1], [0, 1])
    circ = transpile(circ, any_backend, optimization_level=0)
    job = IonQSampler(backend=any_backend).run([circ, (circ, None, 50)], shots=50)
    assert len(fake_cloud.submitted) == 1  # submitted before run() returns

    retrieved = any_backend.retrieve_job(job.job_id())
    assert retrieved.get_counts() == [{"01": 50}, {"01": 50}]
    assert [pub.data.c.get_counts() for pub in job.result()] == [{"01": 50}] * 2
    assert job.status() == JobStatus.DONE
    assert job.done() and job.in_final_state()
    assert not (job.running() or job.cancelled())


def test_estimator_job_id_is_retrievable(any_backend, fake_cloud):
    """The estimator's job ID is the IonQ job that ran all its pubs."""
    circ = QuantumCircuit(1)
    circ.x(0)
    circ = transpile(circ, any_backend, optimization_level=0)
    z_obs = SparsePauliOp("Z").apply_layout(circ.layout)
    estimator = IonQEstimator(backend=any_backend)
    job = estimator.run([(circ, z_obs), (circ, [z_obs, -z_obs])], precision=0.1)
    assert len(fake_cloud.submitted) == 1  # submitted before run() returns

    assert any_backend.retrieve_job(job.job_id()).status() == JobStatus.DONE
    np.testing.assert_allclose(job.result()[0].data.evs, -1)
    np.testing.assert_allclose(job.result()[1].data.evs, [-1, 1])


def test_pubs_must_share_shots(simulator_backend):
    """One IonQ job has one shot count, so pubs that need different ones raise."""
    circ = QuantumCircuit(1, 1)
    circ.measure(0, 0)
    with pytest.raises(ValueError, match="share a shot count; got \\[10, 20\\]"):
        IonQSampler(backend=simulator_backend).run([(circ, None, 10), (circ, None, 20)])
    z_obs = SparsePauliOp("Z")
    with pytest.raises(ValueError, match="share a shot count; got \\[25, 100\\]"):
        IonQEstimator(backend=simulator_backend).run(
            [
                (QuantumCircuit(1), z_obs, None, 0.1),
                (QuantumCircuit(1), z_obs, None, 0.2),
            ]
        )


def test_cancel_cancels_ionq_job(simulator_backend, requests_mock):
    """Cancelling the primitive job cancels the IonQ job."""
    client = simulator_backend.client
    requests_mock.post(client.make_path("jobs"), json={"id": "queued"})
    cancel = requests_mock.put(
        client.make_path("jobs", "queued", "status", "cancel"), json={}
    )
    circ = QuantumCircuit(1, 1)
    circ.measure(0, 0)
    IonQSampler(backend=simulator_backend).run([circ]).cancel()
    assert cancel.called_once
