from __future__ import annotations

from unittest.mock import patch

import pytest
from conftest import make_job_response, make_probs
from ionq_core.models.job_creation_response import JobCreationResponse
from qiskit.circuit import QuantumCircuit

from qiskit_ionq.backend import IonQBackend
from qiskit_ionq.sampler import IonQSampler, IonQSamplerJob


@pytest.fixture
def sampler(client, simulator_backend_info):
    return IonQSampler(IonQBackend(provider=None, backend_info=simulator_backend_info, client=client))


def _mock_resp(mock, job_id="sampler-job"):
    mock.sync.return_value = JobCreationResponse(id=job_id, status="submitted", session_id=None)


class TestIonQSampler:
    @patch("qiskit_ionq._submit.create_job")
    def test_run_returns_job(self, mock, sampler):
        _mock_resp(mock)
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        assert sampler.run([qc], shots=100).job_id() == "sampler-job"

    @patch("qiskit_ionq._submit.create_job")
    def test_batches_into_multi_circuit(self, mock, sampler):
        _mock_resp(mock, "multi-job")
        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc1.cx(0, 1)
        qc2 = QuantumCircuit(2)
        qc2.x(0)
        sampler.run([qc1, qc2], shots=100)
        body = mock.sync.call_args.kwargs["body"]
        assert body.type_ == "ionq.multi-circuit.v1" and len(body.input_.circuits) == 2

    @patch("qiskit_ionq._submit.create_job")
    def test_run_empty_pubs_raises(self, mock, sampler):
        with pytest.raises(ValueError, match="At least one PUB"):
            sampler.run([], shots=100)

    @patch("qiskit_ionq._submit.create_job")
    def test_run_non_uniform_shots_raises(self, mock, sampler):
        qc1 = QuantumCircuit(1)
        qc1.h(0)
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        with pytest.raises(ValueError, match="same shot count"):
            sampler.run([(qc1, None, 50), (qc2, None, 100)])


class TestIonQSamplerJob:
    @patch("qiskit_ionq._result.get_job_probabilities")
    @patch("qiskit_ionq.sampler.wait_for_job")
    @patch("qiskit_ionq._submit.create_job")
    def test_result_multi_circuit(self, mock_create, mock_wait, mock_probs, sampler):
        _mock_resp(mock_create, "parent-job")
        parent = make_job_response(job_id="parent-job", status="completed")
        parent.child_job_ids = ["child-1", "child-2"]
        mock_wait.return_value = parent
        mock_probs.sync.side_effect = [make_probs({"0": 0.5, "3": 0.5}), make_probs({"0": 1.0})]
        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc1.cx(0, 1)
        qc2 = QuantumCircuit(2)
        qc2.x(0)
        result = sampler.run([qc1, qc2], shots=100).result()
        assert len(result) == 2
        assert result[0].data.meas.num_bits == 2 and result[0].data.meas.num_shots == 100
        assert result[1].data.meas.num_shots == 100

    @patch("qiskit_ionq._result.get_job_probabilities")
    @patch("qiskit_ionq.sampler.wait_for_job")
    @patch("qiskit_ionq._submit.create_job")
    def test_single_circuit_result(self, mock_create, mock_wait, mock_probs, sampler):
        _mock_resp(mock_create, "single-job")
        mock_wait.return_value = make_job_response(job_id="single-job", status="completed")
        mock_probs.sync.return_value = make_probs({"0": 0.5, "1": 0.5})
        qc = QuantumCircuit(1)
        qc.h(0)
        result = sampler.run([qc], shots=100).result()
        assert len(result) == 1 and result[0].data.meas.num_bits == 1

    @pytest.mark.parametrize(
        ("api_status", "method", "expected"),
        [
            ("completed", "done", True),
            ("started", "done", False),
            ("started", "running", True),
            ("completed", "running", False),
            ("canceled", "cancelled", True),
            ("completed", "cancelled", False),
            ("completed", "in_final_state", True),
            ("failed", "in_final_state", True),
            ("canceled", "in_final_state", True),
            ("started", "in_final_state", False),
        ],
    )
    @patch("qiskit_ionq.sampler.get_job")
    def test_status_methods(self, mock_get_job, api_status, method, expected):
        mock_get_job.sync.return_value = make_job_response(status=api_status)
        job = IonQSamplerJob(job_id="j1", client=None, num_qubits_list=[2], shots_list=[100])
        assert getattr(job, method)() == expected

    @patch("qiskit_ionq.sampler.get_job")
    def test_status_returns_string(self, mock_get_job):
        mock_get_job.sync.return_value = make_job_response(status="started")
        assert IonQSamplerJob(job_id="j1", client=None, num_qubits_list=[2], shots_list=[100]).status() == "started"

    @patch("qiskit_ionq.sampler.cancel_job")
    def test_cancel(self, mock_cancel):
        IonQSamplerJob(job_id="j1", client="fake", num_qubits_list=[2], shots_list=[100]).cancel()
        mock_cancel.sync.assert_called_once_with(uuid="j1", client="fake")
