from __future__ import annotations

from unittest.mock import patch

import pytest
from conftest import make_job_response, make_probs
from qiskit.providers import JobStatus

from qiskit_ionq.job import IonQJob


@pytest.fixture
def job(client):
    return IonQJob(backend=None, job_id="j1", client=client, num_qubits=2, shots=100)


class TestIonQJob:
    def test_job_id(self, job):
        assert job.job_id() == "j1"

    @pytest.mark.parametrize(
        ("api_status", "expected"),
        [
            ("submitted", JobStatus.QUEUED),
            ("ready", JobStatus.QUEUED),
            ("started", JobStatus.RUNNING),
            ("completed", JobStatus.DONE),
            ("failed", JobStatus.ERROR),
            ("canceled", JobStatus.CANCELLED),
        ],
    )
    @patch("qiskit_ionq.job.get_job")
    def test_status(self, mock_get_job, job, api_status, expected):
        mock_get_job.sync.return_value = make_job_response(status=api_status)
        assert job.status() == expected

    @patch("qiskit_ionq._result.get_job_probabilities")
    @patch("qiskit_ionq.job.wait_for_job")
    def test_result(self, mock_wait, mock_get_probs, client):
        mock_wait.return_value = make_job_response(job_id="j1", status="completed")
        mock_get_probs.sync.return_value = make_probs({"0": 0.6, "1": 0.4})
        job = IonQJob(backend=None, job_id="j1", client=client, num_qubits=1, shots=100)
        result = job.result()
        assert result.success and sum(result.get_counts(0).values()) == 100

    @patch("qiskit_ionq._result.get_job_probabilities")
    @patch("qiskit_ionq.job.wait_for_job")
    def test_result_cached(self, mock_wait, mock_get_probs, client):
        mock_wait.return_value = make_job_response(job_id="j1", status="completed")
        mock_get_probs.sync.return_value = make_probs({"0": 1.0})
        job = IonQJob(backend=None, job_id="j1", client=client, num_qubits=1, shots=100)
        r1 = job.result()
        assert job.result() is r1
        mock_wait.assert_called_once()

    @patch("qiskit_ionq.job.get_job")
    def test_status_unknown_raises(self, mock_get_job, job):
        mock_get_job.sync.return_value = make_job_response(status="some_new_status")
        with pytest.raises(ValueError, match="Unknown IonQ job status"):
            job.status()

    @patch("qiskit_ionq.job.cancel_job")
    def test_cancel(self, mock_cancel, job, client):
        job.cancel()
        mock_cancel.sync.assert_called_once_with(uuid="j1", client=client)
