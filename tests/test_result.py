from __future__ import annotations

from unittest.mock import patch

import pytest
from conftest import make_probs

from qiskit_ionq._result import to_bitarray, to_qiskit_result


class TestToQiskitResult:
    @pytest.mark.parametrize(
        ("probs", "num_qubits", "shots"),
        [
            ({"0": 0.5, "3": 0.5}, 2, 1000),
            ({"0": 0.33, "1": 0.33, "2": 0.34}, 2, 1000),
        ],
    )
    @patch("qiskit_ionq._result.get_job_probabilities")
    def test_counts_sum_to_shots(
        self, mock_get_probs, client, probs, num_qubits, shots
    ):
        mock_get_probs.sync.return_value = make_probs(probs)
        result = to_qiskit_result(client, "job-1", num_qubits=num_qubits, shots=shots)
        assert result.success is True
        assert sum(result.get_counts(0).values()) == shots

    @patch("qiskit_ionq._result.get_job_probabilities")
    def test_single_state(self, mock_get_probs, client):
        mock_get_probs.sync.return_value = make_probs({"0": 1.0})
        result = to_qiskit_result(client, "job-2", num_qubits=1, shots=100)
        assert result.get_counts(0) == {"0": 100}

    @patch("qiskit_ionq._result.get_job_probabilities")
    def test_job_id_preserved(self, mock_get_probs, client):
        mock_get_probs.sync.return_value = make_probs({"0": 1.0})
        assert (
            to_qiskit_result(client, "my-job-42", num_qubits=1, shots=100).job_id
            == "my-job-42"
        )


class TestToBitArray:
    @patch("qiskit_ionq._result.get_job_probabilities")
    def test_simple(self, mock_get_probs, client):
        mock_get_probs.sync.return_value = make_probs({"0": 0.5, "3": 0.5})
        ba = to_bitarray(client, "job-4", num_qubits=2, shots=1000)
        assert ba.num_bits == 2 and ba.num_shots == 1000
        assert sum(ba.get_counts().values()) == 1000

    @patch("qiskit_ionq._result.get_job_probabilities")
    def test_single_state(self, mock_get_probs, client):
        mock_get_probs.sync.return_value = make_probs({"0": 1.0})
        assert to_bitarray(client, "job-5", num_qubits=1, shots=100).get_counts() == {
            "0": 100
        }
