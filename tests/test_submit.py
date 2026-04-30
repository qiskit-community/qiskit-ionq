from __future__ import annotations

from unittest.mock import patch

import pytest
from ionq_core.models.job_creation_response import JobCreationResponse
from qiskit.circuit import QuantumCircuit

from qiskit_ionq._submit import submit
from qiskit_ionq._target import build_target
from qiskit_ionq.gates import GPIGate


@pytest.fixture
def qis_target(client, simulator_backend_info):
    return build_target(simulator_backend_info, client, "qis")


def _mock_resp(mock, job_id="job-1"):
    mock.sync.return_value = JobCreationResponse(id=job_id, status="submitted", session_id=None)


def _body(mock):
    return mock.sync.call_args.kwargs["body"]


def _submit(client, target, circuits, gateset="qis", shots=100, **kw):
    return submit(
        client=client,
        backend="simulator",
        circuits=circuits,
        shots=shots,
        gateset=gateset,
        target=target,
        **kw,
    )


class TestSubmit:
    @patch("qiskit_ionq._submit.create_job")
    def test_submits_single_circuit(self, mock, client, qis_target):
        _mock_resp(mock)
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        assert _submit(client, qis_target, [qc]) == "job-1"
        body = _body(mock)
        assert body.backend == "simulator" and body.shots == 100
        assert body.type_ == "ionq.multi-circuit.v1"
        assert len(body.input_.circuits) == 1

    @patch("qiskit_ionq._submit.create_job")
    def test_passes_session_id(self, mock, client, qis_target):
        _mock_resp(mock)
        qc = QuantumCircuit(1)
        qc.h(0)
        _submit(client, qis_target, [qc], session_id="sess-1")
        assert _body(mock).session_id == "sess-1"

    @patch("qiskit_ionq._submit.create_job")
    def test_submits_multi_circuit(self, mock, client, qis_target):
        _mock_resp(mock, "job-multi")
        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc1.cx(0, 1)
        qc2 = QuantumCircuit(2)
        qc2.x(0)
        assert _submit(client, qis_target, [qc1, qc2], shots=200) == "job-multi"
        body = _body(mock)
        assert body.shots == 200 and len(body.input_.circuits) == 2

    @patch("qiskit_ionq._submit.create_job")
    def test_submits_with_error_mitigation(self, mock, client, qis_target):
        _mock_resp(mock)
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()
        _submit(client, qis_target, [qc], shots=1024, error_mitigation={"debiasing": True})
        assert _body(mock).settings.error_mitigation.debiasing is True

    @patch("qiskit_ionq._submit.create_job")
    def test_submits_with_noise(self, mock, client, qis_target):
        _mock_resp(mock)
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()
        _submit(client, qis_target, [qc], shots=1024, noise_model="aria-1", noise_seed=42)
        noise = _body(mock).noise
        assert noise.model == "aria-1" and noise.seed == 42

    @patch("qiskit_ionq._submit.create_job")
    def test_submits_native_gateset(self, mock, client, simulator_backend_info):
        _mock_resp(mock, "job-native")
        target = build_target(simulator_backend_info, client, "native")
        qc = QuantumCircuit(1)
        qc.append(GPIGate(0.5), [0])
        qc.measure_all()
        assert _submit(client, target, [qc], gateset="native") == "job-native"
        assert _body(mock).input_.gateset == "native"
