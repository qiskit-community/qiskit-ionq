from __future__ import annotations

from unittest.mock import patch

import pytest
from ionq_core.models.job_creation_response import JobCreationResponse
from qiskit.circuit import QuantumCircuit

from qiskit_ionq.backend import IonQBackend


@pytest.fixture
def backend(client, simulator_backend_info):
    return IonQBackend(provider=None, backend_info=simulator_backend_info, client=client)


class TestIonQBackend:
    def test_properties(self, backend):
        assert backend.name == "simulator"
        assert backend.num_qubits == 29
        assert backend.max_circuits is None
        assert backend.options.shots == 1024
        assert {"h", "cx", "measure"} <= backend.target.operation_names

    @patch("qiskit_ionq._submit.create_job")
    def test_run_returns_job(self, mock_create_job, backend):
        mock_create_job.sync.return_value = JobCreationResponse(id="job-run", status="submitted", session_id=None)
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        assert backend.run(qc, shots=100).job_id() == "job-run"

    @patch("qiskit_ionq._submit.create_job")
    def test_run_uses_default_shots(self, mock_create_job, backend):
        mock_create_job.sync.return_value = JobCreationResponse(id="job-default", status="submitted", session_id=None)
        qc = QuantumCircuit(1)
        qc.h(0)
        backend.run(qc)
        assert mock_create_job.sync.call_args.kwargs["body"].shots == 1024

    @patch("qiskit_ionq._submit.create_job")
    def test_run_accepts_multiple_circuits(self, mock_create_job, backend):
        mock_create_job.sync.return_value = JobCreationResponse(id="job-multi", status="submitted", session_id=None)
        qc1 = QuantumCircuit(1)
        qc1.h(0)
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        job = backend.run([qc1, qc2])
        assert job.job_id() == "job-multi"
        assert len(mock_create_job.sync.call_args.kwargs["body"].input_.circuits) == 2

    def test_default_options(self):
        opts = IonQBackend._default_options()
        assert opts.error_mitigation is None
        assert opts.noise_model is None
        assert opts.noise_seed is None

    def test_native_gateset(self, client, simulator_backend_info):
        backend = IonQBackend(provider=None, backend_info=simulator_backend_info, client=client, gateset="native")
        assert backend.gateset == "native"
        assert "gpi" in backend.target.operation_names
        assert "h" not in backend.target.operation_names
