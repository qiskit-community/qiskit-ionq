from __future__ import annotations

# ionq-core requires Python >= 3.12. On older Pythons the package is not
# installed; tell pytest to skip every test file in this directory rather than
# erroring out at conftest-import time.
try:
    import ionq_core  # noqa: F401
except ImportError:
    collect_ignore_glob = ["test_*.py"]
else:
    import pytest
    from ionq_core.client import AuthenticatedClient
    from ionq_core.models.backend import Backend
    from ionq_core.models.characterization import Characterization
    from ionq_core.models.characterization_fidelity import CharacterizationFidelity
    from ionq_core.models.characterization_fidelity_spam import (
        CharacterizationFidelitySpam,
    )
    from ionq_core.models.characterization_timing import CharacterizationTiming
    from ionq_core.models.circuit_job_settings import CircuitJobSettings
    from ionq_core.models.circuit_job_stats import CircuitJobStats
    from ionq_core.models.get_job_response import GetJobResponse
    from ionq_core.models.get_results_response import GetResultsResponse
    from ionq_core.models.json_object import JsonObject

    _BACKEND_DEFAULTS = dict(
        status="available",
        average_queue_time=0.0,
        degraded=False,
        last_updated="2026-01-01T00:00:00Z",
    )

    @pytest.fixture
    def client():
        return AuthenticatedClient(
            base_url="https://test.invalid/v0.4",
            token="test-api-key",
            prefix="apiKey",
            auth_header_name="Authorization",
        )

    @pytest.fixture
    def simulator_backend_info():
        return Backend(backend="simulator", qubits=29, **_BACKEND_DEFAULTS)

    @pytest.fixture
    def qpu_backend_info():
        return Backend(
            backend="qpu.aria-1",
            qubits=25,
            average_queue_time=120.0,
            characterization_id="00000000-0000-0000-0000-000000000123",
            **{k: v for k, v in _BACKEND_DEFAULTS.items() if k != "average_queue_time"},
        )

    @pytest.fixture
    def characterization():
        fidelity = CharacterizationFidelity(
            spam=CharacterizationFidelitySpam(median=0.995)
        )
        fidelity.additional_properties["1q"] = {"mean": 0.9995}
        fidelity.additional_properties["2q"] = {"mean": 0.985}
        timing = CharacterizationTiming(readout=100, reset=50)
        return Characterization(
            backend="qpu.aria-1", qubits=25, fidelity=fidelity, timing=timing
        )

    def make_probs(data: dict[str, float]) -> GetResultsResponse:
        resp = GetResultsResponse()
        resp.additional_properties = data
        return resp

    def make_job_response(job_id="job-123", status="completed", backend="simulator"):
        return GetJobResponse(
            id=job_id,
            status=status,
            type_="ionq.circuit.v1",
            backend=backend,
            dry_run=False,
            submitter_id="user-1",
            project_id=None,
            parent_job_id=None,
            session_id=None,
            metadata=None,
            name=None,
            submitted_at="2026-01-01T00:00:00Z",
            started_at=None,
            completed_at=None,
            predicted_wait_time_ms=None,
            predicted_execution_duration_ms=None,
            execution_duration_ms=None,
            failure=None,
            output=JsonObject(),
            settings=CircuitJobSettings(),
            stats=CircuitJobStats(),
            results=None,
            child_job_ids=None,
        )
