from __future__ import annotations

import datetime
from unittest.mock import patch

import pytest
from ionq_core.models.session import Session

from qiskit_ionq.backend import IonQBackend
from qiskit_ionq.sampler import IonQSampler
from qiskit_ionq.session import IonQSession


def _make_session(session_id="sess-123", status="created"):
    return Session(
        id=session_id,
        created_at=datetime.datetime.now(tz=datetime.UTC),
        organization_id="org-1",
        backend="simulator",
        project_id=None,
        creator_id="user-1",
        ended_at=None,
        ender_id=None,
        active=True,
        status=status,
        started_at=None,
    )


@pytest.fixture
def backend(client, simulator_backend_info):
    return IonQBackend(provider=None, backend_info=simulator_backend_info, client=client)


class TestIonQSession:
    @patch("ionq_core.session.end_session")
    @patch("ionq_core.session.create_session")
    def test_context_manager_lifecycle(self, mock_create, mock_end, backend):
        mock_create.sync.return_value = _make_session("sess-abc")
        with IonQSession(backend) as session:
            assert session.session_id == "sess-abc"
        mock_create.sync.assert_called_once()
        mock_end.sync.assert_called_once()

    @patch("ionq_core.session.end_session")
    @patch("ionq_core.session.create_session")
    def test_end_called_on_exception(self, mock_create, mock_end, backend):
        mock_create.sync.return_value = _make_session("sess-err")
        with pytest.raises(RuntimeError), IonQSession(backend):
            raise RuntimeError("boom")
        mock_end.sync.assert_called_once()

    @patch("ionq_core.session.end_session")
    @patch("ionq_core.session.create_session")
    def test_sampler_factory(self, mock_create, mock_end, backend):
        mock_create.sync.return_value = _make_session()
        with IonQSession(backend) as session:
            sampler = session.sampler()
            assert isinstance(sampler, IonQSampler) and sampler._session is session

    @patch("ionq_core.session.end_session")
    @patch("ionq_core.session.create_session")
    def test_session_with_limits(self, mock_create, mock_end, backend):
        mock_create.sync.return_value = _make_session()
        with IonQSession(backend, max_jobs=5, max_time=10, max_cost=100.0):
            pass
        body = mock_create.sync.call_args.kwargs["body"]
        assert body.settings.job_count_limit == 5
        assert body.settings.duration_limit_min == 10
        assert body.settings.cost_limit.value == 100.0

    @patch("ionq_core.session.get_session")
    def test_from_id(self, mock_get, backend):
        mock_get.sync.return_value = _make_session("sess-42")
        assert IonQSession.from_id(backend, "sess-42").session_id == "sess-42"

    @patch("ionq_core.session.end_session")
    @patch("ionq_core.session.get_session")
    @patch("ionq_core.session.create_session")
    def test_status(self, mock_create, mock_get, mock_end, backend):
        mock_create.sync.return_value = _make_session()
        mock_get.sync.return_value = _make_session(status="started")
        with IonQSession(backend) as session:
            assert session.status() == "started"
