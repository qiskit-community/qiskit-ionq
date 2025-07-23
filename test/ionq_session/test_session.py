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
"""Test the IonQ Session creation, submission, and ending."""

import pytest
from unittest.mock import MagicMock, call
from qiskit_ionq import Session


# Helpers
def _backend(name="ionq_qpu.aria-1"):
    bk = MagicMock()
    bk.name.return_value = name
    return bk


def _client():
    """Return a MagicMock with the attributes Session expects."""
    cl = MagicMock()
    cl.api_headers = {}
    # make_path is only used to build URLs for get_with_retry/post; the exact value
    # is irrelevant so long as it is a string.
    cl.make_path.side_effect = lambda *parts: "/".join(parts)
    return cl


# Tests
def test_create_session_payload():
    client = _client()
    client.post.return_value = {"id": "sess-123"}

    sess = Session(
        backend=_backend(),
        client=client,
        max_jobs=3,
        max_time=10,
        max_cost=1000,
    )

    assert sess.session_id == "sess-123"

    # first positional arg(s) are the *path_parts
    client.post.assert_called_once_with(
        "sessions",
        json_body={
            "backend": "qpu.aria-1",
            "settings": {
                "job_count_limit": 3,
                "duration_limit_min": 10,
                "cost_limit": {"unit": "usd", "value": 1000},
            },
        },
    )


def test_details_status_usage():
    client = _client()
    # side‑effect order: first call is __init__ verification
    client.get_with_retry.return_value.json.return_value = {
        "state": "active",
        "usage_time": 123,
    }

    sess = Session.from_id("sess-456", backend=_backend(), client=client)

    assert sess.status() == "active"
    assert sess.usage() == 123

    client.get_with_retry.assert_called_with(
        "sessions/sess-456", headers=client.api_headers
    )


def test_cancel_queued_jobs():
    client = _client()
    # First call ‑‑ constructor check; second call ‑‑ queued jobs listing
    client.get_with_retry.side_effect = [
        MagicMock(json=MagicMock(return_value={})),  # details()
        MagicMock(
            json=MagicMock(return_value={"jobs": [{"id": "job1"}, {"id": "job2"}]})
        ),
    ]
    client.cancel_jobs = MagicMock()

    sess = Session.from_id("sess-1", backend=_backend(), client=client)
    sess.cancel()

    client.cancel_jobs.assert_called_once_with(["job1", "job2"])


def test_context_manager_closes_on_exception():
    client = _client()
    client.post.return_value = {"id": "sess-789"}

    with pytest.raises(RuntimeError):
        with Session(backend=_backend(), client=client, max_jobs=1) as _sess:
            raise RuntimeError("boom")

    # A POST should have been made to /sessions/<id>/end
    assert client.post.call_count == 2
    expected_end_call = call("sessions", "sess-789", "end")
    assert expected_end_call in client.post.mock_calls


def test_invalid_init_raises():
    with pytest.raises(ValueError):
        Session(backend=_backend(), client=_client(), create_new=False, session_id=None)


def test_backend_run_with_session():
    """Test that the backend.run method uses the session ID from the context."""
    backend = _backend()
    backend.run = MagicMock()

    client = _client()
    client.post.return_value = {"id": "sess-42"}

    with Session(backend=backend, client=client, max_jobs=1) as sess:
        backend.run("circ")  # no session_id parameter

    assert backend.run.call_args.kwargs["session_id"] == "sess-42"
