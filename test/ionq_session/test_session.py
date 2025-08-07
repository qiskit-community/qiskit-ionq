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


@pytest.fixture()
def backend() -> MagicMock:
    """Return a minimally-specced mock backend."""
    bk = MagicMock()
    bk.name.return_value = "ionq_qpu.aria-1"
    bk.client = MagicMock()
    bk.client.make_path.side_effect = lambda *parts: "/".join(parts)
    return bk


def test_create_session_payload(backend):
    backend.client.post.return_value = {"id": "sess-123"}

    sess = Session(backend=backend, max_jobs=3, max_time=10, max_cost=1000)

    assert sess.session_id == "sess-123"
    backend.client.post.assert_called_once_with(
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


def test_status_and_usage(backend):
    # Every call to .json() should return the same dict.
    backend.client.get_with_retry.return_value.json.return_value = {
        "state": "active",
        "usage_time": 123,
    }

    sess = Session.from_id("sess-456", backend=backend)

    assert sess.status() == "active"
    assert sess.usage() == 123

    # The helper is invoked once for __init__, and once per accessor.
    assert backend.client.get_with_retry.call_count == 3
    backend.client.get_with_retry.assert_any_call(
        "sessions/sess-456", headers=backend.client.api_headers
    )


def test_cancel_queued_jobs(backend):
    backend.client.get_with_retry.side_effect = [
        MagicMock(json=lambda: {}),  # details()
        MagicMock(json=lambda: {"jobs": [{"id": "job1"}, {"id": "job2"}]}),
    ]

    sess = Session.from_id("sess-1", backend=backend)
    sess.cancel()

    backend.client.cancel_jobs.assert_called_once_with(["job1", "job2"])


def test_context_manager_closes_on_exception(backend):
    backend.client.post.return_value = {"id": "sess-789"}

    with pytest.raises(RuntimeError):
        with Session(backend=backend, max_jobs=1):
            raise RuntimeError("boom")

    # We only verify that an 'end' call happens; the creation call's payload can be anything.
    assert call("sessions", "sess-789", "end") in backend.client.post.call_args_list


@pytest.mark.parametrize(
    "create_new, session_id",
    [
        (False, None),
        (False, ""),
    ],
)
def test_invalid_init_raises(backend, create_new, session_id):
    with pytest.raises(ValueError):
        Session(backend=backend, create_new=create_new, session_id=session_id)


def test_backend_run_uses_session_id(backend):
    backend.client.post.return_value = {"id": "sess-42"}
    backend.run = MagicMock()

    with Session(backend=backend, max_jobs=1):
        backend.run("circ")

    assert backend.run.call_args.kwargs["session_id"] == "sess-42"


def test_session_run_uses_session_id(backend):
    backend.client.post.return_value = {"id": "sess-99"}
    backend.run = MagicMock(return_value="job-result")

    with Session(backend=backend, max_jobs=1) as sess:
        ret = sess.run("circ")  # invoke the new helper

    # The helper should propagate the return value unchanged.
    assert ret == "job-result"

    # And the wrapped backend.run must receive the session_id we created.
    backend.run.assert_called_once()
    assert backend.run.call_args.kwargs["session_id"] == "sess-99"
