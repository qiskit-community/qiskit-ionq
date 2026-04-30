# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
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

"""Tests for the new IonQSession class introduced in qiskit-ionq 1.1.0.

IonQSession is the migration target for the deprecated Session class. It wraps
ionq-core's SessionManager and is only importable when ionq-core is installed
(Python >=3.12). On older Pythons the entire module is skipped.
"""

from unittest.mock import MagicMock, patch

import pytest

# ionq-core requires Python >=3.12. Skip the module on older interpreters
# rather than failing at collect-time.
pytest.importorskip("ionq_core")

from qiskit_ionq import IonQSession  # noqa: E402


@pytest.fixture
def backend():
    bk = MagicMock()
    bk.name = "ionq_qpu.aria-1"
    bk._provider.credentials = {
        "token": "test-token",
        "url": "https://api.example.invalid/v0.4",
    }
    return bk


def test_init_builds_session_manager(backend):
    with patch("qiskit_ionq.session.SessionManager") as mock_mgr:
        sess = IonQSession(backend, max_jobs=5, max_time=60, max_cost=10.0)
    assert sess._backend is backend
    mock_mgr.assert_called_once()
    args, kwargs = mock_mgr.call_args
    assert args[1] == "qpu.aria-1"
    assert kwargs == {"max_jobs": 5, "max_time": 60, "max_cost": 10.0}


def test_from_id_reattaches_without_create(backend):
    with patch("qiskit_ionq.session.SessionManager") as mock_mgr:
        IonQSession.from_id(backend, "session-uuid-abc")
    mock_mgr.from_id.assert_called_once()
    _client, sid = mock_mgr.from_id.call_args.args
    assert sid == "session-uuid-abc"


def test_session_id_property_delegates(backend):
    with patch("qiskit_ionq.session.SessionManager"):
        sess = IonQSession(backend)
    sess._manager.session_id = "sess-1"
    assert sess.session_id == "sess-1"


def test_status_delegates(backend):
    with patch("qiskit_ionq.session.SessionManager"):
        sess = IonQSession(backend)
    sess._manager.status.return_value = "started"
    assert sess.status() == "started"


def test_context_manager_opens_and_closes(backend):
    with patch("qiskit_ionq.session.SessionManager"):
        sess = IonQSession(backend)
    with sess as s:
        assert s is sess
    sess._manager.open.assert_called_once()
    sess._manager.close.assert_called_once()


def test_close_delegates(backend):
    with patch("qiskit_ionq.session.SessionManager"):
        sess = IonQSession(backend)
    sess.close()
    sess._manager.close.assert_called_once()
