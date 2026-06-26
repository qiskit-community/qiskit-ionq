# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
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

"""Tests for IonQClient.get_artifact content-type handling."""

from unittest.mock import patch, MagicMock

import pytest

from qiskit_ionq.ionq_client import IonQClient


def _response(content_type, *, text=None, content=None):
    res = MagicMock()
    res.status_code = 200
    res.headers = {"Content-Type": content_type}
    res.text = text
    res.content = content
    return res


@pytest.mark.parametrize(
    "content_type, text, content, expected",
    [
        # JSON -> order-preserving dict (native/ore/result artifacts)
        ("application/json", '{"qubits": 2}', None, {"qubits": 2}),
        # text/plain -> str (compiled ionq.qasm3.v1)
        ("text/plain; charset=utf-8", "OPENQASM 3.0;", None, "OPENQASM 3.0;"),
        # octet-stream -> bytes (ionq.mir.v1)
        ("application/octet-stream", None, b"\x08\x01", b"\x08\x01"),
        # unknown content type but valid JSON body -> dict
        ("", '{"a": 1}', None, {"a": 1}),
        # unknown content type, non-JSON body -> str (no crash)
        ("", "not json", None, "not json"),
    ],
)
def test_get_artifact_by_type(content_type, text, content, expected):
    """get_artifact decodes JSON, text, and binary artifacts by content type."""
    client = IonQClient("token", "https://api.example.com/v0.4")
    with patch(
        "requests.get", return_value=_response(content_type, text=text, content=content)
    ):
        assert client.get_artifact("job-id", "artifact-id") == expected
