# -*- coding: utf-8 -*-
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

"""Test basic exceptions behavior"""

from unittest import mock

import pytest

from qiskit_ionq_provider import exceptions


def test_base_str_and_repr():
    """Test basic str and repr support."""
    err = exceptions.IonQError()
    expected = "IonQError('')"
    assert str(err) == expected
    assert repr(err) == repr(expected)


def test_gate_error():
    """Test that IonQAPIError has specific instance attributes."""
    err = exceptions.IonQGateError("a gate")
    assert err.message == "gate 'a gate' not supported"


def test_gate_error_str_and_repr():
    """Test that IonQAPIError has a str/repr that includes args."""
    err = exceptions.IonQGateError("a gate")
    expected = "IonQGateError(gate_name='a gate')"
    assert str(err) == expected
    assert repr(err) == repr(expected)


def test_api_error():
    """Test that IonQAPIError has specific instance attributes."""
    err = exceptions.IonQAPIError("an error", 500, "internal_error")
    assert err.message == "an error"
    assert err.status_code == 500
    assert err.error_type == "internal_error"


def test_api_error_str_and_repr():
    """Test that IonQAPIError has a str/repr that includes args."""
    err = exceptions.IonQAPIError("an error", 500, "internal_error")
    expected = "IonQAPIError(message='an error',status_code=500,error_type='internal_error')"
    assert str(err) == expected
    assert repr(err) == repr(expected)


def test_api_error_from_response():
    """Test that IonQAPIError can be made directly from a response JSON dict."""
    fake_response = mock.MagicMock()
    fake_response.json = mock.MagicMock(
        return_value={
            "error": {
                "type": "internal_error",
                "message": "an error",
            }
        }
    )
    fake_response.status_code = 500

    with pytest.raises(exceptions.IonQAPIError) as exc:
        exceptions.IonQAPIError.from_response(fake_response)

    err = exc.value
    assert err.message == "an error"
    assert err.status_code == 500
    assert err.error_type == "internal_error"
