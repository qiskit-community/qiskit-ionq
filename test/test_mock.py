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

"""Test the test-infrastructure shim that maps the legacy ``requests_mock``
fixture API onto pytest-httpx now that all qiskit-ionq HTTP calls run through
ionq-core (and therefore through ``httpx``)."""

import unittest

import httpx
import pytest


def test_fixture_mock(requests_mock):
    """Test a function-scoped mock returns the registered text body.

    Args:
        requests_mock (_RequestsMockShim): The shim fixture (see ``test/conftest.py``).
    """
    requests_mock.get("https://www.google.com", text="function mock")
    response = httpx.get("https://www.google.com", timeout=30)
    assert response.text == "function mock"


class TestUnittestCompatibility(unittest.TestCase):
    """The legacy ``requests_mock`` shim should be wireable into a unittest
    ``TestCase`` via ``init_requests_mock``."""

    requests_mock = None

    @pytest.fixture(autouse=True)
    def init_requests_mock(self, requests_mock):
        """Initialize the shim mocker for this class.

        Args:
            requests_mock (_RequestsMockShim): The shim fixture from
                ``test/conftest.py``.
        """
        self.requests_mock = requests_mock

    def test_method_mock(self):
        """Test a method-scoped mock returns its registered body."""
        self.requests_mock.get(
            "https://www.google.com", text="instance method fixture mock"
        )
        response = httpx.get("https://www.google.com", timeout=30)
        self.assertEqual(response.text, "instance method fixture mock")

    def test_another_method_mock(self):
        """Test a second method-scoped mock returns its registered body."""
        self.requests_mock.get(
            "https://www.google.com",
            text="another instance method fixture mock",
        )
        response = httpx.get("https://www.google.com", timeout=30)
        self.assertEqual(response.text, "another instance method fixture mock")
