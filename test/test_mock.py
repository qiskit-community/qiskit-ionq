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

"""Test basic request_mock behavior."""

import unittest

import pytest
import requests
from requests_mock import adapter as rm_adapter


def test_global_mock():
    """test the global requests mock"""
    response = requests.get("https://www.google.com")
    assert response.text == "UNHANDLED REQUEST. PLEASE MOCK WITH requests_mock."


def test_fixture_mock(requests_mock):
    """Test a function-scoped mock overrides the global.

    Args:
        requests_mock (:class:`requests_mock.Mocker`): A requests mocker.
    """
    requests_mock.get("https://www.google.com", text="function mock")
    response = requests.get("https://www.google.com")
    assert response.text == "function mock"


class TestUnittestCompatibility(unittest.TestCase):
    """An example of how to use `requests_mock` with a class.

    Attributes:
        requests_mock (:class:`requests_mock.Mocker`): A requests mocker.
    """

    requests_mock = None

    @pytest.fixture(autouse=True)
    def init_requests_mock(self, requests_mock):
        """Initialize a :class:`requests_mock.Mocker` for this class.

        .. NOTE::
           Because this is cached on a class, it means all requests made
           from any methods on this class will use the same registered mocks.

        Args:
            requests_mock (:class:`requests_mock.Mocker`):
                A requests_mock fixture.
        """
        self.requests_mock = requests_mock

        # Register any URIs you want to mock for the entire class here:
        self.requests_mock.register_uri(
            rm_adapter.ANY,
            rm_adapter.ANY,
            response_list=[
                {
                    "status_code": 599,
                    "text": "class fixture mock",
                }
            ],
        )

    def test_class_mock(self):
        """
        Test the class-scoped requests mock, which is setup
        in :meth:`init_requests_mock`
        """
        response = requests.get("https://www.google.com")
        self.assertEqual(response.text, "class fixture mock")

    def test_method_mock(self):
        """Test a method-scoped mock overrides the global."""
        self.requests_mock.get("https://www.google.com", text="instance method fixture mock")
        response = requests.get("https://www.google.com")
        self.assertEqual(response.text, "instance method fixture mock")

    def test_another_method_mock(self):
        """Test a second method-scoped mock overrides the global."""
        self.requests_mock.get(
            "https://www.google.com",
            text="another instance method fixture mock",
        )
        response = requests.get("https://www.google.com")
        self.assertEqual(response.text, "another instance method fixture mock")
