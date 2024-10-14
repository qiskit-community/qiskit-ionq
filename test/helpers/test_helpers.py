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

"""Test the helper functions."""

import re
from unittest.mock import patch, MagicMock
from qiskit_ionq.ionq_client import IonQClient
from qiskit_ionq.helpers import get_n_qubits


def test_user_agent_header():
    """
    Tests whether the generated user_agent contains all the required information with the right
    version format.
    """
    ionq_client = IonQClient()
    generated_user_agent = ionq_client.api_headers["User-Agent"]

    user_agent_info_keywords = ["qiskit-ionq", "qiskit-terra", "os", "python"]
    # Checks if all keywords are present in user-agent string.
    all_user_agent_keywords_avail = all(
        keyword in generated_user_agent for keyword in user_agent_info_keywords
    )

    # Checks whether there is at-least 3 version strings from qiskit-ionq, qiskit-terra, python.
    has_all_version_strings = len(re.findall(r"\s*([\d.]+)", generated_user_agent)) >= 3
    assert all_user_agent_keywords_avail and has_all_version_strings


def test_get_n_qubits_success():
    """Test get_n_qubits returns correct number of qubits and checks correct URL."""
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"qubits": 11}
        mock_get.return_value = mock_response

        backend = "ionq_qpu.aria-1"
        result = get_n_qubits(backend)

        expected_url = (
            "https://api.ionq.co/v0.3/characterizations/backends/aria-1/current"
        )

        # Create a regular expression to match the Authorization header with an apiKey
        expected_headers = {"Authorization": re.compile(r"apiKey\s+\S+")}

        # Check the arguments of the last call to `requests.get`
        mock_get.assert_called()
        _, kwargs = mock_get.call_args
        assert (
            kwargs["url"] == expected_url
        ), f"Expected URL {expected_url}, but got {kwargs['url']}"

        # Assert that the headers contain the apiKey in the expected format
        assert re.match(
            expected_headers["Authorization"], kwargs["headers"]["Authorization"]
        ), (
            f"Expected headers to match {expected_headers['Authorization'].pattern}, "
            f"but got {kwargs['headers']['Authorization']}"
        )

        assert result == 11, f"Expected 11 qubits, but got {result}"


def test_get_n_qubits_fallback():
    """Test get_n_qubits returns fallback number of qubits and checks correct URL on failure."""
    with patch("requests.get", side_effect=Exception("Network error")) as mock_get:
        backend = "aria-1"
        result = get_n_qubits(backend)

        expected_url = (
            "https://api.ionq.co/v0.3/characterizations/backends/aria-1/current"
        )

        # Create a regular expression to match the Authorization header with an apiKey
        expected_headers = {"Authorization": re.compile(r"apiKey\s+\S+")}

        # Check the arguments of the last call to `requests.get`
        mock_get.assert_called()
        _, kwargs = mock_get.call_args
        assert (
            kwargs["url"] == expected_url
        ), f"Expected URL {expected_url}, but got {kwargs['url']}"

        # Assert that the headers contain the apiKey in the expected format
        assert re.match(
            expected_headers["Authorization"], kwargs["headers"]["Authorization"]
        ), (
            f"Expected headers to match {expected_headers['Authorization'].pattern}, "
            f"but got {kwargs['headers']['Authorization']}"
        )

        assert result == 100, f"Expected fallback of 100 qubits, but got {result}"
