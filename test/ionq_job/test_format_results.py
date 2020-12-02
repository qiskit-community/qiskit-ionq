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

"""Test the format_results function"""
import unittest

import pytest
from qiskit.qobj import QobjExperimentHeader


@pytest.mark.usefixtures("formatted_result")
class TestResultsFormatter(unittest.TestCase):
    """Test formatted result content.

    Attributes:
        formatted_result (Result): An injected formatted job result.
    """

    formatted_result = None

    @pytest.fixture(autouse=True)
    def mock_client_response(self, requests_mock):
        self.requests_mock = requests_mock

    def test_results_meta(self):
        """Test basic job attribute values."""
        self.assertEqual(self.formatted_result.backend_name, "ionq_qpu")
        self.assertEqual(self.formatted_result.backend_version, "0.0.1")
        self.assertEqual(self.formatted_result.qobj_id, "test_qobj_id")
        self.assertEqual(self.formatted_result.job_id, "test_id")
        self.assertEqual(self.formatted_result.success, True)

    def test_counts(self):
        """Test counts based on test.conftest.StubbedClient"""
        counts = self.formatted_result.get_counts()
        self.assertEqual({"00": 617, "01": 617}, counts)

    def test_additional_results_details(self):
        """Test shots and headers in the result data."""
        results = self.formatted_result.results[0]
        self.assertEqual(results.shots, "1234")
        self.assertEqual(results.success, True)
        self.assertEqual(
            results.header,
            QobjExperimentHeader.from_dict(
                {
                    "qubit_labels": [["q", 0], ["q", 1]],
                    "n_qubits": 2,
                    "qreg_sizes": [["q", 2]],
                    "clbit_labels": [["c", 0], ["c", 1]],
                    "memory_slots": 2,
                    "creg_sizes": [["c", 2]],
                    "name": "test-circuit",
                    "global_phase": 0,
                }
            ),
        )
