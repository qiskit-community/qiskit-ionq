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

"""global pytest fixtures"""
import json

import pytest
from qiskit_ionq_provider.ionq_client import IonQClient
from qiskit_ionq_provider.ionq_job import IonQJob
from qiskit_ionq_provider.ionq_provider import IonQProvider


class StubbedClient(IonQClient):
    """A mock client to use during testing.

    Args:
        IonQClient ([type]): [description]
    """

    def retrieve_job(self, job_id):
        header_dict = {
            "qubit_labels": [["q", 0], ["q", 1]],
            "n_qubits": 2,
            "qreg_sizes": [["q", 2]],
            "clbit_labels": [["c", 0], ["c", 1]],
            "memory_slots": 2,
            "creg_sizes": [["c", 2]],
            "name": "test-circuit",
            "global_phase": 0,
        }
        return {
            "status": "completed",
            "predicted_execution_time": 4,
            "metadata": {
                "shots": "1234",
                "qobj_id": "test_qobj_id",
                "output_length": "2",
                "output_map": '{"0": 1, "1": 0}',
                "header": json.dumps(header_dict),
            },
            "execution_time": 8,
            "qubits": 2,
            "type": "circuit",
            "request": 1600000000,
            "start": 1600000001,
            "response": 1600000002,
            "data": {"histogram": {"0": 0.5, "2": 0.499999}},
            "target": "qpu",
            "id": "test_id",
        }


@pytest.fixture(scope="class")
def provider(request):
    """Fixture for injecting a test provider into an object instance of a
    sub-class of ``unittest.TestCase``.

    Args:
        request (FixtureRequest): A pytest FixtureRequest.

    Returns:
        IonQProvider: A provider suitable for testing.
    """
    request.cls.provider = instance = IonQProvider("token", "url")
    return instance


@pytest.fixture(scope="class")
def qpu_backend(request, provider):
    """Get the QPU backend from a provider.

    Args:
        request (FixtureRequest): A pytest FixtureRequest.
        provider (IonQProvider): Injected provider from :meth:`provider`.

    Returns:
        IonQQPUBackend: An instance of an IonQQPUBackend.
    """
    request.cls.qpu_backend = instance = provider.get_backend("ionq_qpu")
    return instance


# pylint: disable=redefined-outer-name
@pytest.fixture(scope="class")
def formatted_result(request, provider):
    """Fixture for auto-injecting a formatted IonQJob result object into a
    a sub-class of ``unittest.TestCase``.

    Args:
        request (FixtureRequest): A pytest FixtureRequest.
        provider (IonQProvider): Injected provider from :meth:`provider`.

    Returns:
        Result: A qiskit result from making a fake API call with StubbedClient.
    """
    backend = provider.get_backend("ionq_qpu")
    job = IonQJob(backend, "test_id", StubbedClient())
    request.cls.formatted_result = instance = job.result()
    return instance
