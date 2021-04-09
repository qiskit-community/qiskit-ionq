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

"""Test basic behavior of :class:`IonQJob`."""
from unittest import mock

import pytest
from qiskit import QuantumCircuit
from qiskit.providers import exceptions as q_exc
from qiskit.providers import jobstatus

from qiskit.qobj.utils import MeasLevel
from qiskit_ionq import exceptions, ionq_job
from qiskit_ionq.helpers import compress_dict_to_metadata_string

from .. import conftest


def spy(instance, attr):
    """Spy an attribute on an instance.

    This is just a short-cut function for creating a "spy" mock patch.

    The patch can be used to test properites of the attribute in question
    within a context manager.

    For example::

        my_obj = SomeClass()
        with spy(my_obj, "a_method") as method_spy:
            my_obj.do_something()

        method_spy.assert_called_once()
        ...

    Args:
        instance (object): An instance of an object.
        attr (str): The attribute name to spy on.

    Returns:
        mock.MagicMock: A mock object that will spy on ``attr``.
    """
    actual_attr = getattr(instance, attr)
    patch = mock.patch.object(
        instance,
        attr,
        wraps=actual_attr,
    )
    return patch


@pytest.mark.parametrize(
    "data,error_msg",
    [
        (None, "Cannot remap counts without an API response!"),
        ({}, "Cannot remap counts without an API response!"),
        ({"anything": "anything"}, "Cannot remap counts without qubits!"),
        ({"qubits": 2}, "Cannot remap counts without metadata!"),
        ({"qubits": 2, "metadata": {}}, "Cannot remap counts without result data!"),
    ],
)
def test_build_counts__bad_input(data, error_msg):
    """Test that _build_counts raises specific exceptions based on provided input.

    Args:
        data (dict): A dict that will trigger known exception cases.
        error_msg (str): The expected error message based on ``data``.
    """
    with pytest.raises(exceptions.IonQJobError) as exc_info:
        ionq_job._build_counts(data)
    assert exc_info.value.message == error_msg


def test_build_counts():
    """Test basic count remapping."""
    result = {
        "qubits": 3,
        "data": {
            "histogram": {"5": 0.5, "7": 0.5},
            "registers": {
                "meas_mapped": {
                    "3": 0.5,
                    "7": 0.5,
                }
            },
        },
        "metadata": {
            "shots": "100",
            "qiskit_header": compress_dict_to_metadata_string({"memory_slots": 3}),
        },
    }
    (counts, probabilties) = ionq_job._build_counts(result)
    assert ({"0x3": 50, "0x7": 50}) == counts
    assert ({"0x3": 0.5, "0x7": 0.5}) == probabilties


def test_results_meta(formatted_result):
    """Test basic job attribute values."""
    assert formatted_result.backend_name == "ionq_qpu"
    assert formatted_result.backend_version == "0.0.1"
    assert formatted_result.qobj_id == "test_qobj_id"
    assert formatted_result.job_id == "test_id"
    assert formatted_result.success is True
    assert formatted_result.time_taken == 0.008


def test_counts(formatted_result):
    """Test counts based on a dummy result (see global conftest.py)."""
    counts = formatted_result.get_counts()
    assert {"01": 617, "11": 617} == counts


def test_probabilities(formatted_result):
    """Test counts based on a dummy result (see global conftest.py)."""
    probabilities = formatted_result.get_probabilities()
    assert {"01": 0.499999, "11": 0.5} == probabilities


def test_counts__simulator_probs(simulator_backend, requests_mock):
    """Test that the simulator uses the sampler to produce counts and probs"""
    # Dummy job ID for formatted results fixture.
    job_id = "test_id"

    # Create the request path for accessing the dummy job:
    path = simulator_backend.client.make_path("jobs", job_id)
    requests_mock.get(path, json=conftest.dummy_job_response(job_id))
    job = ionq_job.IonQJob(simulator_backend, job_id)

    formatted_result = job.result()
    counts = formatted_result.get_counts()
    probabilities = formatted_result.get_probabilities()
    assert {"01": 609, "11": 625} == counts
    assert {"01": 0.499999, "11": 0.5} == probabilities


def test_counts_and_probs_from_job(simulator_backend, requests_mock):
    """Test that the helper methods on the job return the same data as the methods on the result"""
    # Dummy job ID for formatted results fixture.
    job_id = "test_id"

    # Create the request path for accessing the dummy job:
    path = simulator_backend.client.make_path("jobs", job_id)
    requests_mock.get(path, json=conftest.dummy_job_response(job_id))
    job = ionq_job.IonQJob(simulator_backend, job_id)

    counts = job.get_counts()
    probabilities = job.get_probabilities()
    assert {"01": 609, "11": 625} == counts
    assert {"01": 0.499999, "11": 0.5} == probabilities


def test_submit__without_circuit(mock_backend, requests_mock):
    """Test the behavior of attempting to submit a job with no circuit.

    Args:
        mock_backend (MockBackend): A fake/mock IonQBackend.
        requests_mock (:class:`request_mock.Mocker`): A requests mocker.
    """
    job_id = "test_id"

    # Mock the initial status call.
    fetch_path = mock_backend.client.make_path("jobs", job_id)
    requests_mock.get(
        fetch_path,
        status_code=200,
        json=conftest.dummy_job_response(job_id),
    )

    # Create the job (this calls .status())
    job = ionq_job.IonQJob(mock_backend, job_id)
    with pytest.raises(exceptions.IonQJobError) as exc_info:
        job.submit()

    expected_err = (
        "Cannot submit a job without a circuit. "
        "Please create a job with a circuit and try again."
    )
    assert exc_info.value.message == expected_err


def test_submit(mock_backend, requests_mock):
    """Test that job submission calls the IonQ API via the job's client.

    Args:
        mock_backend (MockBackend): A fake/mock IonQBackend.
        requests_mock (:class:`request_mock.Mocker`): A requests mocker.]
    """
    client = mock_backend.client

    # Mock the initial status call.
    fetch_path = mock_backend.client.make_path("jobs")
    requests_mock.post(
        fetch_path,
        status_code=200,
        json=conftest.dummy_job_response("server_job_id"),
    )

    # Create a job ref (this does not call status, since circuit is not None).
    job = ionq_job.IonQJob(mock_backend, None, circuit=QuantumCircuit(1, 1))
    with spy(client, "submit_job") as submit_spy:
        job.submit()

    # Validate the job was submitted to the API client.
    submit_spy.assert_called_with(job=job)
    assert job._job_id == "server_job_id"


def test_cancel(mock_backend, requests_mock):
    """Test cancelling the job will use a client to cancel the job via the API.

    Args:
        mock_backend (MockBackend): A fake/mock IonQBackend.
        requests_mock (:class:`request_mock.Mocker`): A requests mocker.
    """
    # Mock the initial status call.
    job_id = "test_id"
    client = mock_backend.client
    fetch_path = client.make_path("jobs", job_id)
    requests_mock.get(
        fetch_path,
        status_code=200,
        json=conftest.dummy_job_response(job_id),
    )

    # Mock a request to cancel.
    cancel_path = client.make_path("jobs", job_id, "status", "cancel")
    requests_mock.put(cancel_path, status_code=200, json={})

    # Create the job:
    job = ionq_job.IonQJob(mock_backend, job_id)

    # Tell the job to cancel.
    with spy(mock_backend.client, "cancel_job") as cancel_spy:
        job.cancel()

    # Verify that the API was called to cancel the job.
    cancel_spy.assert_called_with(job_id)


def test_result__timeout(mock_backend, requests_mock):
    """Test that timeouts are re-raised as IonQJobTimeoutErrors.

    Args:
        mock_backend (MockBackend): A fake/mock IonQBackend.
        requests_mock (:class:`request_mock.Mocker`): A requests mocker.
    """

    # Create the job:
    job_id = "test_id"
    client = mock_backend.client
    job_result = conftest.dummy_job_response(job_id)
    job_result.update({"status": "submitted"})

    # Mock the job response API call.
    path = client.make_path("jobs", job_id)
    requests_mock.get(path, status_code=200, json=job_result)

    # Create a job ref (this will call .status() to fetch our mock above)
    job = ionq_job.IonQJob(mock_backend, job_id)

    # Patch `wait_for_final_state` to force throwing a timeout.
    exc_patch = mock.patch.object(job, "wait_for_final_state", side_effect=q_exc.JobTimeoutError())

    # Use the patch, then expect `result` to raise out.
    with exc_patch, pytest.raises(exceptions.IonQJobTimeoutError) as exc_info:
        job.result()
    assert exc_info.value.message == "Timed out waiting for job to complete."


expected_result = {
    "backend_name": "ionq_mock_backend",
    "backend_version": "0.0.1",
    "job_id": "test_id",
    "qobj_id": "test_qobj_id",
    "results": [
        {
            "data": {
                "counts": {"0x1": 617, "0x3": 617},
                "probabilities": {"0x1": 0.499999, "0x3": 0.5},
            },
            "header": {
                "clbit_labels": [["c", 0], ["c", 1]],
                "creg_sizes": [["c", 2]],
                "global_phase": 0,
                "memory_slots": 2,
                "n_qubits": 2,
                "name": "test_id",
                "qreg_sizes": [["q", 2]],
                "qubit_labels": [["q", 0], ["q", 1]],
            },
            "meas_level": MeasLevel.CLASSIFIED,
            "shots": 1234,
            "success": True,
        }
    ],
    "success": True,
    "time_taken": 0.008,
}
# Validate the result
def test_result(mock_backend, requests_mock):
    """Test basic "happy path" for result fetching.

    Args:
        mock_backend (MockBackend): A fake/mock IonQBackend.
        requests_mock (:class:`request_mock.Mocker`): A requests mocker.
    """
    # Create the job:
    job_id = "test_id"
    client = mock_backend.client
    job_result = conftest.dummy_job_response(job_id)

    # Mock the job response API call.
    path = client.make_path("jobs", job_id)
    requests_mock.get(path, status_code=200, json=job_result)

    # Create a job ref (this will call .status() to fetch our mock above)
    job = ionq_job.IonQJob(mock_backend, job_id)

    assert job.result().to_dict() == expected_result


def test_result__from_circuit(mock_backend, requests_mock):
    """Test result fetching when the job did not already exist at creation time.

    Args:
        mock_backend (MockBackend): A fake/mock IonQBackend.
        requests_mock (:class:`request_mock.Mocker`): A requests mocker.
    """
    # Create the job:
    job_id = "test_id"
    job_result = conftest.dummy_job_response(job_id)
    client = mock_backend.client

    # Create a job ref (this does not call status, since circuit is not None).
    job = ionq_job.IonQJob(mock_backend, None, circuit=QuantumCircuit(1, 1))

    # Mock the create:
    create_path = client.make_path("jobs")
    requests_mock.post(create_path, status_code=200, json=job_result)

    # Submit the job.
    job.submit()

    # Mock the fetch from `result`, since status should still be "initializing".
    fetch_path = client.make_path("jobs", job_id)
    requests_mock.get(fetch_path, status_code=200, json=job_result)

    # Validate the result and its format. should be the same as base case.
    assert job.result().to_dict() == expected_result


def test_result__failed_from_api(mock_backend, requests_mock):
    """Test result fetching when the job fails on the API side (e.g. due to bad input)

    Args:
        mock_backend (MockBackend): A mock IonQBackend.
        requests_mock (:class:`request_mock.Mocker`): A requests mocker.
    """
    # Create the job:
    job_id = "test_id"
    job_result = conftest.dummy_failed_job(job_id)
    client = mock_backend.client

    # Create a job ref (this won't call status, since circuit is not None).
    job = ionq_job.IonQJob(mock_backend, None, circuit=QuantumCircuit(1, 1))

    # Mock the create:
    create_path = client.make_path("jobs")
    requests_mock.post(create_path, status_code=200, json=job_result)

    # Submit the job.
    job.submit()

    # Mock the fetch from `result`, since status should still be "initializing".
    fetch_path = client.make_path("jobs", job_id)
    requests_mock.get(fetch_path, status_code=200, json=job_result)

    with pytest.raises(exceptions.IonQJobFailureError) as exc:
        job.result()
    # assert fails
    assert 'Failure from IonQ API "ExampleError: example error"' in str(exc.value)


def test_result__cancelled(mock_backend, requests_mock):
    """Test result fetching when the job fails on the API side (e.g. due to bad input)

    Args:
        mock_backend (MockBackend): A mock IonQBackend.
        requests_mock (:class:`request_mock.Mocker`): A requests mocker.
    """
    # Create the job:
    job_id = "test_id"
    job_result = conftest.dummy_job_response(job_id, "canceled")
    client = mock_backend.client

    # Create a job ref (this won't call status, since circuit is not None).
    job = ionq_job.IonQJob(mock_backend, None, circuit=QuantumCircuit(1, 1))

    # Mock the create:
    create_path = client.make_path("jobs")
    requests_mock.post(create_path, status_code=200, json=job_result)

    # Submit the job.
    job.submit()

    # Mock the fetch from `result`, since status should still be "initializing".
    fetch_path = client.make_path("jobs", job_id)
    requests_mock.get(fetch_path, status_code=200, json=job_result)

    with pytest.raises(exceptions.IonQJobStateError) as exc:
        job.result()
    # assert fails
    assert 'Job was cancelled"' in str(exc.value)


def test_status__no_job_id(mock_backend):
    """Test status() returns early when the job has not yet been created.

    Args:
        mock_backend (MockBackend): A fake/mock IonQBackend.
    """
    # Create a job:

    # Create a job ref (this does not call status, since circuit is not None).
    job = ionq_job.IonQJob(mock_backend, None, circuit=QuantumCircuit(1, 1))
    assert job._status == jobstatus.JobStatus.INITIALIZING

    # Call status:
    with spy(mock_backend.client, "retrieve_job") as job_fetch_spy:
        actual_status = job.status()

    job_fetch_spy.assert_not_called()
    assert actual_status is job._status is jobstatus.JobStatus.INITIALIZING


def test_status__already_final_state(mock_backend, requests_mock):  # pylint: disable=invalid-name
    """Test status() returns early when the job is already completed.

    Args:
        mock_backend (MockBackend): A fake/mock IonQBackend.
        requests_mock (:class:`request_mock.Mocker`): A requests mocker.
    """
    # Create a job:
    job_id = "test_id"
    job_result = conftest.dummy_job_response(job_id, status="completed")
    client = mock_backend.client

    # Mock the fetch from `result`, since status should still be "initializing".
    fetch_path = client.make_path("jobs", job_id)
    requests_mock.get(fetch_path, status_code=200, json=job_result)

    # Create a job ref (this does not call status, since circuit is not None).
    job = ionq_job.IonQJob(mock_backend, "test_id")

    # Call status:
    # fmt: off
    #import pdb; pdb.set_trace()
    with spy(client, "retrieve_job") as job_fetch_spy:
        actual_status = job.status()

    job_fetch_spy.assert_not_called()
    assert actual_status is job._status is jobstatus.JobStatus.DONE
