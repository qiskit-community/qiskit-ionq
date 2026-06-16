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
import warnings

import pytest
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers import exceptions as q_exc
from qiskit.providers import jobstatus

from qiskit_ionq import exceptions, ionq_job
from qiskit_ionq._qiskit_compat import MEAS_LEVEL_CLASSIFIED
from qiskit_ionq.helpers import compress_to_metadata_string
from qiskit_ionq.ionq_result import IonQResult


from .. import conftest


def _mock_job_payload(job_id, qiskit_header, circuits, children=None, qubits=3):
    """Return the minimal JSON payload that IonQJob.status() expects."""
    return {
        "id": job_id,
        "status": "completed",
        "stats": {"qubits": qubits, "circuits": circuits},
        "children": (
            children
            if children is not None
            else (
                [f"{job_id}_child_{i}" for i in range(circuits)]
                if circuits > 1
                else None
            )
        ),
        # parent-level header contains all circuits
        "metadata": {"qiskit_header": compress_to_metadata_string(qiskit_header)},
        # dummy results URL so status() sets _results_url without error
        "results": {
            "probabilities": {"url": f"/jobs/{job_id}/results/probabilities"},
            "histogram": {"url": f"/jobs/{job_id}/results/histogram"},
        },
    }


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
    "qubits, histogram, clbits, mapped_histogram",
    [
        (2, {0: 0.499, 3: 0.499}, [0], {0: 0.499, 1: 0.499}),
        (2, {0: 0.499, 3: 0.499}, [1], {0: 0.499, 1: 0.499}),
        (2, {0: 0.499, 3: 0.499}, [0, 1], {0: 0.499, 3: 0.499}),
        (2, {0: 0.499, 3: 0.499}, [None, 0, 1], {0: 0.499, 6: 0.499}),
        (2, {0: 0.499, 3: 0.499}, [0, 1, None], {0: 0.499, 3: 0.499}),
        (4, {3: 0.99}, [0, 1, 2, 3], {3: 0.99}),
        (
            3,
            {0: 0.4999999999999999, 3: 0.4999999999999999},
            [0, 1, None],
            {0: 0.4999999999999999, 3: 0.4999999999999999},
        ),
        (
            2,
            {
                0: 0.2499999999999999,
                1: 0.2499999999999999,
                2: 0.2499999999999999,
                3: 0.2499999999999999,
            },
            [0, 1],
            {
                0: 0.2499999999999999,
                1: 0.2499999999999999,
                2: 0.2499999999999999,
                3: 0.2499999999999999,
            },
        ),
        (
            4,
            {
                0: 0.0625,
                1: 0.0625,
                2: 0.0625,
                3: 0.0625,
                4: 0.0625,
                5: 0.0625,
                6: 0.0625,
                7: 0.0625,
                8: 0.0625,
                9: 0.0625,
                10: 0.0625,
                11: 0.0625,
                12: 0.0625,
                13: 0.0625,
                14: 0.0625,
                15: 0.0625,
            },
            [0, 1, 2, 3],
            {
                0: 0.0625,
                1: 0.0625,
                2: 0.0625,
                3: 0.0625,
                4: 0.0625,
                5: 0.0625,
                6: 0.0625,
                7: 0.0625,
                8: 0.0625,
                9: 0.0625,
                10: 0.0625,
                11: 0.0625,
                12: 0.0625,
                13: 0.0625,
                14: 0.0625,
                15: 0.0625,
            },
        ),
        (
            4,
            {
                0: 0.2499999999999999,
                1: 0.2499999999999999,
                2: 0.2499999999999999,
                3: 0.2499999999999999,
            },
            [0, 1, None, None],
            {
                0: 0.2499999999999999,
                1: 0.2499999999999999,
                2: 0.2499999999999999,
                3: 0.2499999999999999,
            },
        ),
        (
            2,
            {
                0: 0.2499999999999999,
                1: 0.2499999999999999,
                2: 0.2499999999999999,
                3: 0.2499999999999999,
            },
            [0, 0],
            {
                0: 0.4999999999999998,
                3: 0.4999999999999998,
            },
        ),
        (2, {0: 0.499, 3: 0.499}, [], {}),
    ],
)
def test_map_output(histogram, clbits, qubits, mapped_histogram):
    """Test that map_output maps histogram with classical register correctly"""
    assert mapped_histogram == ionq_job.map_output(histogram, clbits, qubits)


def test_build_counts__bad_input():
    """Test that _build_counts raises specific exceptions based on provided input."""
    with pytest.raises(exceptions.IonQJobError) as exc_info:
        ionq_job._build_counts(None, 1, [], 100)
    assert exc_info.value.message == "Cannot remap counts without data!"


def test_build_counts():
    """Test basic count remapping."""
    (counts, probabilties) = ionq_job._build_counts(
        {"5": 0.5, "7": 0.5}, 3, [0, 1, 2], 100
    )
    assert ({"101": 50, "111": 50}) == counts
    assert ({"101": 0.5, "111": 0.5}) == probabilties


def test_results_meta(formatted_result):
    """Test basic job attribute values."""
    assert formatted_result.backend_name.startswith("ionq_qpu")
    assert formatted_result.backend_version == "0.0.1"
    assert formatted_result.qobj_id == "test_qobj_id"
    assert formatted_result.job_id == "test_id"
    assert formatted_result.success is True
    assert formatted_result.time_taken == 0.008


def test_counts(formatted_result):
    """Test counts based on a dummy result (see global conftest.py)."""
    counts = formatted_result.get_counts()
    assert {"00": 617, "10": 617} == counts


def test_probabilities(formatted_result):
    """Test counts based on a dummy result (see global conftest.py)."""
    probabilities = formatted_result.get_probabilities()
    assert {"00": 0.5, "10": 0.499999} == probabilities


def test_counts__simulator_probs(simulator_backend, requests_mock):
    """Test that the simulator uses the sampler to produce counts and probs"""
    # Dummy job ID for formatted results fixture.
    job_id = "test_id"

    # Create the request path for accessing the dummy job:
    path = simulator_backend.client.make_path("jobs", job_id)
    requests_mock.get(path, json=conftest.dummy_job_response(job_id))

    results_path = simulator_backend.client.make_path(
        "jobs", job_id, "results", "probabilities"
    )
    requests_mock.get(results_path, json={"0": 0.5, "2": 0.499999})
    job = ionq_job.IonQJob(simulator_backend, job_id)

    formatted_result = job.result()
    counts = formatted_result.get_counts()
    probabilities = formatted_result.get_probabilities()

    assert {"00": 609, "10": 625} == counts
    assert {"00": 0.5, "10": 0.499999} == probabilities


def test_build_counts__with_int():
    """Test that a result with an integer doesn't break everything."""
    counts, probabilties = ionq_job._build_counts(
        {"1": 1}, 1, [0], 100, use_sampler=True, sampler_seed=42
    )
    assert ({"1": 100}) == counts
    assert ({"1": 1.0}) == probabilties


def test_counts_and_probs_from_job(simulator_backend, requests_mock):
    """Test that the helper methods on the job return the same data as the methods on the result"""
    # Dummy job ID for formatted results fixture.
    job_id = "test_id"

    # Create the request path for accessing the dummy job:
    path = simulator_backend.client.make_path("jobs", job_id)
    requests_mock.get(path, json=conftest.dummy_job_response(job_id))
    results_path = simulator_backend.client.make_path(
        "jobs", job_id, "results", "probabilities"
    )
    requests_mock.get(results_path, status_code=200, json={"0": 0.5, "2": 0.499999})
    job = ionq_job.IonQJob(simulator_backend, job_id)

    counts = job.get_counts()
    probabilities = job.get_probabilities()
    assert {"00": 609, "10": 625} == counts
    assert {"00": 0.5, "10": 0.499999} == probabilities


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
    job_result.update({"status": "submitted", "warning": {"messages": ["TimedOut"]}})

    # Mock the job response API call.
    path = client.make_path("jobs", job_id)
    requests_mock.get(path, status_code=200, json=job_result)

    # Create a job ref (this will call .status() to fetch our mock above)
    job = ionq_job.IonQJob(mock_backend, job_id)

    # Patch `wait_for_final_state` to force throwing a timeout.
    exc_patch = mock.patch.object(
        job, "wait_for_final_state", side_effect=q_exc.JobTimeoutError()
    )

    # Use the patch, then expect `result` to raise out.
    with (
        exc_patch,
        pytest.raises(exceptions.IonQJobTimeoutError) as exc_info,
        warnings.catch_warnings(record=True) as w,
    ):
        job.result()
        assert len(w) == 1
        assert "TimedOut" in str(w[0].message)
    assert exc_info.value.message == "Timed out waiting for job to complete."


expected_result = {
    "backend_name": "ionq_mock_backend",
    "backend_version": "0.0.1",
    "job_id": "test_id",
    "qobj_id": "test_qobj_id",
    "results": [
        {
            "data": {
                "counts": {"00": 617, "10": 617},
                "probabilities": {"00": 0.5, "10": 0.499999},
                "metadata": {
                    "clbit_labels": [["c", 0], ["c", 1]],
                    "creg_sizes": [["c", 2]],
                    "global_phase": 0,
                    "memory_slots": 2,
                    "n_qubits": 2,
                    "name": "test_id",
                    "qreg_sizes": [["q", 2]],
                    "qubit_labels": [["q", 0], ["q", 1]],
                },
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
            "meas_level": MEAS_LEVEL_CLASSIFIED,
            "shots": 1234,
            "success": True,
        }
    ],
    "success": True,
    "time_taken": 0.008,
    "status": None,
    "date": None,
    "header": None,
}


# Validate the result
def test_result_from_dict_headers():
    """IonQResult.from_dict normalizes headers for Qiskit v1 and v2."""
    result = IonQResult.from_dict(expected_result)

    assert result.results[0].header["memory_slots"] == 2
    assert result.to_dict() == expected_result


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

    results_path = client.make_path("jobs", job_id, "results", "probabilities")
    requests_mock.get(results_path, status_code=200, json={"0": 0.5, "2": 0.499999})

    # Create a job ref (this will call .status() to fetch our mock above)
    job = ionq_job.IonQJob(mock_backend, job_id)

    assert job.result().to_dict() == expected_result


def test_result__with_sharpen(mock_backend, requests_mock):
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

    results_path = (
        client.make_path("jobs", job_id, "results", "probabilities") + "?sharpen=false"
    )
    requests_mock.get(results_path, status_code=200, json={"0": 0.5, "2": 0.499999})

    # Create a job ref (this will call .status() to fetch our mock above)
    job = ionq_job.IonQJob(mock_backend, job_id)

    assert job.result(sharpen=False).to_dict() == expected_result


def test_result__with_extra_payload(mock_backend, requests_mock):
    """Test result allows for arbitrary query parameters

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

    results_path = (
        client.make_path("jobs", job_id, "results", "probabilities") + "?sharpen=false"
    )
    requests_mock.get(results_path, status_code=200, json={"0": 0.5, "2": 0.499999})

    # Create a job ref (this will call .status() to fetch our mock above)
    job = ionq_job.IonQJob(mock_backend, job_id)

    assert (
        job.result(extra_query_params={"sharpen": False}).to_dict() == expected_result
    )


def test_result__bad_sharpen(mock_backend, requests_mock):
    """Test basic "happy path" for result fetching.

    Args:
        mock_backend (MockBackend): A fake/mock IonQBackend.
        requests_mock (:class:`request_mock.Mocker`): A requests mocker.
    """
    # Create the job:
    job_id = "test_id"
    job_response = conftest.dummy_job_response(job_id)
    client = mock_backend.client

    # Mock the fetch from `result`, since status should still be "initializing".
    fetch_path = client.make_path("jobs", job_id)
    requests_mock.get(fetch_path, status_code=200, json=job_response)

    results_path = client.make_path("jobs", job_id, "results", "probabilities")
    requests_mock.get(results_path, status_code=200, json={"0": 0.5, "2": 0.499999})

    # Create a job ref (this will call .status() to fetch our mock above)
    job = ionq_job.IonQJob(mock_backend, job_id)

    with pytest.warns(UserWarning, match="Invalid sharpen type"):
        job.result(sharpen="blah")


def test_result__from_circuit(mock_backend, requests_mock):
    """Test result fetching when the job did not already exist at creation time.

    Args:
        mock_backend (MockBackend): A fake/mock IonQBackend.
        requests_mock (:class:`request_mock.Mocker`): A requests mocker.
    """
    # Create the job:
    job_id = "test_id"
    job_response = conftest.dummy_job_response(job_id)
    client = mock_backend.client

    # Create a job ref (this does not call status, since circuit is not None).
    job = ionq_job.IonQJob(mock_backend, None, circuit=QuantumCircuit(1, 1))

    # Mock the create:
    create_path = client.make_path("jobs")
    requests_mock.post(create_path, status_code=200, json=job_response)

    # Submit the job.
    job.submit()

    # Mock the fetch from `result`, since status should still be "initializing".
    fetch_path = client.make_path("jobs", job_id)
    requests_mock.get(fetch_path, status_code=200, json=job_response)

    results_path = client.make_path("jobs", job_id, "results", "probabilities")
    requests_mock.get(results_path, status_code=200, json={"0": 0.5, "2": 0.499999})

    # Validate the result and its format. should be the same as base case.
    res = job.result().to_dict()
    assert res == expected_result


def test_result__meas_mapped(mock_backend, requests_mock):
    """Test result fetching when the job has uses meas_mapped to remap results.

    Args:
        mock_backend (MockBackend): A fake/mock IonQBackend.
        requests_mock (:class:`request_mock.Mocker`): A requests mocker.
    """
    # Create the job:
    job_id = "test_id"
    job_response = conftest.dummy_mapped_job_response(job_id)
    client = mock_backend.client

    meas_mapped = QuantumCircuit(2, 2, name="meas_mapped")
    meas_mapped.x(0)
    meas_mapped.measure([0, 1], [1, 0])

    # Create a job ref (this does not call status, since circuit is not None).
    job = ionq_job.IonQJob(mock_backend, None, circuit=meas_mapped)

    # Mock the create:
    create_path = client.make_path("jobs")
    requests_mock.post(create_path, status_code=200, json=job_response)

    # Submit the job.
    job.submit()

    # Mock the fetch from `result`, since status should still be "initializing".
    fetch_path = client.make_path("jobs", job_id)
    requests_mock.get(fetch_path, status_code=200, json=job_response)

    # Mock the fetch from `results`
    fetch_path = client.make_path("jobs", job_id, "results", "probabilities")
    requests_mock.get(fetch_path, status_code=200, json={"2": 1})

    # Validate the result and its format. should be the same as base case.
    res = job.result().get_counts()
    assert res == {"01": 1234}  # 1234 shots, all 10(2) remapped to 01(1)


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
    """Test result fetching when the job is canceled on the API side.

    Args:
        mock_backend (MockBackend): A mock IonQBackend.
        requests_mock (:class:`request_mock.Mocker`): A requests mocker.
    """
    # Create the job:
    job_id = "test_id"
    job_result = conftest.dummy_job_response(job_id, status="canceled")
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
    assert "Cannot retrieve result for canceled job" in str(exc.value)


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


def test_status_with_detailed(mock_backend, requests_mock):
    """Test status() with detailed argument returns detailed children status.

    Args:
        mock_backend (MockBackend): A fake/mock IonQBackend.
        requests_mock (:class:`requests_mock.Mocker`): A requests mocker.
    """
    job_id = "test_id"
    child_job_id_1 = "child_test_id_1"
    child_job_id_2 = "child_test_id_2"

    # Create dummy job responses
    job_response = conftest.dummy_job_response(
        job_id, status="completed", children=[child_job_id_1, child_job_id_2]
    )
    child_job_1_response = conftest.dummy_job_response(
        child_job_id_1, status="completed"
    )
    child_job_2_response = conftest.dummy_job_response(child_job_id_2, status="started")

    # Mock the job response API calls
    client = mock_backend.client
    requests_mock.get(
        client.make_path("jobs", job_id), status_code=200, json=job_response
    )
    requests_mock.get(
        client.make_path("jobs", child_job_id_1),
        status_code=200,
        json=child_job_1_response,
    )
    requests_mock.get(
        client.make_path("jobs", child_job_id_2),
        status_code=200,
        json=child_job_2_response,
    )

    # Create a job ref (this will call .status() to fetch our mock above)
    job = ionq_job.IonQJob(mock_backend, job_id)

    # Call status with detailed=True
    detailed_status = job.status(detailed=True)

    # Expected detailed status
    expected_detailed_status = {
        "total": 2,
        "completed": 1,
        "failed": 0,
        "percentage_complete": 0.5,
        "statuses": [jobstatus.JobStatus.DONE, jobstatus.JobStatus.RUNNING],
    }

    # Assert the detailed status
    assert detailed_status == expected_detailed_status


def test_single_circuit_clbit_map(mock_backend, requests_mock):
    """
    For a single-circuit submission the meas-map is a dict, not a list.
    Verify IonQJob.status() still ends up with a one-element _clbits list.
    """
    job_id = "single_meas_map"
    meas_map = [2, 1, 0]
    header = {
        "n_qubits": 3,
        "memory_slots": 3,
        "meas_mapped": meas_map,
    }

    child_id = f"{job_id}_child_0"
    payload = _mock_job_payload(
        job_id, header, circuits=1, qubits=3, children=[child_id]
    )
    requests_mock.get(
        mock_backend.client.make_path("jobs", job_id), status_code=200, json=payload
    )
    # mock the child-job fetch
    requests_mock.get(
        mock_backend.client.make_path("jobs", child_id),
        status_code=200,
        json={"id": child_id, "status": "completed", "metadata": {}},
    )

    job = ionq_job.IonQJob(mock_backend, job_id)
    assert job._clbits == [meas_map]


def test_multi_circuit_clbit_map(mock_backend, requests_mock):
    """
    For a multi-circuit submission the parent metadata contains a list
    of per-circuit headers. Each meas-map must be preserved in order.
    """
    job_id = "multi_meas_map"
    meas_maps = [[0, 1], [1, 0], [None, 0]]  # three different circuits
    header_list = [
        {"n_qubits": 2, "memory_slots": 2, "meas_mapped": m} for m in meas_maps
    ]

    child_ids = [f"{job_id}_child_{i}" for i in range(len(meas_maps))]
    payload = _mock_job_payload(
        job_id, header_list, circuits=len(meas_maps), qubits=2, children=child_ids
    )
    requests_mock.get(
        mock_backend.client.make_path("jobs", job_id), status_code=200, json=payload
    )
    # mock every child-job fetch
    for cid in child_ids:
        requests_mock.get(
            mock_backend.client.make_path("jobs", cid),
            status_code=200,
            json={"id": cid, "status": "completed", "metadata": {}},
        )

    job = ionq_job.IonQJob(mock_backend, job_id)
    assert job._clbits == meas_maps


def test_no_memory_skips_shots(mock_backend, requests_mock):
    """memory=False yields memory=None and skips the shots GET."""
    job_id = "test_no_memory"
    client = mock_backend.client

    requests_mock.get(
        client.make_path("jobs", job_id),
        status_code=200,
        json=conftest.dummy_job_response(job_id),
    )
    requests_mock.get(
        client.make_path("jobs", job_id, "results", "probabilities"),
        status_code=200,
        json={"0": 0.5, "2": 0.499999},
    )

    job = ionq_job.IonQJob(
        mock_backend,
        job_id,
        passed_args={"memory": False, "shots": 1024, "sampler_seed": None},
    )
    result = job.result()
    assert result.data(0).get("memory") is None
    assert result.get_counts()


def test_ideal_sim_skips_shots(simulator_backend, requests_mock):
    """Ideal simulator never fetches shots."""
    job_id = "test_ideal_sim"
    client = simulator_backend.client

    requests_mock.get(
        client.make_path("jobs", job_id),
        status_code=200,
        json=conftest.dummy_job_response(job_id),
    )
    requests_mock.get(
        client.make_path("jobs", job_id, "results", "probabilities"),
        status_code=200,
        json={"0": 0.5, "2": 0.499999},
    )

    job = ionq_job.IonQJob(simulator_backend, job_id)
    result = job.result()
    assert result.data(0).get("memory") is None
    assert result.get_counts()


def test_no_shots_url_returns_none(mock_backend, requests_mock):
    """memory=True + no shots URL in the response -> memory=None, no fetch."""
    job_id = "no_shots_url"
    client = mock_backend.client

    resp = conftest.dummy_job_response(job_id)
    resp["results"] = {k: v for k, v in resp["results"].items() if k != "shots"}
    requests_mock.get(client.make_path("jobs", job_id), json=resp)
    requests_mock.get(
        client.make_path("jobs", job_id, "results", "probabilities"),
        json={"0": 0.5, "2": 0.499999},
    )

    job = ionq_job.IonQJob(
        mock_backend,
        job_id,
        passed_args={"memory": True, "shots": 1024, "sampler_seed": None},
    )
    result = job.result()
    assert result.data(0).get("memory") is None
    assert result.get_counts()


def test_build_memory_3q_format():
    """Wire format is decimal-encoded outcome ints; verify on 3 qubits where
    decimal- and binary-string interpretations diverge.
    """
    from qiskit_ionq.ionq_job import _build_memory

    raw = ["6", "1", "0", "7"]
    out = _build_memory(raw, n_qubits=3, clbits=[0, 1, 2])
    assert out == ["110", "001", "000", "111"]
    assert _build_memory([6, 1, 0, 7], n_qubits=3, clbits=[0, 1, 2]) == out


def test_get_memory_raises_when_off(mock_backend, requests_mock):
    """get_memory() raises IonQBackendError if the job ran with memory=False."""
    job_id = "memory_off"
    client = mock_backend.client
    requests_mock.get(
        client.make_path("jobs", job_id),
        status_code=200,
        json=conftest.dummy_job_response(job_id),
    )
    requests_mock.get(
        client.make_path("jobs", job_id, "results", "probabilities"),
        status_code=200,
        json={"0": 0.5, "2": 0.499999},
    )
    job = ionq_job.IonQJob(
        mock_backend,
        job_id,
        passed_args={"memory": False, "shots": 1024, "sampler_seed": None},
    )
    with pytest.raises(exceptions.IonQBackendError, match="memory=True"):
        job.get_memory()


def test_multi_mem_per_child(mock_backend, requests_mock):
    """Multi-circuit jobs assemble per-shot memory by fetching each child's
    own ``results.shots.url``. The parent only advertises aggregated
    probabilities, but every child carries a shots payload.
    """
    job_id = "multi_mem"
    child_ids = ["child_a", "child_b"]
    parent = conftest.dummy_multi_parent_response(job_id, child_ids)

    client = mock_backend.client
    requests_mock.get(client.make_path("jobs", job_id), json=parent)
    requests_mock.get(
        client.make_path("jobs", job_id, "results", "probabilities", "aggregated"),
        json={child_ids[0]: {"0": 0.5, "3": 0.5}, child_ids[1]: {"1": 1.0}},
    )
    # Each child advertises its own shots URL on retrieval.
    for cid in child_ids:
        requests_mock.get(
            client.make_path("jobs", cid),
            json={
                "id": cid,
                "status": "completed",
                "metadata": {},
                "results": {"shots": {"url": f"/v0.4/jobs/{cid}/results/shots"}},
            },
        )
    requests_mock.get(
        client.make_path("jobs", child_ids[0], "results", "shots"),
        json=[0, 3, 0, 3],
    )
    requests_mock.get(
        client.make_path("jobs", child_ids[1], "results", "shots"),
        json=[1, 1, 1],
    )

    job = ionq_job.IonQJob(
        mock_backend,
        job_id,
        passed_args={"memory": True, "shots": 1024, "sampler_seed": None},
    )
    result = job.result()

    # Bell-like child: outcomes 0 and 3 on 2 inferred qubits -> '00' / '11'.
    assert result.data(0).get("memory") == ["00", "11", "00", "11"]
    # X-like child: outcome 1 on 1 inferred qubit -> '1'.
    assert result.data(1).get("memory") == ["1", "1", "1"]


def test_multi_mem_partial(mock_backend, requests_mock):
    """If one child lacks a shots URL, only that circuit's memory is ``None``;
    the others still assemble.
    """
    job_id = "multi_mem_partial"
    child_ids = ["child_ok", "child_no_shots"]
    parent = conftest.dummy_multi_parent_response(job_id, child_ids)

    client = mock_backend.client
    requests_mock.get(client.make_path("jobs", job_id), json=parent)
    requests_mock.get(
        client.make_path("jobs", job_id, "results", "probabilities", "aggregated"),
        json={child_ids[0]: {"0": 0.5, "3": 0.5}, child_ids[1]: {"1": 1.0}},
    )
    requests_mock.get(
        client.make_path("jobs", child_ids[0]),
        json={
            "id": child_ids[0],
            "status": "completed",
            "metadata": {},
            "results": {"shots": {"url": f"/v0.4/jobs/{child_ids[0]}/results/shots"}},
        },
    )
    requests_mock.get(
        client.make_path("jobs", child_ids[1]),
        json={
            "id": child_ids[1],
            "status": "completed",
            "metadata": {},
            "results": {},
        },
    )
    requests_mock.get(
        client.make_path("jobs", child_ids[0], "results", "shots"),
        json=[0, 3],
    )

    job = ionq_job.IonQJob(
        mock_backend,
        job_id,
        passed_args={"memory": True, "shots": 1024, "sampler_seed": None},
    )
    result = job.result()
    assert result.data(0).get("memory") == ["00", "11"]
    assert result.data(1).get("memory") is None


def test_shots_fetch_warns_on_error(mock_backend, requests_mock):
    """A 5xx on /results/shots surfaces a UserWarning and yields memory=None
    rather than failing silently.
    """
    job_id = "shots_500"
    client = mock_backend.client
    requests_mock.get(
        client.make_path("jobs", job_id),
        json=conftest.dummy_job_response(job_id),
    )
    requests_mock.get(
        client.make_path("jobs", job_id, "results", "probabilities"),
        json={"0": 0.5, "2": 0.5},
    )
    requests_mock.get(
        client.make_path("jobs", job_id, "results", "shots"),
        status_code=500,
        json={"error": {"type": "InternalServerError", "message": "boom"}},
    )
    job = ionq_job.IonQJob(
        mock_backend,
        job_id,
        passed_args={"memory": True, "shots": 1024, "sampler_seed": None},
    )
    with pytest.warns(UserWarning, match="Failed to fetch per-shot memory"):
        result = job.result()
    assert result.data(0).get("memory") is None


# ---------------------------------------------------------------------------
# dry_run / compilation-as-a-service
# ---------------------------------------------------------------------------


def _dry_run_job_response(job_id, target="qpu.forte-1"):
    """Mimic the API response for a completed dry-run job.

    Per the v0.4 spec, the top-level ``dry_run`` boolean is echoed back, and
    ``results`` is null because no measurement data is produced.
    """
    qiskit_header = compress_to_metadata_string(
        {
            "qubit_labels": [["q", 0], ["q", 1]],
            "n_qubits": 2,
            "qreg_sizes": [["q", 2]],
            "clbit_labels": [["c", 0], ["c", 1]],
            "memory_slots": 2,
            "creg_sizes": [["c", 2]],
            "name": job_id,
            "global_phase": 0,
        }
    )
    return {
        "id": job_id,
        "status": "completed",
        "type": "ionq.circuit.v1",
        "backend": target,
        "dry_run": True,
        "shots": 1024,
        "metadata": {"qiskit_header": qiskit_header, "shots": "1024"},
        "stats": {"qubits": 2, "circuits": 1},
        # Compiled circuits: artifacts keyed by format, fetched via /artifacts/{id}.
        "output": {
            "compilation": {
                "opt": 1,
                "precision": "1E-3",
                "gate_basis": "ZZ",
                "service_version": "v0.4",
                "compiled_circuits": {
                    "ionq.native.v1": {
                        "id": "native-aid",
                        "format": "ionq.native.v1",
                        "media_type": "application/json",
                    },
                    "ionq.ore.v1": {
                        "id": "ore-aid",
                        "format": "ionq.ore.v1",
                        "media_type": "application/json",
                    },
                },
            }
        },
        # Per v0.4 spec, dry-run jobs have results=null.
        "results": None,
    }


def test_dry_run_set_before_poll(mock_backend):
    """`dry_run=True` must be reflected on the job immediately, not only after
    the first status() poll. The v0.4 ``JobCreationResponse`` (POST /jobs 201)
    only carries ``{id, status, session_id}`` and omits ``dry_run``, so we
    must mirror the kwarg into ``self._dry_run`` at construction."""
    job = ionq_job.IonQJob(
        mock_backend,
        None,
        circuit=QuantumCircuit(1, 1),
        passed_args={"dry_run": True, "shots": 64},
    )
    assert job.dry_run is True
    # ...and the flag must remain in _passed_args so the wire payload still
    # includes it (helpers.qiskit_to_ionq reads it via .get()).
    assert job._passed_args.get("dry_run") is True


def test_dry_run_no_results_urls(mock_backend, requests_mock):
    """Dry-run jobs should reach DONE without a results URL crash."""
    job_id = "dry_run_id"
    fetch_path = mock_backend.client.make_path("jobs", job_id)
    requests_mock.get(fetch_path, json=_dry_run_job_response(job_id))

    # status() runs as part of __init__ when job_id is supplied
    job = ionq_job.IonQJob(mock_backend, job_id)

    assert job.status() == jobstatus.JobStatus.DONE
    assert job.dry_run is True
    assert job._results_urls == {}


def test_dry_run_result_raises(mock_backend, requests_mock):
    """Calling .result() on a dry-run job should raise a clear IonQJobError
    instead of the cryptic TypeError from passing None into make_path()."""
    job_id = "dry_run_id"
    fetch_path = mock_backend.client.make_path("jobs", job_id)
    requests_mock.get(fetch_path, json=_dry_run_job_response(job_id))

    job = ionq_job.IonQJob(mock_backend, job_id)

    with pytest.raises(exceptions.IonQJobError, match="dry_run=True"):
        job.result()


def test_dry_run_compiled_native(mock_backend, requests_mock):
    """compiled_circuit('native') resolves ionq.native.v1 and fetches the artifact."""
    job_id = "dry_run_id"
    requests_mock.get(
        mock_backend.client.make_path("jobs", job_id),
        json=_dry_run_job_response(job_id),
    )

    native = {"gateset": "native", "circuit": [{"gate": "gpi2", "target": 0}]}
    requests_mock.get(
        mock_backend.client.make_path("jobs", job_id, "artifacts", "native-aid"),
        json=native,
    )

    job = ionq_job.IonQJob(mock_backend, job_id)
    assert job.compiled_circuit() == native
    assert job.compiled_circuit(lang="native") == native
    # An exact format key resolves too.
    assert job.compiled_circuit(lang="ionq.native.v1") == native


def test_dry_run_compiled_ore(mock_backend, requests_mock):
    """compiled_circuit('ore') resolves ionq.ore.v1 and fetches the artifact."""
    job_id = "dry_run_id"
    requests_mock.get(
        mock_backend.client.make_path("jobs", job_id),
        json=_dry_run_job_response(job_id),
    )

    ore = {"format": "ore", "circuit": [{"op": "rz", "target": 0, "angle": 0.5}]}
    requests_mock.get(
        mock_backend.client.make_path("jobs", job_id, "artifacts", "ore-aid"),
        json=ore,
    )

    job = ionq_job.IonQJob(mock_backend, job_id)
    assert job.compiled_circuit(lang="ore") == ore
    assert job.compiled_circuit(lang="ionq.ore.v1") == ore


def test_compiled_mir_bytes(mock_backend, requests_mock):
    """A qasm3 job's ionq.mir.v1 (octet-stream) returns raw bytes, not JSON."""
    job_id = "mcm_mir"
    response = _qasm3_job_response(job_id, "unused")
    response["output"] = {
        "compilation": {
            "compiled_circuits": {
                "ionq.mir.v1": {
                    "id": "mir-aid",
                    "format": "ionq.mir.v1",
                    "media_type": "application/octet-stream",
                }
            }
        }
    }
    client = mock_backend.client
    requests_mock.get(client.make_path("jobs", job_id), json=response)
    blob = b"\x00\x01MIR\xff"
    requests_mock.get(
        client.make_path("jobs", job_id, "artifacts", "mir-aid"),
        content=blob,
        headers={"Content-Type": "application/octet-stream"},
    )

    job = ionq_job.IonQJob(mock_backend, job_id)
    assert job.compiled_circuit(lang="mir") == blob
    with pytest.raises(exceptions.IonQJobError, match="Available formats"):
        job.compiled_circuit()


def _strip_native_id(response):
    """Republish the native format without an artifact id (treated unavailable)."""
    response["output"]["compilation"]["compiled_circuits"] = {
        "ionq.native.v1": {"format": "ionq.native.v1", "media_type": "application/json"}
    }
    return response


@pytest.mark.parametrize(
    "mutate, lang",
    [
        (lambda r: r, "mir"),  # lang matches no published format
        (lambda r: {**r, "output": None}, "native"),  # nothing published (prod)
        (_strip_native_id, "native"),  # published format lacks an id
    ],
)
def test_compiled_unavailable(mock_backend, requests_mock, mutate, lang):
    """compiled_circuit() raises IonQJobError when no usable artifact matches."""
    job_id = "dry_run_id"
    requests_mock.get(
        mock_backend.client.make_path("jobs", job_id),
        json=mutate(_dry_run_job_response(job_id)),
    )
    job = ionq_job.IonQJob(mock_backend, job_id)
    with pytest.raises(exceptions.IonQJobError, match="Available formats"):
        job.compiled_circuit(lang=lang)


def test_compiled_artifact_error(mock_backend, requests_mock):
    """A non-2xx on the artifact fetch (e.g. entitlement) surfaces as IonQAPIError."""
    job_id = "dry_run_id"
    requests_mock.get(
        mock_backend.client.make_path("jobs", job_id),
        json=_dry_run_job_response(job_id),
    )
    requests_mock.get(
        mock_backend.client.make_path("jobs", job_id, "artifacts", "native-aid"),
        status_code=403,
        json={
            "statusCode": 403,
            "error": "Forbidden",
            "message": "Organization does not have access to this artifact",
        },
    )
    job = ionq_job.IonQJob(mock_backend, job_id)
    with pytest.raises(exceptions.IonQAPIError):
        job.compiled_circuit(lang="native")


def test_dry_run_property_false(mock_backend, requests_mock):
    """A regular (non-dry-run) job exposes dry_run=False."""
    job_id = "regular_job"
    fetch_path = mock_backend.client.make_path("jobs", job_id)
    requests_mock.get(fetch_path, json=conftest.dummy_job_response(job_id))

    job = ionq_job.IonQJob(mock_backend, job_id)
    assert job.dry_run is False


def test_multi_null_meta_result(mock_backend, requests_mock):
    """Multicircuit parent with ``metadata: null`` (raw-submitted) must not
    crash on retrieval; qubit count is inferred from result keys.
    """
    job_id = "parent_null_meta"
    child_ids = ["child_a", "child_b"]
    parent = conftest.dummy_multi_parent_response(job_id, child_ids)
    aggregated = {child_ids[0]: {"0": 0.5, "3": 0.5}, child_ids[1]: {"1": 1.0}}

    client = mock_backend.client
    requests_mock.get(client.make_path("jobs", job_id), json=parent)
    requests_mock.get(
        client.make_path("jobs", job_id, "results", "probabilities", "aggregated"),
        json=aggregated,
    )

    # Null-meta handling is the focus here; memory is exercised separately.
    job = ionq_job.IonQJob(
        mock_backend,
        job_id,
        passed_args={"memory": False, "shots": 1024, "sampler_seed": None},
    )
    result = job.result()

    assert result.success is True
    # Bell: max key 3 -> 2 qubits inferred -> 2-char bitstrings.
    assert result.get_counts(0) == {"00": 512, "11": 512}
    # X: max key 1 -> 1 qubit inferred -> 1-char bitstrings.
    assert result.get_counts(1) == {"1": 1024}


def _mcm_circuit():
    """1-qubit mid-circuit-measurement circuit (the API schema example)."""
    qr = QuantumRegister(1, "q")
    mid = ClassicalRegister(1, "mid")
    result = ClassicalRegister(2, "result")
    qc = QuantumCircuit(qr, mid, result, name="mcm")
    qc.h(0)
    qc.measure(0, mid[0])
    qc.x(0)
    qc.measure(0, result[0])
    qc.x(0)
    qc.measure(0, result[1])
    return qc


def _qasm3_job_response(job_id, shots_artifact_id):
    """Completed ionq.qasm3.v1 response: v1 result URLs + a v2 shots artifact id."""
    header = compress_to_metadata_string(
        {
            "memory_slots": 3,
            "creg_sizes": [["mid", 1], ["result", 2]],
            "clbit_labels": [["mid", 0], ["result", 0], ["result", 1]],
            "qreg_sizes": [["q", 1]],
            "qubit_labels": [["q", 0]],
            "n_qubits": 1,
            "name": "mcm",
            "global_phase": 0,
        }
    )
    return {
        "id": job_id,
        "type": "ionq.qasm3.v1",
        "status": "completed",
        "stats": {"qubits": 1, "circuits": 1},
        "metadata": {"qiskit_header": header, "shots": "4"},
        "results": {
            "shots": {"url": f"/v0.4/jobs/{job_id}/results/shots"},
            "histogram": {"url": f"/v0.4/jobs/{job_id}/results/histogram"},
            "probabilities": {"url": f"/v0.4/jobs/{job_id}/results/probabilities"},
            "ionq.result.shots.json.v2": {
                "id": shots_artifact_id,
                "format": "ionq.result.shots.json.v2",
                "media_type": "application/json",
            },
        },
        "execution_duration_ms": 0,
    }


# v2 shots artifact: 3x mid=0/result=00, 1x mid=1/result=10.
_QASM3_SHOTS = {
    "shots": [
        {"registers": {"mid": [0], "result": [0, 0], "output_all": [1]}},
        {"registers": {"mid": [0], "result": [0, 0], "output_all": [0]}},
        {"registers": {"mid": [0], "result": [0, 0], "output_all": [1]}},
        {"registers": {"mid": [1], "result": [1, 0], "output_all": [0]}},
    ]
}


def test_qasm3_result_counts(mock_backend, requests_mock):
    """MCM jobs fold per-register shots into register-split counts + memory."""
    job_id = "mcm_job"
    artifact_id = "shots-v2-uuid"
    client = mock_backend.client
    requests_mock.post(client.make_path("jobs"), json={"id": job_id})
    requests_mock.get(
        client.make_path("jobs", job_id),
        json=_qasm3_job_response(job_id, artifact_id),
    )
    requests_mock.get(
        client.make_path("jobs", job_id, "artifacts", artifact_id),
        json=_QASM3_SHOTS,
    )

    job = mock_backend.run(_mcm_circuit(), shots=4, memory=True)
    res = job.result()

    # Qiskit splits as "result mid": mid=1,result=10 -> "01 1".
    assert res.get_counts() == {"00 0": 3, "01 1": 1}
    assert res.get_memory() == ["00 0", "00 0", "00 0", "01 1"]


def test_qasm3_ideal_sim_no_shots(mock_backend, requests_mock):
    """The ideal simulator publishes only aggregate probabilities (no shots), so
    result() raises an actionable error pointing to a noise model / QPU."""
    job_id = "mcm_ideal"
    response = _qasm3_job_response(job_id, "unused")
    response["results"] = {
        "probabilities": {"url": f"/v0.4/jobs/{job_id}/results/probabilities"},
        "ionq.result.probabilities.json.v2": {
            "id": "probs-aid",
            "format": "ionq.result.probabilities.json.v2",
            "media_type": "application/json",
        },
    }
    client = mock_backend.client
    requests_mock.post(client.make_path("jobs"), json={"id": job_id})
    requests_mock.get(client.make_path("jobs", job_id), json=response)

    job = mock_backend.run(_mcm_circuit(), shots=4)
    with pytest.raises(exceptions.IonQJobError, match="noise model"):
        job.result()


def test_qasm3_high_bit(mock_backend, requests_mock):
    """A set high-order register bit (result[1]) folds to the MSB position."""
    job_id = "mcm_hi"
    artifact_id = "shots-hi"
    client = mock_backend.client
    requests_mock.post(client.make_path("jobs"), json={"id": job_id})
    requests_mock.get(
        client.make_path("jobs", job_id),
        json=_qasm3_job_response(job_id, artifact_id),
    )
    # result[1] is clbit index 2 -> MSB of the 3-bit word; expect "10 0".
    requests_mock.get(
        client.make_path("jobs", job_id, "artifacts", artifact_id),
        json={"shots": [{"registers": {"mid": [0], "result": [0, 1]}}]},
    )

    job = mock_backend.run(_mcm_circuit(), shots=1)
    assert job.result().get_counts() == {"10 0": 1}


def test_qasm3_memory_false(mock_backend, requests_mock):
    """Without memory=True, MCM counts still decode but get_memory() raises."""
    job_id = "mcm_nomem"
    artifact_id = "shots-nomem"
    client = mock_backend.client
    requests_mock.post(client.make_path("jobs"), json={"id": job_id})
    requests_mock.get(
        client.make_path("jobs", job_id),
        json=_qasm3_job_response(job_id, artifact_id),
    )
    requests_mock.get(
        client.make_path("jobs", job_id, "artifacts", artifact_id),
        json=_QASM3_SHOTS,
    )

    job = mock_backend.run(_mcm_circuit(), shots=4)  # memory defaults to False
    res = job.result()
    assert res.get_counts() == {"00 0": 3, "01 1": 1}
    assert res.data(0).get("memory") is None
    with pytest.raises(exceptions.IonQBackendError, match="memory=True"):
        job.get_memory()
