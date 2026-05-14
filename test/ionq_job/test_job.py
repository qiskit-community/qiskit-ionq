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
from qiskit import QuantumCircuit
from qiskit.providers import exceptions as q_exc
from qiskit.providers import jobstatus
from qiskit.result import MeasLevel

from qiskit_ionq import exceptions, ionq_job
from qiskit_ionq.helpers import compress_to_metadata_string


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
    shots_path = simulator_backend.client.make_path("jobs", job_id, "results", "shots")
    requests_mock.get(results_path, json={"0": 0.5, "2": 0.499999})
    requests_mock.get(shots_path, json=617 * ["00"] + 617 * ["10"])
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
    shots_path = simulator_backend.client.make_path("jobs", job_id, "results", "shots")
    requests_mock.get(results_path, status_code=200, json={"0": 0.5, "2": 0.499999})
    requests_mock.get(shots_path, status_code=200, json=609 * ["00"] + 625 * ["10"])
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
                "memory": 617 * ["00"] + 617 * ["10"],
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
            "meas_level": MeasLevel.CLASSIFIED,
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
    shots_path = client.make_path("jobs", job_id, "results", "shots")
    requests_mock.get(results_path, status_code=200, json={"0": 0.5, "2": 0.499999})
    requests_mock.get(shots_path, status_code=200, json=617 * ["00"] + 617 * ["10"])

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
    shots_path = client.make_path("jobs", job_id, "results", "shots")
    requests_mock.get(results_path, status_code=200, json={"0": 0.5, "2": 0.499999})
    requests_mock.get(shots_path, status_code=200, json=617 * ["00"] + 617 * ["10"])

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
    shots_path = client.make_path("jobs", job_id, "results", "shots")
    requests_mock.get(results_path, status_code=200, json={"0": 0.5, "2": 0.499999})
    requests_mock.get(shots_path, status_code=200, json=617 * ["00"] + 617 * ["10"])

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
    shots_path = client.make_path("jobs", job_id, "results", "shots")
    requests_mock.get(results_path, status_code=200, json={"0": 0.5, "2": 0.499999})
    requests_mock.get(shots_path, status_code=200, json=617 * ["00"] + 617 * ["10"])

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
    shots_path = client.make_path("jobs", job_id, "results", "shots")
    requests_mock.get(results_path, status_code=200, json={"0": 0.5, "2": 0.499999})
    requests_mock.get(shots_path, status_code=200, json=617 * ["00"] + 617 * ["10"])

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
    shots_path = client.make_path("jobs", job_id, "results", "shots")
    requests_mock.get(fetch_path, status_code=200, json={"2": 1})
    requests_mock.get(shots_path, status_code=200, json=["10", "10"])

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
    """When memory=False the shots endpoint should not be hit."""
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
    # Deliberately NOT mocking the shots endpoint - requests_mock will raise
    # NoMockAddress if anything tries to GET it.

    job = ionq_job.IonQJob(
        mock_backend,
        job_id,
        passed_args={"memory": False, "shots": 1024, "sampler_seed": None},
    )
    result = job.result()
    assert result.data(0).get("memory") is None
    assert result.get_counts()


def test_ideal_sim_skips_shots(simulator_backend, requests_mock):
    """Ideal simulator should not fetch shots even when memory=True."""
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
    # Deliberately NOT mocking the shots endpoint.

    job = ionq_job.IonQJob(simulator_backend, job_id)
    result = job.result()
    assert result.data(0).get("memory") is None
    assert result.get_counts()


def test_no_shots_url_returns_none(mock_backend, requests_mock):
    """A job whose response does not advertise a shots URL must yield
    memory=None without an HTTP round-trip (regression-guard for the
    case where the IonQ backend has not yet rolled out shotwise output).
    """
    job_id = "no_shots_url"
    client = mock_backend.client

    resp = conftest.dummy_job_response(job_id)
    # Strip the shots URL the way the API does today for QPU jobs
    # that pre-date the shotwise rollout.
    resp["results"] = {k: v for k, v in resp["results"].items() if k != "shots"}
    requests_mock.get(client.make_path("jobs", job_id), json=resp)
    requests_mock.get(
        client.make_path("jobs", job_id, "results", "probabilities"),
        json={"0": 0.5, "2": 0.499999},
    )

    job = ionq_job.IonQJob(mock_backend, job_id)
    result = job.result()
    assert result.data(0).get("memory") is None
    assert result.get_counts()


def test_build_memory_3q_wire_format():
    """Lock the wire-format contract: shots come back as decimal-stringified
    outcome integers. Test with 3 qubits where decimal- vs binary-string
    interpretation produces different results, which would have been masked
    by a 2-qubit-only fixture.
    """
    from qiskit_ionq.ionq_job import _build_memory

    # API wire format: list of decimal-encoded outcomes.
    # 6 = binary 110 -> Qiskit bitstring "110" (q2=1, q1=1, q0=0).
    # 1 = binary 001 -> Qiskit bitstring "001".
    raw = ["6", "1", "0", "7"]
    out = _build_memory(raw, n_qubits=3, clbits=[0, 1, 2])
    assert out == ["110", "001", "000", "111"]

    # Integer inputs work the same way.
    assert _build_memory([6, 1, 0, 7], n_qubits=3, clbits=[0, 1, 2]) == out


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
        # Per v0.4 spec, dry-run jobs have results=null.
        "results": None,
    }


def test_dry_run_no_results_url(mock_backend, requests_mock):
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
    """compiled_circuit(lang='native') hits /jobs/<id>/circuits/native and
    returns the JSON-decoded body as a string."""
    job_id = "dry_run_id"
    requests_mock.get(
        mock_backend.client.make_path("jobs", job_id),
        json=_dry_run_job_response(job_id),
    )

    native_body = (
        '{"gateset":"native","circuit":[{"gate":"gpi2","target":0,"phase":0.0}]}'
    )
    requests_mock.get(
        mock_backend.client.make_path("jobs", job_id, "circuits", "native"),
        json=native_body,
    )

    job = ionq_job.IonQJob(mock_backend, job_id)
    assert job.compiled_circuit() == native_body
    assert job.compiled_circuit(lang="native") == native_body


def test_dry_run_compiled_qasm3(mock_backend, requests_mock):
    """compiled_circuit(lang='qasm3') returns the OpenQASM 3 string."""
    job_id = "dry_run_id"
    requests_mock.get(
        mock_backend.client.make_path("jobs", job_id),
        json=_dry_run_job_response(job_id),
    )

    qasm3 = "OPENQASM 3.0;\ngate gpi2(p) q { } // ...\n"
    requests_mock.get(
        mock_backend.client.make_path("jobs", job_id, "circuits", "qasm3"),
        json=qasm3,
    )

    job = ionq_job.IonQJob(mock_backend, job_id)
    assert job.compiled_circuit(lang="qasm3") == qasm3


def test_compiled_lang_passthrough(mock_backend, requests_mock):
    """Any string is forwarded to the API as-is.

    The server is the source of truth for which lang values are accepted
    and which are gated behind per-organization entitlement; the SDK does
    not duplicate that policy. This test mocks the request URL with a
    non-default lang and confirms it is reached.
    """
    job_id = "dry_run_id"
    requests_mock.get(
        mock_backend.client.make_path("jobs", job_id),
        json=_dry_run_job_response(job_id),
    )
    requests_mock.get(
        mock_backend.client.make_path("jobs", job_id, "circuits", "future-lang"),
        json="some-payload",
    )
    job = ionq_job.IonQJob(mock_backend, job_id)
    assert job.compiled_circuit(lang="future-lang") == "some-payload"


def test_compiled_lang_api_error(mock_backend, requests_mock):
    """Server-side rejection of an unsupported / non-entitled lang surfaces
    as IonQAPIError, matching every other non-2xx API response."""
    job_id = "dry_run_id"
    requests_mock.get(
        mock_backend.client.make_path("jobs", job_id),
        json=_dry_run_job_response(job_id),
    )
    requests_mock.get(
        mock_backend.client.make_path("jobs", job_id, "circuits", "nope"),
        status_code=403,
        json={
            "statusCode": 403,
            "error": "Forbidden",
            "message": "Organization does not have access to this compiled language",
        },
    )
    job = ionq_job.IonQJob(mock_backend, job_id)
    with pytest.raises(exceptions.IonQAPIError):
        job.compiled_circuit(lang="nope")


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

    result = ionq_job.IonQJob(mock_backend, job_id).result()

    assert result.success is True
    # Bell: max key 3 -> 2 qubits inferred -> 2-char bitstrings.
    assert result.get_counts(0) == {"00": 512, "11": 512}
    # X: max key 1 -> 1 qubit inferred -> 1-char bitstrings.
    assert result.get_counts(1) == {"1": 1024}
