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

# Copyright 2026 IonQ, Inc. (www.ionq.com)
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

"""Result artifact selection preserves cloud processing and Qiskit bit order."""

import warnings

import pytest

from qiskit_ionq import exceptions, ionq_job
from qiskit_ionq.constants import ResultFormat
from qiskit_ionq.helpers import compress_to_metadata_string

from ..utils import dummy_job_response


def _artifact(
    requests_mock, client, response, result_format, payload, *, aggregation=None
):
    """Serve ``payload`` as one of ``response``'s artifacts, listed under its
    results or, with ``aggregation``, as that method's output."""
    artifact_id = f"{response['id']}-{aggregation or result_format.value}"
    descriptor = {"id": artifact_id, "format": result_format.value}
    if aggregation is None:
        response["results"][result_format.value] = descriptor
    else:
        mitigation = response.setdefault("output", {}).setdefault(
            "error_mitigation", {}
        )
        mitigation.setdefault("aggregations", {})[aggregation] = descriptor
    requests_mock.get(
        client.make_path("jobs", response["id"], "artifacts", artifact_id),
        json=payload,
    )


def _distribution(requests_mock, client, response, states, *, aggregation=None):
    """Publish ``states`` (wire order) as the default probabilities, or as
    ``aggregation``'s histogram, the formats the API uses for each."""
    if aggregation is None:
        result_format, key = ResultFormat.PROBABILITIES_V2, "probabilities"
    else:
        result_format, key = ResultFormat.HISTOGRAM_V2, "histogram"
    payload = {key: {"registers": {"output_all": states}}}
    _artifact(
        requests_mock, client, response, result_format, payload, aggregation=aggregation
    )


@pytest.mark.parametrize("v2_only", [False, True])
def test_default_preserves_cloud_mitigation(mock_backend, requests_mock, v2_only):
    """The default reads the probabilities artifact, not a raw or method histogram."""
    response = dummy_job_response("default", shots=10)
    client = mock_backend.client
    _distribution(requests_mock, client, response, {"10": 0.3, "01": 0.7})
    _artifact(
        requests_mock,
        client,
        response,
        ResultFormat.HISTOGRAM_V2,
        {"histogram": {"registers": {"output_all": {"00": 10}}}},
    )
    _distribution(requests_mock, client, response, {"11": 4}, aggregation="average")
    # memory=False must not request even an advertised v2 shots artifact.
    response["results"][ResultFormat.SHOTS_V2] = {"id": "unused"}
    if v2_only:
        response["results"] = {
            key: value
            for key, value in response["results"].items()
            if key.startswith("ionq.result.")
        }

    requests_mock.get(client.make_path("jobs", response["id"]), json=response)
    result = ionq_job.IonQJob(mock_backend, response["id"]).result()

    assert result.get_counts() == {"01": 3, "10": 7}
    assert result.get_probabilities() == {"01": 0.3, "10": 0.7}
    assert result.results[0].shots == 10
    assert result.data(0).get("memory") is None
    assert requests_mock.last_request.path.endswith(ResultFormat.PROBABILITIES_V2.value)


def test_default_legacy_fallback(mock_backend, requests_mock):
    """Jobs from before v2 result artifacts keep the legacy distribution."""
    response = dummy_job_response("legacy", shots=10)
    client = mock_backend.client
    requests_mock.get(
        client.make_path("jobs", "legacy", "results", "probabilities"), json={"2": 1.0}
    )
    requests_mock.get(client.make_path("jobs", response["id"]), json=response)
    result = ionq_job.IonQJob(mock_backend, response["id"]).result()
    assert result.get_counts() == {"10": 10}


@pytest.mark.parametrize("aggregation", ["average", "voting", "dnl"])
def test_requested_method_over_default(mock_backend, requests_mock, aggregation):
    """An explicit method selects its histogram and preserves its actual total."""
    response = dummy_job_response("selected", shots=10)
    client = mock_backend.client
    _distribution(requests_mock, client, response, {"00": 1.0})
    _distribution(
        requests_mock, client, response, {"10": 3, "01": 1}, aggregation=aggregation
    )
    requests_mock.get(client.make_path("jobs", response["id"]), json=response)
    job = ionq_job.IonQJob(mock_backend, response["id"])
    result = job.result(aggregation=aggregation)
    assert result.get_counts() == {"01": 3, "10": 1}
    assert result.get_probabilities() == {"01": 0.75, "10": 0.25}
    assert result.results[0].shots == 4


def test_missing_method_keeps_legacy_query(mock_backend, requests_mock):
    """A method the job doesn't publish keeps the legacy query, never the default."""
    response = dummy_job_response("legacy-method", shots=10)
    client = mock_backend.client
    _distribution(requests_mock, client, response, {"00": 1.0})
    requests_mock.get(
        client.make_path("jobs", response["id"], "results", "probabilities")
        + "?aggregation=voting",
        json={"3": 1.0},
        complete_qs=True,
    )
    requests_mock.get(client.make_path("jobs", response["id"]), json=response)
    result = ionq_job.IonQJob(mock_backend, response["id"]).result(aggregation="voting")
    assert result.get_counts() == {"11": 10}


@pytest.mark.parametrize("aggregation,value,total", [(None, 1.0, 10), ("dnl", 4, 4)])
def test_batch_order(mock_backend, requests_mock, aggregation, value, total):
    """Each child's distribution keeps its circuit's position in the batch."""
    parent = dummy_job_response("batch", shots=10)
    parent["child_job_ids"] = ["z-first", "a-second"]
    client = mock_backend.client
    for child_id, wire_state in zip(parent["child_job_ids"], ["10", "01"]):
        child = dummy_job_response(child_id)
        _distribution(
            requests_mock, client, child, {wire_state: value}, aggregation=aggregation
        )
        requests_mock.get(client.make_path("jobs", child_id), json=child)
    requests_mock.get(client.make_path("jobs", parent["id"]), json=parent)
    job = ionq_job.IonQJob(mock_backend, parent["id"])
    result = job.result(aggregation=aggregation)
    assert result.get_counts() == [{"01": total}, {"10": total}]
    assert [experiment.shots for experiment in result.results] == [total, total]


def test_batch_legacy_fallback(mock_backend, requests_mock):
    """A batch with an older child falls back as a unit before fetching artifacts."""
    parent = dummy_job_response("mixed-batch", shots=10)
    parent["child_job_ids"] = ["modern", "older"]
    client = mock_backend.client
    modern = dummy_job_response("modern")
    modern["results"][ResultFormat.PROBABILITIES_V2] = {
        "id": "unused",
        "format": ResultFormat.PROBABILITIES_V2,
    }
    requests_mock.get(client.make_path("jobs", "modern"), json=modern)
    older = dummy_job_response("older")
    requests_mock.get(client.make_path("jobs", "older"), json=older)
    requests_mock.get(
        client.make_path("jobs", parent["id"], "results", "probabilities"),
        json={"modern": {"1": 1.0}, "older": {"2": 1.0}},
    )
    requests_mock.get(client.make_path("jobs", parent["id"]), json=parent)
    result = ionq_job.IonQJob(mock_backend, parent["id"]).result()
    assert result.get_counts() == [{"01": 10}, {"10": 10}]


def test_one_child_batch_uses_artifact(mock_backend, requests_mock):
    """A one-circuit batch reads its results from the child."""
    parent = dummy_job_response("one-parent", shots=10)
    parent["child_job_ids"] = ["only-child"]
    child = dummy_job_response("only-child")
    client = mock_backend.client
    _distribution(requests_mock, client, child, {"10": 1.0})
    requests_mock.get(client.make_path("jobs", child["id"]), json=child)
    requests_mock.get(client.make_path("jobs", parent["id"]), json=parent)
    result = ionq_job.IonQJob(mock_backend, parent["id"]).result()
    assert result.get_counts() == {"01": 10}


def test_batch_retrieves_each_child_once(mock_backend, requests_mock):
    """Distributions and memory share a single retrieval of each child."""
    parent = dummy_job_response("once", shots=10)
    parent["child_job_ids"] = ["child-a", "child-b"]
    client = mock_backend.client
    for child_id in parent["child_job_ids"]:
        child = dummy_job_response(child_id)
        _distribution(requests_mock, client, child, {"10": 1.0})
        _artifact(
            requests_mock,
            client,
            child,
            ResultFormat.SHOTS_V2,
            {"shots": [{"registers": {"output_all": [1, 0]}}]},
        )
        requests_mock.get(client.make_path("jobs", child_id), json=child)
    requests_mock.get(client.make_path("jobs", parent["id"]), json=parent)
    job = ionq_job.IonQJob(mock_backend, parent["id"], passed_args={"memory": True})
    result = job.result()
    assert result.get_memory(0) == result.get_memory(1) == ["01"]
    requested = [request.url for request in requests_mock.request_history]
    for child_id in parent["child_job_ids"]:
        assert requested.count(client.make_path("jobs", child_id)) == 1


def test_default_simulator_sampling(simulator_backend, requests_mock):
    """V2 probabilities preserve seeded sampling on the ideal simulator."""
    client = simulator_backend.client
    legacy = dummy_job_response("legacy-sim", shots=10)
    requests_mock.get(
        client.make_path("jobs", legacy["id"], "results", "probabilities"),
        json={"1": 0.25, "2": 0.75},
    )
    modern = dummy_job_response("modern-sim", shots=10)
    _distribution(requests_mock, client, modern, {"10": 0.25, "01": 0.75})
    # An advertised shots artifact must not be fetched for ideal simulation.
    modern["results"][ResultFormat.SHOTS_V2] = {"id": "unused"}
    requests_mock.get(client.make_path("jobs", legacy["id"]), json=legacy)
    expected = ionq_job.IonQJob(
        simulator_backend, legacy["id"], passed_args={"memory": True}
    ).result()
    requests_mock.get(client.make_path("jobs", modern["id"]), json=modern)
    actual = ionq_job.IonQJob(
        simulator_backend, modern["id"], passed_args={"memory": True}
    ).result()
    assert actual.get_counts() == expected.get_counts()
    assert actual.get_probabilities() == expected.get_probabilities()
    assert actual.data(0).get("memory") is None


def test_v2_memory_maps_bits_and_flags_leakage(mock_backend, requests_mock):
    """Memory keeps every shot in order and mapped; drop_leaked drops flagged ones."""
    response = dummy_job_response("mapped", shots=10)
    response["metadata"]["qiskit_header"] = compress_to_metadata_string(
        {
            "n_qubits": 3,
            "memory_slots": 4,
            "meas_mapped": [2, None, 0, 1],
            "creg_sizes": [["c", 4]],
        }
    )
    client = mock_backend.client
    _distribution(requests_mock, client, response, {"011": 1.0})
    _artifact(
        requests_mock,
        client,
        response,
        ResultFormat.SHOTS_V2,
        {
            "shots": [
                {
                    "registers": {"output_all": [1, 0, 0], "unused": [1]},
                    "leakage_bits": [],
                },
                {"registers": {"output_all": [0, 1, 1]}, "leakage_bits": [0, 1, 0]},
                {"registers": {"output_all": [0, 1, 1]}, "leakage_bits": [0, 0, 0]},
                {"registers": {"output_all": [1, 0, 0]}},
            ]
        },
    )
    requests_mock.get(client.make_path("jobs", response["id"]), json=response)
    job = ionq_job.IonQJob(mock_backend, response["id"], passed_args={"memory": True})
    result = job.result()
    assert result.get_memory() == ["0100", "1001", "1001", "0100"]
    # Any non-empty leakage_bits marks a leaked shot, even all zeros.
    assert result.data(0)["leaked"] == [False, True, True, False]
    assert result.get_memory(drop_leaked=True) == ["0100", "0100"]
    assert job.get_memory(drop_leaked=True) == ["0100", "0100"]
    assert result.get_counts() == {"1001": 10}


def test_empty_memory(mock_backend, requests_mock):
    """An empty shot stream gives empty memory and leakage flags."""
    response = dummy_job_response("empty-memory", shots=10)
    client = mock_backend.client
    _distribution(requests_mock, client, response, {"00": 1.0})
    _artifact(requests_mock, client, response, ResultFormat.SHOTS_V2, {"shots": []})
    requests_mock.get(client.make_path("jobs", response["id"]), json=response)
    job = ionq_job.IonQJob(mock_backend, response["id"], passed_args={"memory": True})
    result = job.result()
    assert result.data(0)["memory"] == []
    assert result.data(0)["leaked"] == []
    assert result.get_memory(drop_leaked=True) == []


def test_legacy_memory_has_no_leak_flags(mock_backend, requests_mock):
    """Legacy shots carry no leak flags, so drop_leaked raises instead of guessing."""
    response = dummy_job_response("legacy-memory", shots=10)
    client = mock_backend.client
    requests_mock.get(
        client.make_path("jobs", response["id"], "results", "probabilities"),
        json={"2": 1.0},
    )
    requests_mock.get(
        client.make_path("jobs", response["id"], "results", "shots"), json=[2, 2]
    )
    requests_mock.get(client.make_path("jobs", response["id"]), json=response)
    job = ionq_job.IonQJob(mock_backend, response["id"], passed_args={"memory": True})
    result = job.result()
    assert result.get_memory() == ["10", "10"]
    assert "leaked" not in result.data(0)
    with pytest.raises(exceptions.IonQJobError, match="No leakage flags"):
        result.get_memory(drop_leaked=True)


def test_batch_mixed_memory_sources(mock_backend, requests_mock):
    """Memory can come from v2 on one child and legacy shots on another."""
    parent = dummy_job_response("memory-batch", shots=10)
    parent["child_job_ids"] = ["new-memory", "old-memory"]
    client = mock_backend.client
    for child_id in parent["child_job_ids"]:
        child = dummy_job_response(child_id)
        _distribution(requests_mock, client, child, {"10": 1.0})
        if child_id == "new-memory":
            _artifact(
                requests_mock,
                client,
                child,
                ResultFormat.SHOTS_V2,
                {"shots": [{"registers": {"output_all": [1, 0]}}]},
            )
        else:
            requests_mock.get(
                client.make_path("jobs", child_id, "results", "shots"), json=[2]
            )
        requests_mock.get(client.make_path("jobs", child_id), json=child)
    requests_mock.get(client.make_path("jobs", parent["id"]), json=parent)
    job = ionq_job.IonQJob(mock_backend, parent["id"], passed_args={"memory": True})
    result = job.result()
    assert result.get_memory(0) == ["01"]
    assert result.get_memory(1) == ["10"]


def _batch_with_unretrievable_child(requests_mock, client, aggregation=None):
    """A two-circuit batch whose second child cannot be retrieved."""
    parent = dummy_job_response("partial-batch", shots=10)
    parent["child_job_ids"] = ["reachable", "unreachable"]
    child = dummy_job_response("reachable")
    _distribution(requests_mock, client, child, {"10": 1}, aggregation=aggregation)
    _artifact(
        requests_mock,
        client,
        child,
        ResultFormat.SHOTS_V2,
        {"shots": [{"registers": {"output_all": [1, 0]}}]},
    )
    requests_mock.get(client.make_path("jobs", "reachable"), json=child)
    requests_mock.get(
        client.make_path("jobs", "unreachable"),
        status_code=404,
        json={"statusCode": 404, "error": "Not Found"},
    )
    legacy = requests_mock.get(
        client.make_path("jobs", parent["id"], "results", "probabilities"),
        json={"reachable": {"1": 1.0}, "unreachable": {"2": 1.0}},
    )
    return parent, legacy


@pytest.mark.parametrize("memory", [False, True])
def test_batch_unretrieved_child_keeps_counts(mock_backend, requests_mock, memory):
    """A child that cannot be retrieved costs only its own memory, not the batch."""
    client = mock_backend.client
    parent, _ = _batch_with_unretrievable_child(requests_mock, client)
    requests_mock.get(client.make_path("jobs", parent["id"]), json=parent)
    job = ionq_job.IonQJob(mock_backend, parent["id"], passed_args={"memory": memory})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = job.result()

    # Counts come from the parent's legacy endpoint, as before v2 artifacts.
    assert result.get_counts() == [{"01": 10}, {"10": 10}]
    warned = any(
        "Failed to fetch per-shot memory for unreachable" in str(w.message)
        for w in caught
    )
    assert warned is memory
    if memory:
        assert result.get_memory(0) == ["01"]
        assert result.data(1).get("memory") is None


def test_batch_unretrieved_child_fails_aggregation(mock_backend, requests_mock):
    """The legacy endpoint ignores aggregation, so it cannot cover for a lost child."""
    client = mock_backend.client
    parent, legacy = _batch_with_unretrievable_child(
        requests_mock, client, aggregation="dnl"
    )
    requests_mock.get(client.make_path("jobs", parent["id"]), json=parent)
    job = ionq_job.IonQJob(mock_backend, parent["id"])
    with pytest.raises(exceptions.IonQAPIError):
        job.result(aggregation="dnl")
    assert not legacy.called


def test_artifact_failure_is_not_fallback(mock_backend, requests_mock):
    """A published artifact that cannot be read must not silently change source."""
    response = dummy_job_response("forbidden", shots=10)
    response["results"][ResultFormat.PROBABILITIES_V2] = {
        "id": "denied",
        "format": ResultFormat.PROBABILITIES_V2,
    }
    client = mock_backend.client
    requests_mock.get(
        client.make_path("jobs", response["id"], "artifacts", "denied"),
        status_code=403,
        json={"statusCode": 403, "error": "Forbidden"},
    )
    requests_mock.get(client.make_path("jobs", response["id"]), json=response)
    job = ionq_job.IonQJob(mock_backend, response["id"])
    with pytest.raises(exceptions.IonQAPIError):
        job.result()


def test_v2_memory_fetch_failure_warns(mock_backend, requests_mock):
    """Optional memory failure leaves counts available, matching legacy behavior."""
    response = dummy_job_response("memory-forbidden", shots=10)
    client = mock_backend.client
    _distribution(requests_mock, client, response, {"10": 1.0})
    response["results"][ResultFormat.SHOTS_V2] = {"id": "denied"}
    requests_mock.get(
        client.make_path("jobs", response["id"], "artifacts", "denied"),
        status_code=403,
        json={"statusCode": 403, "error": "Forbidden"},
    )
    requests_mock.get(client.make_path("jobs", response["id"]), json=response)
    job = ionq_job.IonQJob(mock_backend, response["id"], passed_args={"memory": True})
    with pytest.warns(UserWarning, match="Failed to fetch per-shot memory"):
        result = job.result()
    assert result.get_counts() == {"01": 10}
    assert result.data(0).get("memory") is None


def test_invalid_shots_raise(mock_backend, requests_mock):
    """Missing output_all is malformed, not an all-zero measurement."""
    response = dummy_job_response("invalid-shots", shots=10)
    client = mock_backend.client
    _distribution(requests_mock, client, response, {"10": 1.0})
    _artifact(
        requests_mock,
        client,
        response,
        ResultFormat.SHOTS_V2,
        {"shots": [{"registers": {"unrelated": [0]}}]},
    )
    requests_mock.get(client.make_path("jobs", response["id"]), json=response)
    job = ionq_job.IonQJob(mock_backend, response["id"], passed_args={"memory": True})
    with pytest.raises(exceptions.IonQJobError, match="Invalid v2 shots artifact"):
        job.result()


def test_default_extra_query_params(mock_backend, requests_mock):
    """Extra result parameters are forwarded to the artifact request."""
    response = dummy_job_response("extra-query", shots=10)
    client = mock_backend.client
    _distribution(requests_mock, client, response, {"10": 1.0})
    requests_mock.get(client.make_path("jobs", response["id"]), json=response)
    job = ionq_job.IonQJob(mock_backend, response["id"])
    job.result(extra_query_params={"custom": "value"})
    assert requests_mock.last_request.qs == {"custom": ["value"]}
