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

import pytest

from qiskit_ionq import exceptions, ionq_job
from qiskit_ionq.constants import ResultFormat
from qiskit_ionq.helpers import compress_to_metadata_string

from ..utils import dummy_job_response


def _response(job_id):
    response = dummy_job_response(job_id)
    response["metadata"]["shots"] = "10"
    return response


def _artifact(
    requests_mock, client, response, result_format, payload, *, aggregation=None
):
    artifact_id = f"{response['id']}-{aggregation or result_format.value}"
    descriptor = {"id": artifact_id, "format": result_format.value}
    if aggregation is None:
        response["results"][result_format.value] = descriptor
    else:
        response.setdefault("output", {}).setdefault("error_mitigation", {}).setdefault(
            "aggregations", {}
        )[aggregation] = descriptor
    requests_mock.get(
        client.make_path("jobs", response["id"], "artifacts", artifact_id),
        json=payload,
    )


def _job(backend, requests_mock, response, memory=False):
    requests_mock.get(backend.client.make_path("jobs", response["id"]), json=response)
    return ionq_job.IonQJob(backend, response["id"], passed_args={"memory": memory})


@pytest.mark.parametrize("v2_only", [False, True])
def test_default_preserves_cloud_mitigation(mock_backend, requests_mock, v2_only):
    """Default probabilities can differ from both raw and per-method histograms."""
    response = _response("default")
    client = mock_backend.client
    _artifact(
        requests_mock,
        client,
        response,
        ResultFormat.PROBABILITIES_V2,
        {"probabilities": {"registers": {"output_all": {"10": 0.3, "01": 0.7}}}},
    )
    _artifact(
        requests_mock,
        client,
        response,
        ResultFormat.HISTOGRAM_V2,
        {"histogram": {"registers": {"output_all": {"00": 10}}}},
    )
    _artifact(
        requests_mock,
        client,
        response,
        ResultFormat.HISTOGRAM_V2,
        {"histogram": {"registers": {"output_all": {"11": 4}}}},
        aggregation="average",
    )
    # memory=False must not request even an advertised v2 shots artifact.
    response["results"][ResultFormat.SHOTS_V2] = {"id": "unused"}
    if v2_only:
        response["results"] = {
            key: value
            for key, value in response["results"].items()
            if key.startswith("ionq.result.")
        }

    result = _job(mock_backend, requests_mock, response).result()

    assert result.get_counts() == {"01": 3, "10": 7}
    assert result.get_probabilities() == {"01": 0.3, "10": 0.7}
    assert result.results[0].shots == 10
    assert result.data(0).get("memory") is None
    assert requests_mock.last_request.path.endswith(ResultFormat.PROBABILITIES_V2.value)


@pytest.mark.parametrize(
    "descriptor",
    [
        None,
        {},
        {"id": "unsupported", "format": "future"},
        {"id": "wrong-format", "format": ResultFormat.HISTOGRAM_V2},
    ],
)
def test_default_legacy_fallback(mock_backend, requests_mock, descriptor):
    """Missing/unusable default descriptors retain the legacy distribution."""
    response = _response("legacy")
    response["results"][ResultFormat.PROBABILITIES_V2] = descriptor
    response["results"][ResultFormat.HISTOGRAM_V2] = {
        "id": "raw-histogram",
        "format": ResultFormat.HISTOGRAM_V2,
    }
    client = mock_backend.client
    requests_mock.get(
        client.make_path("jobs", "legacy", "results", "probabilities"), json={"2": 1.0}
    )
    assert _job(mock_backend, requests_mock, response).result().get_counts() == {
        "10": 10
    }


@pytest.mark.parametrize("aggregation", ["average", "voting", "dnl"])
def test_requested_method_over_default(mock_backend, requests_mock, aggregation):
    """An explicit method selects its histogram and preserves its actual total."""
    response = _response("selected")
    client = mock_backend.client
    _artifact(
        requests_mock,
        client,
        response,
        ResultFormat.PROBABILITIES_V2,
        {"probabilities": {"registers": {"output_all": {"00": 1.0}}}},
    )
    _artifact(
        requests_mock,
        client,
        response,
        ResultFormat.HISTOGRAM_V2,
        {"histogram": {"registers": {"output_all": {"10": 3, "01": 1}}}},
        aggregation=aggregation,
    )
    result = _job(mock_backend, requests_mock, response).result(aggregation=aggregation)
    assert result.get_counts() == {"01": 3, "10": 1}
    assert result.get_probabilities() == {"01": 0.75, "10": 0.25}
    assert result.results[0].shots == 4


def test_missing_method_keeps_legacy_query(mock_backend, requests_mock):
    """Missing voting must not silently select the default v2 probabilities."""
    response = _response("legacy-method")
    client = mock_backend.client
    _artifact(
        requests_mock,
        client,
        response,
        ResultFormat.PROBABILITIES_V2,
        {"probabilities": {"registers": {"output_all": {"00": 1.0}}}},
    )
    requests_mock.get(
        client.make_path("jobs", response["id"], "results", "probabilities")
        + "?aggregation=voting",
        json={"3": 1.0},
        complete_qs=True,
    )
    result = _job(mock_backend, requests_mock, response).result(aggregation="voting")
    assert result.get_counts() == {"11": 10}


@pytest.mark.parametrize(
    "aggregation,formats",
    [
        (None, [ResultFormat.PROBABILITIES_V2, ResultFormat.PROBABILITIES_V2]),
        ("dnl", [ResultFormat.PROBABILITIES_V2, ResultFormat.PROBABILITIES_V2]),
        ("dnl", [ResultFormat.PROBABILITIES_V2, ResultFormat.HISTOGRAM_V2]),
        ("dnl", [ResultFormat.HISTOGRAM_V2, ResultFormat.PROBABILITIES_V2]),
    ],
)
def test_batch_order_and_mixed_formats(
    mock_backend, requests_mock, aggregation, formats
):
    """Each child's type and input order survive decoding, including mixed formats."""
    parent = _response("batch")
    parent["child_job_ids"] = ["z-first", "a-second"]
    client = mock_backend.client
    expected = []
    totals = []
    for child_id, wire_state, result_format in zip(
        parent["child_job_ids"], ["10", "01"], formats
    ):
        child = _response(child_id)
        is_histogram = result_format == ResultFormat.HISTOGRAM_V2
        key = "histogram" if is_histogram else "probabilities"
        value = 4 if is_histogram else 1.0
        _artifact(
            requests_mock,
            client,
            child,
            result_format,
            {key: {"registers": {"output_all": {wire_state: value}}}},
            aggregation=aggregation,
        )
        requests_mock.get(client.make_path("jobs", child_id), json=child)
        total = 4 if is_histogram else 10
        expected.append({wire_state[::-1]: total})
        totals.append(total)
    result = _job(mock_backend, requests_mock, parent).result(aggregation=aggregation)
    assert result.get_counts() == expected
    assert [experiment.shots for experiment in result.results] == totals


def test_batch_legacy_fallback(mock_backend, requests_mock):
    """A batch with an older child falls back as a unit before fetching artifacts."""
    parent = _response("mixed-batch")
    parent["child_job_ids"] = ["modern", "older"]
    client = mock_backend.client
    modern = _response("modern")
    modern["results"][ResultFormat.PROBABILITIES_V2] = {
        "id": "unused",
        "format": ResultFormat.PROBABILITIES_V2,
    }
    requests_mock.get(client.make_path("jobs", "modern"), json=modern)
    requests_mock.get(client.make_path("jobs", "older"), json=_response("older"))
    requests_mock.get(
        client.make_path("jobs", parent["id"], "results", "probabilities"),
        json={"modern": {"1": 1.0}, "older": {"2": 1.0}},
    )
    result = _job(mock_backend, requests_mock, parent).result()
    assert result.get_counts() == [{"01": 10}, {"10": 10}]


def test_one_child_batch_uses_artifact(mock_backend, requests_mock):
    """A list containing one circuit still carries results on its child."""
    parent = _response("one-parent")
    parent["child_job_ids"] = ["only-child"]
    child = _response("only-child")
    client = mock_backend.client
    _artifact(
        requests_mock,
        client,
        child,
        ResultFormat.PROBABILITIES_V2,
        {"probabilities": {"registers": {"output_all": {"10": 1.0}}}},
    )
    requests_mock.get(client.make_path("jobs", child["id"]), json=child)
    assert _job(mock_backend, requests_mock, parent).result().get_counts() == {"01": 10}


def test_default_simulator_sampling(simulator_backend, requests_mock):
    """V2 probabilities preserve seeded sampling on the ideal simulator."""
    client = simulator_backend.client
    legacy = _response("legacy-sim")
    requests_mock.get(
        client.make_path("jobs", legacy["id"], "results", "probabilities"),
        json={"1": 0.25, "2": 0.75},
    )
    modern = _response("modern-sim")
    _artifact(
        requests_mock,
        client,
        modern,
        ResultFormat.PROBABILITIES_V2,
        {"probabilities": {"registers": {"output_all": {"10": 0.25, "01": 0.75}}}},
    )
    # An advertised shots artifact must not be fetched for ideal simulation.
    modern["results"][ResultFormat.SHOTS_V2] = {"id": "unused"}
    expected = _job(simulator_backend, requests_mock, legacy, memory=True).result()
    actual = _job(simulator_backend, requests_mock, modern, memory=True).result()
    assert actual.get_counts() == expected.get_counts()
    assert actual.get_probabilities() == expected.get_probabilities()
    assert actual.data(0).get("memory") is None


def test_v2_memory_maps_bits_and_filters_leakage(mock_backend, requests_mock):
    """Keep shot order and measurement mapping without replacing mitigated counts."""
    response = _response("mapped")
    response["metadata"]["qiskit_header"] = compress_to_metadata_string(
        {
            "n_qubits": 3,
            "memory_slots": 4,
            "meas_mapped": [2, None, 0, 1],
            "creg_sizes": [["c", 4]],
        }
    )
    client = mock_backend.client
    _artifact(
        requests_mock,
        client,
        response,
        ResultFormat.PROBABILITIES_V2,
        {"probabilities": {"registers": {"output_all": {"011": 1.0}}}},
    )
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
    result = _job(mock_backend, requests_mock, response, memory=True).result()
    assert result.get_memory() == ["0100", "1001", "0100"]
    assert result.get_counts() == {"1001": 10}


@pytest.mark.parametrize(
    "shots", [[], [{"registers": {"output_all": [1, 0]}, "leakage_bits": [1, 0]}]]
)
def test_empty_clean_memory(mock_backend, requests_mock, shots):
    """An empty shot stream, including all-leaked samples, stays empty."""
    response = _response("empty-memory")
    client = mock_backend.client
    _artifact(
        requests_mock,
        client,
        response,
        ResultFormat.PROBABILITIES_V2,
        {"probabilities": {"registers": {"output_all": {"00": 1.0}}}},
    )
    _artifact(requests_mock, client, response, ResultFormat.SHOTS_V2, {"shots": shots})
    assert (
        _job(mock_backend, requests_mock, response, memory=True)
        .result()
        .data(0)["memory"]
        == []
    )


def test_batch_mixed_memory_sources(mock_backend, requests_mock):
    """Memory can come from v2 on one child and legacy shots on another."""
    parent = _response("memory-batch")
    parent["child_job_ids"] = ["new-memory", "old-memory"]
    client = mock_backend.client
    for child_id in parent["child_job_ids"]:
        child = _response(child_id)
        _artifact(
            requests_mock,
            client,
            child,
            ResultFormat.PROBABILITIES_V2,
            {"probabilities": {"registers": {"output_all": {"10": 1.0}}}},
        )
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
    result = _job(mock_backend, requests_mock, parent, memory=True).result()
    assert result.get_memory(0) == ["01"]
    assert result.get_memory(1) == ["10"]


def test_artifact_failure_is_not_fallback(mock_backend, requests_mock):
    """A published artifact that cannot be read must not silently change source."""
    response = _response("forbidden")
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
    with pytest.raises(exceptions.IonQAPIError):
        _job(mock_backend, requests_mock, response).result()


def test_v2_memory_fetch_failure_warns(mock_backend, requests_mock):
    """Optional memory failure leaves counts available, matching legacy behavior."""
    response = _response("memory-forbidden")
    client = mock_backend.client
    _artifact(
        requests_mock,
        client,
        response,
        ResultFormat.PROBABILITIES_V2,
        {"probabilities": {"registers": {"output_all": {"10": 1.0}}}},
    )
    response["results"][ResultFormat.SHOTS_V2] = {"id": "denied"}
    requests_mock.get(
        client.make_path("jobs", response["id"], "artifacts", "denied"),
        status_code=403,
        json={"statusCode": 403, "error": "Forbidden"},
    )
    with pytest.warns(UserWarning, match="Failed to fetch per-shot memory"):
        result = _job(mock_backend, requests_mock, response, memory=True).result()
    assert result.get_counts() == {"01": 10}
    assert result.data(0).get("memory") is None


def test_invalid_shots_raise(mock_backend, requests_mock):
    """Missing output_all is malformed, not an all-zero measurement."""
    response = _response("invalid-shots")
    client = mock_backend.client
    _artifact(
        requests_mock,
        client,
        response,
        ResultFormat.PROBABILITIES_V2,
        {"probabilities": {"registers": {"output_all": {"10": 1.0}}}},
    )
    _artifact(
        requests_mock,
        client,
        response,
        ResultFormat.SHOTS_V2,
        {"shots": [{"registers": {"unrelated": [0]}}]},
    )
    with pytest.raises(exceptions.IonQJobError, match="Invalid v2 shots artifact"):
        _job(mock_backend, requests_mock, response, memory=True).result()


def test_default_extra_query_params(mock_backend, requests_mock):
    """Extra result parameters are forwarded to the artifact request."""
    response = _response("extra-query")
    client = mock_backend.client
    _artifact(
        requests_mock,
        client,
        response,
        ResultFormat.PROBABILITIES_V2,
        {"probabilities": {"registers": {"output_all": {"10": 1.0}}}},
    )
    _job(mock_backend, requests_mock, response).result(
        extra_query_params={"custom": "value"}
    )
    assert requests_mock.last_request.qs == {"custom": ["value"]}
