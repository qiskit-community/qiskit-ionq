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

"""Shared (non-fixture) test helpers, importable from any test module."""

from qiskit_ionq.helpers import compress_to_metadata_string


def _def_results_template(job_id):
    """A template for the results field in a job response."""
    return {
        "histogram": {
            # v0.4 returns a relative path - the client prefixes it with the base URL
            # https://docs.ionq.com/api-reference/v0.4/jobs/get-job
            "url": f"/v0.4/jobs/{job_id}/results/histogram"
        },
        "probabilities": {"url": f"/v0.4/jobs/{job_id}/results/probabilities"},
        "shots": {"url": f"/v0.4/jobs/{job_id}/results/shots"},
    }


def dummy_job_response(
    job_id, target="mock_backend", status="completed", job_settings=None, children=None
):
    """A dummy response payload for `job_id`.

    Args:
        job_id (str): An arbitrary job id.
        target (str): Backend target string.
        status (str): A provided status string.
        job_settings (dict): Settings provided to the API.
        children (list): A list of child job IDs.

    Returns:
        dict: A json response dict.
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
    response = {
        "status": status,
        "predicted_execution_time": 4,
        "metadata": {
            "qobj_id": "test_qobj_id",
            "shots": "1234",
            "sampler_seed": "42",
            "output_length": "2",
            "qiskit_header": qiskit_header,
        },
        "execution_time": 8,
        "qubits": 2,
        "type": "circuit",
        "request": 1600000000,
        "start": 1600000001,
        "response": 1600000002,
        "backend": target,
        "results": _def_results_template(job_id),
        "id": job_id,
        "settings": (job_settings or {}),
        "name": "test_name",
    }

    if children is not None:
        response["children"] = children

    return response
