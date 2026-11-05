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

import json
import re
import uuid

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import library
from qiskit.quantum_info import Statevector

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


# IonQ QIS wire name -> (Qiskit gate class, takes a rotation angle).
_QIS_GATES = {
    "h": (library.HGate, False),
    "x": (library.XGate, False),
    "y": (library.YGate, False),
    "z": (library.ZGate, False),
    "s": (library.SGate, False),
    "si": (library.SdgGate, False),
    "t": (library.TGate, False),
    "ti": (library.TdgGate, False),
    "v": (library.SXGate, False),
    "vi": (library.SXdgGate, False),
    "swap": (library.SwapGate, False),
    "rx": (library.RXGate, True),
    "ry": (library.RYGate, True),
    "rz": (library.RZGate, True),
    "xx": (library.RXXGate, True),
    "yy": (library.RYYGate, True),
    "zz": (library.RZZGate, True),
}


def _simulate(gates, num_qubits):
    """Probabilities of an IonQ QIS gate list; index ``i`` has qubit 0 as LSB."""
    circ = QuantumCircuit(num_qubits)
    for inst in gates:
        gate_cls, rotates = _QIS_GATES[inst["gate"]]
        if inst["gate"] == "z" and "rotation" in inst:
            gate = library.PhaseGate(inst["rotation"])  # "p" is sent as "z"
        else:
            gate = gate_cls(inst["rotation"]) if rotates else gate_cls()
        controls = inst.get("controls", [])
        if controls:
            gate = gate.control(len(controls))
        circ.append(gate, controls + inst["targets"])
    return Statevector(circ).probabilities()


class FakeIonQCloud:
    """Simulates each circuit POSTed to ``/jobs`` and serves the results.

    The result shapes match IonQ Cloud's: the ideal simulator serves exact
    probabilities, a QPU (as forte-1 does) serves sampled probabilities only,
    and only a noisy simulator also serves per-shot data. A job with several
    circuits keeps each circuit's results on a child; a one-circuit list has
    no children.
    """

    def __init__(self, requests_mock, client, seed=1234):
        self._rng = np.random.default_rng(seed)
        self._jobs = {}
        self._results = {}
        self.submitted = []
        requests_mock.post(client.make_path("jobs"), json=self._submit)
        requests_mock.get(re.compile(r".*/jobs/[^/]+$"), json=self._get_job)
        requests_mock.get(re.compile(r".*/results/.*"), json=self._get_results)

    def _run(self, job_id, gates, num_qubits, shots, noise):
        """Store one circuit's results; return its ``results`` URLs.

        ``noise`` is the payload's noise model, or ``None`` for a QPU.
        """
        probs = _simulate(gates, num_qubits)
        urls = {"probabilities": {"url": f"/v0.4/jobs/{job_id}/results/probabilities"}}
        if noise == "ideal":
            self._results[f"{job_id}/probabilities"] = {
                str(k): p for k, p in enumerate(probs) if p > 1e-12
            }
            return urls
        raw = self._rng.choice(len(probs), size=shots, p=probs / probs.sum())
        values, freq = np.unique(raw, return_counts=True)
        self._results[f"{job_id}/probabilities"] = {
            str(v): f / shots for v, f in zip(values.tolist(), freq.tolist())
        }
        if noise is None:
            return urls
        self._results[f"{job_id}/shots"] = raw.tolist()
        return {**urls, "shots": {"url": f"/v0.4/jobs/{job_id}/results/shots"}}

    def _submit(self, request, _context):
        payload = json.loads(request.body)
        self.submitted.append(payload)
        job_id = str(uuid.uuid4())
        noise = (payload.get("noise") or {}).get("model")
        circuits = payload["input"].get("circuits") or [payload["input"]]
        num_qubits, shots = payload["input"]["qubits"], payload["shots"]
        job = {
            "id": job_id,
            "status": "completed",
            "type": payload["type"],
            "metadata": payload["metadata"],
            "noise": payload.get("noise"),
            "stats": {"qubits": num_qubits},
            "execution_duration_ms": 1,
        }
        if len(circuits) > 1:
            children = {}
            for circuit in circuits:
                child_id = str(uuid.uuid4())
                results = self._run(
                    child_id, circuit["circuit"], num_qubits, shots, noise
                )
                self._jobs[child_id] = {
                    "id": child_id,
                    "status": "completed",
                    "results": results,
                }
                children[child_id] = self._results[f"{child_id}/probabilities"]
            self._results[f"{job_id}/probabilities/aggregated"] = children
            job["child_job_ids"] = list(children)
            url = f"/v0.4/jobs/{job_id}/results/probabilities/aggregated"
            job["results"] = {"probabilities": {"url": url}}
        else:
            gates = circuits[0]["circuit"]
            job["results"] = self._run(job_id, gates, num_qubits, shots, noise)
        self._jobs[job_id] = job
        return {"id": job_id, "status": "submitted"}

    def _get_job(self, request, context):
        job_id = request.path.rsplit("/", 1)[-1]
        if job_id not in self._jobs:
            context.status_code = 404
            return {"statusCode": 404, "error": "Not Found", "message": "Not Found"}
        return self._jobs[job_id]

    def _get_results(self, request, _context):
        job_id, kind = re.match(r".*/jobs/([^/]+)/results/(.*)", request.path).groups()
        return self._results[f"{job_id}/{kind}"]
