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

"""Tests for JobEstimate payload parsing."""

from qiskit_ionq.ionq_client import JobEstimate


def test_cost_reads_estimated_total_cost():
    """cost comes from the "estimated_total_cost" key the API actually returns."""
    est = JobEstimate({"estimated_total_cost": 2254.88})
    assert est.cost == 2254.88


def test_cost_falls_back_to_estimated_cost():
    """The older "estimated_cost" spelling still populates cost."""
    est = JobEstimate({"estimated_cost": 1.23})
    assert est.cost == 1.23


def test_estimated_total_cost_wins_over_estimated_cost():
    """When both keys are present the documented one takes precedence."""
    est = JobEstimate({"estimated_total_cost": 10.0, "estimated_cost": 99.0})
    assert est.cost == 10.0


def test_cost_is_none_when_absent():
    """A payload carrying neither key leaves cost unset rather than raising."""
    est = JobEstimate({})
    assert est.cost is None


def test_realistic_payload_is_fully_parsed():
    """A representative GET /jobs/estimate body populates every attribute."""
    est = JobEstimate(
        {
            "input_values": {
                "backend": "qpu.forte-1",
                "1q_gates": 285,
                "2q_gates": 11,
                "qubits": 8,
                "shots": 1100,
                "error_mitigation": False,
                "session": False,
            },
            "estimated_total_cost": 2254.88,
            "estimated_execution_time": 42.5,
            "current_predicted_queue_time": 1250.0,
        }
    )

    assert est.cost == 2254.88
    assert est.backend == "qpu.forte-1"
    assert est.oneq_gates == 285
    assert est.twoq_gates == 11
    assert est.qubits == 8
    assert est.shots == 1100
    assert est.total_runtime == 42.5 + 1250.0


def test_repr_omits_cost_unit_when_absent():
    """Without a unit the repr reads "cost=2254.88", not "cost=2254.88 None"."""
    est = JobEstimate({"estimated_total_cost": 2254.88})
    assert "cost=2254.88 None" not in repr(est)
    assert "cost=2254.88" in repr(est)


def test_repr_includes_cost_unit_when_present():
    """A unit supplied by the API is still shown."""
    est = JobEstimate({"estimated_total_cost": 2254.88, "cost_unit": "USD"})
    assert "cost=2254.88 USD" in repr(est)
