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

# Copyright 2024 IonQ, Inc. (www.ionq.com)
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

"""Test IonQ transpiler."""

import pytest
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit_ionq import IonQProvider, ionq_transpile

SEED = 123  # master seed for reproducibility


@pytest.fixture(name="backend")
def _backend():
    """Fixture for IonQ backend."""
    provider = IonQProvider()
    return provider.get_backend("qpu.aria-1", gateset="native")


def test_ionq_transpile(backend):
    """Transpile a simple circuit."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    qiskit_transpiled = transpile(
        qc, backend=backend, optimization_level=3, seed_transpiler=SEED
    )
    ionq_transpiled = ionq_transpile(
        qc, backend=backend, optimization_level=3, seed_transpiler=SEED
    )

    assert ionq_transpiled.depth() < qiskit_transpiled.depth()
    assert ionq_transpiled.num_qubits == qc.num_qubits


def test_random_circuit_transpile(backend):
    """Reproducible transpile on many random circuits."""
    # Local RNG so results don't depend on test order or other tests.
    rng = np.random.default_rng(SEED)

    for _ in range(100):
        # Derive per-circuit seeds deterministically from the master seed.
        rc_seed = int(rng.integers(0, 2**32 - 1))

        qc = random_circuit(
            num_qubits=int(rng.integers(2, 4)),
            depth=int(rng.integers(6, 10)),
            max_operands=4,
            measure=bool(rng.integers(0, 2)),
            seed=rc_seed,  # <- makes the circuit’s internal randomness reproducible
        )

        qiskit_transpiled = transpile(
            qc, backend=backend, optimization_level=3, seed_transpiler=SEED
        )
        ionq_transpiled = ionq_transpile(
            qc, backend=backend, optimization_level=3, seed_transpiler=SEED
        )

        assert ionq_transpiled.depth() < qiskit_transpiled.depth()
        assert ionq_transpiled.num_qubits == qc.num_qubits
