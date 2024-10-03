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


from qiskit import transpile
from qiskit.transpiler import PassManager
from qiskit.converters import circuit_to_dag

from qiskit_ionq.rewrite_rules import (
    GPI2_Adjoint,
    GPI_Adjoint,
    CommuteGPI2MS,
    CancelFourGPI2,
    GPI2TwiceIsGPITimesI,
    SimplifyIds,
    CommuteGPisAndIds,
    CommuteMSAndIds,
)


class IonQTranspiler:
    """A custom transpiler that optimizes a circuit composed of IonQ native gates
    and reduces the number of gates in the circuit."""

    def __init__(self, backend):
        self.backend = backend
        self.full_pass_manager = self.full_custom_pass_manager()

    @staticmethod
    def initial_custom_pass_manager():
        """A pass manager intended to optimize native gates which adds +i*Id, -i*Id and -Id
        gates where necessary. When this is pass is complete the +i*Id, -i*Id and -Id gates
        should be found only at the end of the circuit since commutation rules should achieve
        this result.
        """
        pm = PassManager()
        pm.append(
            [
                GPI2_Adjoint(),
                GPI_Adjoint(),
                CommuteGPI2MS(),
                CancelFourGPI2(),
                GPI2TwiceIsGPITimesI(),
                SimplifyIds(),
                CommuteGPisAndIds(),
                CommuteMSAndIds(),
            ]
        )
        return pm

    @staticmethod
    def final_custom_pass_manager():
        """A pass manager intended to optmize the gates in circuit on the right-end
        after +i*Id, -i*Id and -Id have been replaced by IonQ native gates."""
        pm = PassManager()
        pm.append(
            [
                GPI2_Adjoint(),
                GPI_Adjoint(),
                CommuteGPI2MS(),
                CancelFourGPI2(),
            ]
        )
        return pm

    def transpile(self, qc, optimization_level=1):
        """Transpile and optimize a circuit."""
        ibm_transpiled = transpile(
            qc, backend=self.backend, optimization_level=optimization_level
        )
        optimized_circuit = self.initial_custom_pass_manager.run(ibm_transpiled)

        # Run the initial pass manager until no further optimizations are possible
        while True:
            previous_dag = circuit_to_dag(optimized_circuit)
            optimized_circuit = self.initial_custom_pass_manager.run(optimized_circuit)
            if circuit_to_dag(optimized_circuit) == previous_dag:
                break

        # Run the final pass manager until no further optimizations are possible
        while True:
            previous_dag = circuit_to_dag(optimized_circuit)
            optimized_circuit = self.final_custom_pass_manager.run(optimized_circuit)
            if circuit_to_dag(optimized_circuit) == previous_dag:
                break

        return optimized_circuit
