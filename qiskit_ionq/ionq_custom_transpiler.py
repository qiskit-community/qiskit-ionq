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

from qiskit.transpiler import PassManager, PassManagerConfig
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin

from qiskit_ionq.rewrite_rules import (
    GPI2_Adjoint,
    GPI_Adjoint,
    CommuteGPI2MS,
    CancelFourGPI2,
    GPI2TwiceIsGPI,
    CollapseMoreThanThreeSingleQubitGates,
)


class TrappedIonOptimizerPlugin(PassManagerStagePlugin):
    def pass_manager(
        self, pass_manager_config: PassManagerConfig, optimization_level: int = 0
    ) -> PassManager:
        custom_pass_manager = PassManager()
        if optimization_level == 0:
            pass
        if optimization_level >= 1:
            custom_pass_manager.append(GPI2_Adjoint())
            custom_pass_manager.append(GPI_Adjoint())
            custom_pass_manager.append(CommuteGPI2MS())
            custom_pass_manager.append(CancelFourGPI2())
            custom_pass_manager.append(GPI2TwiceIsGPI())
            custom_pass_manager.append(CollapseMoreThanThreeSingleQubitGates())
        return custom_pass_manager

    # def __init__(self, backend):
    #     """A custom transpiler that optimizes a circuit composed of IonQ native gates
    #     and reduces the number of gates in the circuit."""
    #     self.backend = backend
    #     self.pass_manager = self.custom_pass_manager()

    # @staticmethod
    # def pass_manager(self, pass_manager_config: PassManagerConfig, optimization_level: int = 0):
    #     custom_pass_manager = PassManager()
    #     custom_pass_manager.append([
    #         GPI2_Adjoint(),
    #         GPI_Adjoint(),
    #         CommuteGPI2MS(),
    #         CancelFourGPI2(),
    #         GPI2TwiceIsGPI(),
    #     ])
    #     return custom_pass_manager

    # def transpile(self, qc, optimization_level=1):

    #     ibm_transpiled = transpile(qc, backend=self.backend, optimization_level=optimization_level)
    #     optimized_circuit = self.pass_manager.run(ibm_transpiled)

    #     # Run the pass manager until no further optimizations are possible
    #     while True:
    #         previous_dag = circuit_to_dag(optimized_circuit)
    #         optimized_circuit = self.pass_manager.run(optimized_circuit)
    #         if circuit_to_dag(optimized_circuit) == previous_dag:
    #             break

    #     return optimized_circuit
