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
from qiskit.converters import circuit_to_dag

from qiskit_ionq.rewrite_rules import (
    GPI2_Adjoint,
    GPI_Adjoint,
    GPI2TwiceIsGPI,
    CompactMoreThanThreeSingleQubitGates,
    CommuteGPIsThroughMS,
)


class CustomPassManager(PassManager):
    def run(self, circuit):
        """
        Runs the pass manager iteratively until no further optimizations are possible.
        """
        optimized_circuit = super().run(circuit)

        while True:
            previous_dag = circuit_to_dag(optimized_circuit)
            optimized_circuit = super().run(optimized_circuit)
            current_dag = circuit_to_dag(optimized_circuit)

            if current_dag == previous_dag:
                break

        return optimized_circuit


class TrappedIonOptimizerPluginSimpleRules(PassManagerStagePlugin):
    """
    This class is no intended to be used in production, is is meant
     to test the following transformation passes in isolation:
        - GPI2_Adjoint
        - GPI_Adjoint
        - GPI2TwiceIsGPI
    """

    def pass_manager(
        self, pass_manager_config: PassManagerConfig, optimization_level: int = 0
    ) -> PassManager:
        """
        Creates a PassManager class with added custom transformation passes.
        """
        custom_pass_manager = CustomPassManager()
        if optimization_level == 0:
            pass
        if optimization_level >= 1:
            custom_pass_manager.append(GPI2_Adjoint())
            custom_pass_manager.append(GPI_Adjoint())
            custom_pass_manager.append(GPI2TwiceIsGPI())
        return custom_pass_manager


class TrappedIonOptimizerPluginCompactGates(PassManagerStagePlugin):
    """
    This class is no intended to be used in production, is is meant
     to test the following transformation passes in isolation:
        - CompactMoreThanThreeSingleQubitGates
    """

    def pass_manager(
        self, pass_manager_config: PassManagerConfig, optimization_level: int = 0
    ) -> PassManager:
        """
        Creates a PassManager class with added custom transformation passes.
        """
        custom_pass_manager = CustomPassManager()
        if optimization_level == 0:
            pass
        if optimization_level >= 1:
            custom_pass_manager.append(CompactMoreThanThreeSingleQubitGates())
        return custom_pass_manager


class TrappedIonOptimizerPluginCommuteGpi2ThroughMs(PassManagerStagePlugin):
    def pass_manager(
        self, pass_manager_config: PassManagerConfig, optimization_level: int = 0
    ) -> PassManager:
        """
        Creates a PassManager class with added custom transformation passes.
        This class is meant to be used in production.
        """
        custom_pass_manager = CustomPassManager()
        if optimization_level == 0:
            pass
        if optimization_level >= 1:
            custom_pass_manager.append(CommuteGPIsThroughMS())
        return custom_pass_manager


class TrappedIonOptimizerPlugin(PassManagerStagePlugin):
    def pass_manager(
        self, pass_manager_config: PassManagerConfig, optimization_level: int = 0
    ) -> PassManager:
        """
        Creates a PassManager class with added custom transformation passes.
        This class is meant to be used in production.
        """
        custom_pass_manager = CustomPassManager()
        if optimization_level == 0:
            pass
        if optimization_level >= 1:
            # Note the TransformationPasses should be applied
            # in the order were added to the PassManager
            custom_pass_manager.append(GPI2_Adjoint())
            custom_pass_manager.append(GPI_Adjoint())
            custom_pass_manager.append(GPI2TwiceIsGPI())
            custom_pass_manager.append(CommuteGPIsThroughMS())
            custom_pass_manager.append(CompactMoreThanThreeSingleQubitGates())
        return custom_pass_manager
