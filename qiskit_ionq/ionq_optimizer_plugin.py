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

"""Optimize the transpiling of IonQ native gates using a custom
pass manager and a pass manager plugin TrappedIonOptimizerPlugin
which consolidates all the rewrite rules in one single plugin.
The other plugin classes are intended for testing various rewrite
rules in isolation.
"""

from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.converters import circuit_to_dag
from qiskit_ionq.rewrite_rules import (
    CancelGPI2Adjoint,
    CancelGPIAdjoint,
    GPI2TwiceIsGPI,
    CompactMoreThanThreeSingleQubitGates,
    CommuteGPIsThroughMS,
)


class CustomPassManager(PassManager):
    """A custom pass manager."""

    # pylint: disable=arguments-differ
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
    This class is not intended to be used in production, is meant
     to test the following transformation passes in isolation:
        - CancelGPI2Adjoint
        - CancelGPIAdjoint
        - GPI2TwiceIsGPI
    """

    def pass_manager(
        self,
        pass_manager_config: PassManagerConfig = None,
        optimization_level: int = 0,
    ) -> PassManager:  # pylint: disable=unused-argument
        """
        Creates a PassManager class with added custom transformation passes.
        """
        custom_pass_manager = CustomPassManager()
        if optimization_level == 0:
            pass
        if optimization_level >= 1:
            custom_pass_manager.append(CancelGPI2Adjoint())
            custom_pass_manager.append(CancelGPIAdjoint())
            custom_pass_manager.append(GPI2TwiceIsGPI())
        return custom_pass_manager


class TrappedIonOptimizerPluginCompactGates(PassManagerStagePlugin):
    """
    This class is not intended to be used in production, is meant
     to test the following transformation passes in isolation:
        - CompactMoreThanThreeSingleQubitGates
    """

    def pass_manager(
        self,
        pass_manager_config: PassManagerConfig = None,
        optimization_level: int = 0,
    ) -> PassManager:  # pylint: disable=unused-argument
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
    """
    This class is not intended to be used in production, is meant
     to test the following transformation passes in isolation:
        - CommuteGPIsThroughMS
    """

    def pass_manager(
        self,
        pass_manager_config: PassManagerConfig = None,
        optimization_level: int = 0,
    ) -> PassManager:  # pylint: disable=unused-argument
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
    """
    This optimizer plugin is intended to be used in production.
    """

    def pass_manager(
        self,
        pass_manager_config: PassManagerConfig = None,
        optimization_level: int = 0,
    ) -> PassManager:  # pylint: disable=unused-argument
        """
        Creates a PassManager class with added custom transformation passes.
        This class is meant to be used in production.
        """
        custom_pass_manager = CustomPassManager()
        if optimization_level == 0:
            pass
        if optimization_level >= 1:
            # Note that the TransformationPasses will be applied
            # in the order they have been added to the pass manager
            custom_pass_manager.append(CancelGPI2Adjoint())
            custom_pass_manager.append(CancelGPIAdjoint())
            custom_pass_manager.append(GPI2TwiceIsGPI())
            custom_pass_manager.append(CommuteGPIsThroughMS())
            custom_pass_manager.append(CompactMoreThanThreeSingleQubitGates())
        return custom_pass_manager
