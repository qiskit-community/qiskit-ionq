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

""" Some utility functions for inspecting IonQ native gates transpiled circuits. """

from qiskit.converters import circuit_to_dag

def compare_circuits(original_circuit, optimized_circuit):
    original_dag = circuit_to_dag(original_circuit)
    optimized_dag = circuit_to_dag(optimized_circuit)

    original_metrics = {
        'depth': original_dag.depth(),
        'size': original_dag.size(),
        'gpi2_count': original_dag.count_ops().get('gpi2', 0),
        'gpi_count': original_dag.count_ops().get('gpi', 0),
        'ms_count': original_dag.count_ops().get('ms', 0),
        'zz_count': original_dag.count_ops().get('zz', 0)
    }

    optimized_metrics = {
        'depth': optimized_dag.depth(),
        'size': optimized_dag.size(),
        'gpi2_count': optimized_dag.count_ops().get('gpi2', 0),
        'gpi_count': optimized_dag.count_ops().get('gpi', 0),
        'ms_count': optimized_dag.count_ops().get('ms', 0),
        'zz_count': optimized_dag.count_ops().get('zz', 0)
    }

    print(f"The circuit size has reduced from {original_metrics.get('size')} to {optimized_metrics.get('size')}")
    
    return original_metrics, optimized_metrics

def print_metrics(metrics):
    print(f"- Depth: {metrics['depth']}")
    print(f"- Size: {metrics['size']}")
    print(f"- GPI2 Count: {metrics['gpi2_count']}")
    print(f"- GPI Count: {metrics['gpi_count']}")
    print(f"- MS Count: {metrics['ms_count']}")
    print(f"- ZZ Count: {metrics['zz_count']}")
    
