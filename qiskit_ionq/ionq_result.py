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

"""
IonQ result implementation that extends to allow for retrieval of probabilities.
"""

from qiskit.exceptions import QiskitError
from qiskit.result import Result
from qiskit.result.counts import Counts
from . import exceptions


class IonQResult(Result):
    """
    An IonQ-specific result object.

    The primary reason this class extends the base Qiskit result object is to
    provide an API for retrieving result probabilities directly.
    """

    def get_probabilities(self, experiment=None):
        """
        Get probabilities for the experiment.

        If ``experiment`` is None, ``self.results`` will be used in its place.

        Args:
            experiment (Union[int, QuantumCircuit, Schedule, dict], optional): If provided, this
                argument is used to get an experiment using Result's ``_get_experiment`` method.

        Raises:
            IonQJobError: A given experiment in our results had no probabilities.

        Returns:
            Union[Counts, list(Counts)]: Single count if the result list
                was size one, else the entire list.
        """
        if experiment is None:
            exp_keys = range(len(self.results))
        else:
            exp_keys = [experiment]

        dict_list = []
        for key in exp_keys:
            exp = self._get_experiment(key)
            try:
                header = exp.header.to_dict()
            except (AttributeError, QiskitError):  # header is not available
                header = None

            if "probabilities" in self.data(key).keys():
                counts_header = {}
                if header:
                    counts_header = {
                        k: v
                        for k, v in header.items()
                        if k in {"time_taken", "creg_sizes", "memory_slots"}
                    }
                dict_list.append(
                    Counts(self.data(key)["probabilities"], **counts_header)
                )
            else:
                raise exceptions.IonQJobError(
                    f'No probabilities for experiment "{repr(key)}"'
                )

        # Return first item of dict_list if size is 1
        if len(dict_list) == 1:
            return dict_list[0]
        return dict_list

    def get_memory(self, experiment=None, *, drop_leaked=False):
        """
        Get the experiment's per-shot memory.

        Memory keeps every recorded shot, including shots the hardware
        flagged as leaked (an ion left the qubit states mid-shot).
        ``data(experiment)["leaked"]`` holds the per-shot flags, aligned with
        memory.

        Args:
            experiment (Union[int, QuantumCircuit, Schedule, dict], optional): If provided, this
                argument is used to get an experiment using Result's ``_get_experiment`` method.
            drop_leaked (bool): Drop the shots the hardware flagged as leaked.

        Raises:
            QiskitError: If the experiment has no memory, or ``drop_leaked`` is
                set and its shots have no leak flags (they predate v2 result
                artifacts).

        Returns:
            list[str]: One bitstring per shot.
        """
        memory = super().get_memory(experiment)
        if not drop_leaked:
            return memory
        leaked = self.data(experiment).get("leaked")
        if leaked is None:
            raise QiskitError(
                f'No leakage flags for experiment "{experiment!r}"; '
                "its shots predate v2 result artifacts."
            )
        return [
            shot
            for shot, is_leaked in zip(memory, leaked, strict=True)
            if not is_leaked
        ]
