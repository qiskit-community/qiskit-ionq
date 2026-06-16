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

# Copyright 2026 IonQ, Inc. (www.ionq.com)
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

"""Qiskit v1/v2 compatibility helpers."""

from __future__ import annotations

from typing import Any, cast

from qiskit.quantum_info import SparsePauliOp

try:
    from qiskit.result import MeasLevel
except ImportError:
    MEAS_LEVEL_CLASSIFIED = 2
else:
    MEAS_LEVEL_CLASSIFIED = MeasLevel.CLASSIFIED

try:
    SPARSE_PAULI_FROM_SPARSE_OBSERVABLE = SparsePauliOp.from_sparse_observable
except AttributeError:
    SPARSE_PAULI_FROM_SPARSE_OBSERVABLE = None

SUPPORTS_SPARSE_OBSERVABLE_PAULI_EVOLUTION = (
    SPARSE_PAULI_FROM_SPARSE_OBSERVABLE is not None
)


class ResultHeader(dict):
    """Dict header that also satisfies Qiskit v1 ``to_dict`` calls."""

    def to_dict(self) -> dict:
        """Return the header as a plain dict."""
        return dict(self)


def header_to_dict(header: Any) -> dict | None:
    """Return a result header as a plain dict across Qiskit versions."""
    if header is None:
        return None
    if isinstance(header, dict):
        return header
    try:
        return header.to_dict()
    except AttributeError:
        return dict(header)


def normalize_result_headers(result: Any) -> Any:
    """Normalize Qiskit Result experiment headers to dicts in v1 and v2."""
    for experiment in result.results:
        header = experiment.header
        if header is None or isinstance(header, dict):
            continue
        header_dict = cast(dict, header_to_dict(header))
        experiment.header = ResultHeader(header_dict)
    return result


__all__ = [
    "MEAS_LEVEL_CLASSIFIED",
    "ResultHeader",
    "SPARSE_PAULI_FROM_SPARSE_OBSERVABLE",
    "SUPPORTS_SPARSE_OBSERVABLE_PAULI_EVOLUTION",
    "header_to_dict",
    "normalize_result_headers",
]
