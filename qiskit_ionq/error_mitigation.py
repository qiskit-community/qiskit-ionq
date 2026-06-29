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

"""Error mitigation config types."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class PhiChiPattern(enum.Enum):
    """Two-qubit gate twirling pattern."""

    CHI_ONLY = "chi_only"
    STANDARD = "standard"
    ALTERNATIVE = "alternative"
    EXTENDED = "extended"


class OneQubitTwirling(enum.Enum):
    """Single-qubit gate twirling strategy."""

    NONE = "none"
    DECOMPOSITION = "decomposition"
    ORDER = "order"
    BOTH = "both"


class AggregationMethod(enum.Enum):
    """Aggregation method for results from a debiased job."""

    AVERAGE = "average"
    VOTING = "voting"
    DNL = "dnl"


@dataclass
class TwirlingConfig:
    """Twirling options for debiasing."""

    pattern: str | PhiChiPattern | None = None
    one_qubit: str | OneQubitTwirling = OneQubitTwirling.NONE

    def to_dict(self) -> dict:
        """Serialize to API wire format."""
        result = {}
        if self.pattern is not None:
            result["pattern"] = (
                self.pattern.value
                if isinstance(self.pattern, PhiChiPattern)
                else self.pattern
            )
        result["one_qubit_twirling"] = (
            self.one_qubit.value
            if isinstance(self.one_qubit, OneQubitTwirling)
            else self.one_qubit
        )
        return result


@dataclass
class DebiasingConfig:
    """Fine-grained debiasing options."""

    num_variants: int | None = None
    twirling: TwirlingConfig | None = None

    def to_dict(self) -> dict:
        """Serialize to API wire format."""
        result: dict = {"debiasing": True}
        if self.num_variants is not None:
            result["num_variants"] = self.num_variants
        if self.twirling is not None:
            result["phi_chi_twirling"] = self.twirling.to_dict()
        return result


@dataclass
class ErrorMitigationConfig:
    """Bundle of all error mitigation configuration."""

    debiasing: bool | DebiasingConfig = True
    symmetry_verification: bool = True

    def to_dict(self) -> dict:
        """Serialize to API wire format."""
        result = {"symmetry_verification": self.symmetry_verification}

        if isinstance(self.debiasing, bool):
            result["debiasing"] = self.debiasing
        else:
            result.update(self.debiasing.to_dict())

        return result


__all__ = [
    "AggregationMethod",
    "DebiasingConfig",
    "ErrorMitigationConfig",
    "OneQubitTwirling",
    "PhiChiPattern",
    "TwirlingConfig",
]
