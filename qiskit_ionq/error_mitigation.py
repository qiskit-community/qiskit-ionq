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
from dataclasses import dataclass, field


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
class Twirling:
    """Two-qubit gate twirling options for debiasing."""

    pattern: str | PhiChiPattern | None = None
    one_qubit: str | OneQubitTwirling = OneQubitTwirling.NONE

    def to_dict(self) -> dict:
        d = {}
        if self.pattern is not None:
            p = self.pattern
            d["pattern"] = p.value if isinstance(p, PhiChiPattern) else p
        oq = self.one_qubit
        d["one_qubit_twirling"] = oq.value if isinstance(oq, OneQubitTwirling) else oq
        return d


@dataclass
class Debiasing:
    num_variants: int | None = None
    twirling: Twirling | None = None

    def to_dict(self) -> dict:
        d = {"debiasing": True}
        if self.num_variants is not None:
            d["num_variants"] = self.num_variants
        if self.twirling is not None:
            d["phi_chi_twirling"] = self.twirling.to_dict()
        return d


@dataclass
class ErrorMitigationConfig:
    """Bundle of all error mitigation options, for use with
    ``backend.options.update_options()`` or as the ``error_mitigation`` kwarg
    on ``backend.run()``.
    """

    debiasing: bool | Debiasing = True
    symmetry_verification: bool = True

    def to_dict(self) -> dict:
        d = {}
        if isinstance(self.debiasing, Debiasing):
            d.update(self.debiasing.to_dict())
        else:
            d["debiasing"] = bool(self.debiasing)
        d["symmetry_verification"] = self.symmetry_verification
        return d


__all__ = [
    "AggregationMethod",
    "Debiasing",
    "ErrorMitigationConfig",
    "OneQubitTwirling",
    "PhiChiPattern",
    "Twirling",
]
