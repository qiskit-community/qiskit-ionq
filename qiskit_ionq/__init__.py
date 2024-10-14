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

"""Provider for IonQ backends"""

# warn if qiskit is not installed
try:
    from qiskit.version import get_version_info  # pylint: disable=unused-import
except ImportError as exc:
    raise ImportError(
        "Qiskit is not installed. Please install the latest version of Qiskit."
    ) from exc

from .ionq_provider import IonQProvider
from .version import __version__
from .ionq_gates import GPIGate, GPI2Gate, MSGate, ZZGate
from .constants import ErrorMitigation
from .ionq_equivalence_library import add_equivalences

__all__ = [
    "__version__",
    "IonQProvider",
    "GPIGate",
    "GPI2Gate",
    "MSGate",
    "ZZGate",
    "ErrorMitigation",
    "add_equivalences",
]
