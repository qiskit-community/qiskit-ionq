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

"""OpenQASM 3 annotation support for IonQ-specific compiler directives.

Tempo's compiled-circuit output carries IonQ-specific directives that don't
exist as quantum gates. The OpenQASM 3 spec recommends representing them as
namespaced annotations on a host statement (``box`` for position-sensitive
directives), and Qiskit 2.1+ supports round-tripping annotations on
``box`` statements via a per-namespace
:class:`~qiskit.circuit.annotation.OpenQASM3Serializer`.

Currently exposed:

* ``@ionq.arrange`` — ion-parcel rearrangement inserted by the Cypress
  compiler. Carries the source and destination arrangement labels plus the
  qubits the operation touches. Only ever appears on the readback path
  (:meth:`qiskit_ionq.ionq_job.IonQJob.compiled_circuit`); external users
  cannot include it on submission.

Concrete QASM 3 form::

    @ionq.arrange {"from": "A12S", "to": "A23S", "targets": [0, 1, 2]}
    box {
    }

References:
    OpenQASM 3.1 spec, §Directives (annotations + ``box``):
      https://openqasm.com/language/directives.html
    Qiskit 2.1 annotation framework:
      https://docs.quantum.ibm.com/api/qiskit/circuit_annotation
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from qiskit.circuit.annotation import Annotation, OpenQASM3Serializer


IONQ_ARRANGE_NAMESPACE = "ionq.arrange"


@dataclass(frozen=True)
class IonQArrangeAnnotation(Annotation):
    """An ``@ionq.arrange`` directive parsed from a compiled QASM 3 program.

    Attributes:
        from_label: Source arrangement identifier (e.g. ``"A12S"``).
        to_label: Destination arrangement identifier (e.g. ``"A23S"``).
        targets: Qubit indices the rearrangement involves.

    These annotations ride on empty :class:`~qiskit.circuit.BoxOp` instances
    in the round-tripped :class:`~qiskit.circuit.QuantumCircuit`; the box
    body is empty because the underlying operation is metadata, not a
    quantum computation.
    """

    from_label: str
    to_label: str
    targets: tuple[int, ...]

    # The Annotation base class declares ``namespace`` as a str attribute; we
    # need a default so the frozen dataclass can be instantiated positionally
    # without users restating the namespace each time.
    namespace: str = IONQ_ARRANGE_NAMESPACE


class IonQArrangeSerializer(OpenQASM3Serializer):
    """Serialise / deserialise :class:`IonQArrangeAnnotation` to OpenQASM 3.

    Payload format is a JSON object: ``{"from": ..., "to": ..., "targets": [...]}``.
    JSON is chosen because (a) it fits OpenQASM 3's "free-form, no newlines"
    annotation-payload contract, (b) it round-trips faithfully through the
    OpenQASM 3 lexer, and (c) it stays self-describing when the schema grows
    (additional arrangement metadata can be added without breaking older
    consumers).
    """

    def dump(self, annotation: Annotation) -> Any:
        """Emit ``{"from": <str>, "to": <str>, "targets": [<int>, ...]}``.

        Called by :func:`qiskit.qasm3.dumps` when emitting a circuit that
        contains an :class:`IonQArrangeAnnotation`. Returns
        :data:`NotImplemented` for any other annotation namespace so Qiskit
        can route to the right serializer; otherwise returns a single-line
        JSON string.
        """
        if not isinstance(annotation, IonQArrangeAnnotation):
            return NotImplemented
        return json.dumps(
            {
                "from": annotation.from_label,
                "to": annotation.to_label,
                "targets": list(annotation.targets),
            },
            separators=(",", ":"),
        )

    def load(self, namespace: str, payload: str) -> Any:
        """Parse a JSON ``@ionq.arrange`` payload into an :class:`IonQArrangeAnnotation`.

        Called by :func:`qiskit.qasm3.loads` (or :func:`qiskit_qasm3_import.parse`)
        when it encounters an annotation in the ``ionq.arrange`` namespace.
        Missing or malformed fields raise :class:`ValueError` so a corrupt
        compiled-circuit response surfaces as a clear parser error rather
        than a silently-misparsed annotation.
        """
        if namespace != IONQ_ARRANGE_NAMESPACE:
            return NotImplemented
        data: dict[str, Any] = json.loads(payload)
        try:
            return IonQArrangeAnnotation(
                from_label=str(data["from"]),
                to_label=str(data["to"]),
                targets=tuple(int(t) for t in data["targets"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Malformed @ionq.arrange annotation payload: {payload!r}. "
                "Expected JSON object with string 'from', string 'to', and "
                "list-of-int 'targets'."
            ) from exc


def ionq_annotation_handlers() -> dict[str, OpenQASM3Serializer]:
    """Standard ``annotation_handlers`` mapping for IonQ QASM 3 round-trips.

    Pass to :func:`qiskit.qasm3.loads` (and :func:`qiskit.qasm3.dumps` when
    emitting circuits that carry these annotations) so the IonQ-specific
    namespaces survive parsing rather than being rejected by the importer.
    """
    return {IONQ_ARRANGE_NAMESPACE: IonQArrangeSerializer()}


__all__ = [
    "IONQ_ARRANGE_NAMESPACE",
    "IonQArrangeAnnotation",
    "IonQArrangeSerializer",
    "ionq_annotation_handlers",
]
