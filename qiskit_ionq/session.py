# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
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

"""IonQSession: ionq-core-backed session class.

Added in qiskit-ionq 1.1.0 as the migration target for the deprecated
:class:`qiskit_ionq.Session`. In 2.0 this class is the only Session
implementation and lives at the same path (qiskit_ionq/session.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ionq_core import IonQClient, SessionManager

if TYPE_CHECKING:
    from .ionq_backend import IonQBackend


class IonQSession:
    """Session lifecycle manager backed by ionq-core's SessionManager.

    In 1.1.0 this constructs its own ionq-core IonQClient from the legacy
    IonQProvider's resolved credentials. In 2.0 it will share the provider's
    ionq-core client directly.

    Example:
        >>> from qiskit_ionq import IonQProvider, IonQSession
        >>> backend = IonQProvider().get_backend("ionq_qpu.aria-1")
        >>> with IonQSession(backend, max_jobs=10) as sess:
        ...     job1 = backend.run(qc1, session_id=sess.session_id)
        ...     job2 = backend.run(qc2, session_id=sess.session_id)
    """

    def __init__(
        self,
        backend: IonQBackend,
        *,
        max_jobs: int | None = None,
        max_time: int | None = None,
        max_cost: float | None = None,
    ):
        creds = backend._provider.credentials
        self._backend = backend
        self._client = IonQClient(api_key=creds["token"], base_url=creds["url"])
        # Strip the local "ionq_" name prefix; the IonQ API uses bare names
        # like "qpu.aria-1" / "simulator".
        api_name = backend.name.removeprefix("ionq_")
        self._manager = SessionManager(
            self._client,
            api_name,
            max_jobs=max_jobs,
            max_time=max_time,
            max_cost=max_cost,
        )

    @classmethod
    def from_id(cls, backend: IonQBackend, session_id: str) -> IonQSession:
        """Re-attach to an existing IonQ session by UUID."""
        inst = cls.__new__(cls)
        creds = backend._provider.credentials
        inst._backend = backend
        inst._client = IonQClient(api_key=creds["token"], base_url=creds["url"])
        inst._manager = SessionManager.from_id(inst._client, session_id)
        return inst

    @property
    def session_id(self) -> str | None:
        """Return the IonQ session UUID."""
        return self._manager.session_id

    def status(self) -> str:
        """Return the current session status (`created`/`started`/`ended`)."""
        return self._manager.status()

    def close(self) -> None:
        """Close the session."""
        self._manager.close()

    def __enter__(self) -> IonQSession:
        self._manager.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self._manager.close()
