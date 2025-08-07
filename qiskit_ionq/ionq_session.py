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
Session object to both create and end sessions as well as
use in a context manager to for automatic lifecycle management.
"""

from __future__ import annotations

from .ionq_backend import IonQBackend


class Session:
    """Session object to manage IonQ sessions."""

    def __init__(
        self,
        backend: IonQBackend,
        *,
        max_time: int | str | None = None,
        max_cost: int | str | None = None,
        max_jobs: int | None = None,
        create_new: bool = True,
        session_id: str | None = None,
    ):
        self._backend = backend
        self._client = backend.client
        self._orig_run = None

        # Re-connect or create
        # TODO check if there's an open session for this backend
        # < 2 active
        # and the benefit of using a session
        if create_new and not session_id:
            self._create_session(max_time, max_cost, max_jobs)
        elif session_id:
            self._session_id = session_id
            # lazily verify existence
            self.details()  # will raise if unknown
        else:
            raise ValueError("Either create_new must be True or a session_id supplied.")

    @property
    def session_id(self) -> str:  # noqa: D401
        """Return the IonQ session UUID."""
        return self._session_id

    def details(self) -> dict:
        """Return JSON for this session."""
        return self._client.get_with_retry(
            self._client.make_path("sessions", self._session_id),
            headers=self._client.api_headers,
        ).json()

    def status(self) -> str | None:
        """Return the status of this session."""
        return self.details().get("state")

    def usage(self) -> float | None:
        """Return the usage time of this session in seconds."""
        return self.details().get("usage_time")

    def cancel(self) -> None:
        """Cancel all queued jobs inside this session."""
        jobs = (
            self._client.get_with_retry(
                self._client.make_path("jobs"),
                headers=self._client.api_headers,
                params={"session_id": self._session_id, "status": "queued"},
            )
            .json()
            .get("jobs", [])
        )
        self._client.cancel_jobs([j["id"] for j in jobs])

    def close(self) -> None:
        """POST /sessions/<id>/end   (idempotent)."""
        self._client.post("sessions", self._session_id, "end")

    def __enter__(self):
        # inject the session_id into any backend.run call
        assert self._orig_run is None  # only set once per context
        self._orig_run = self._backend.run

        def _run_with_session(*args, **kwargs):
            kwargs.setdefault("session_id", self._session_id)
            return self._orig_run(*args, **kwargs)

        self._backend.run = _run_with_session  # monkey‑patch for life of the context
        return self

    def __exit__(self, exc_type, *_):
        # On exception try to cancel queued jobs before closing the session.
        if exc_type is not None:
            self.cancel()
        self.close()

        # restore the backend.run we overwrote in __enter__
        if self._orig_run is not None:  # defensive
            self._backend.run = self._orig_run
            self._orig_run = None

        # propagate exceptions
        return False

    def _create_session(
        self,
        max_time: int | str | None,
        max_cost: int | str | None,
        max_jobs: int | None,
    ) -> None:
        """Create a new session."""
        payload = {
            "backend": self._backend.name().replace("ionq_qpu", "qpu"),
            "settings": {},
        }
        if max_jobs is not None:
            payload["settings"]["job_count_limit"] = max_jobs
        if max_time is not None:
            payload["settings"]["duration_limit_min"] = max_time
        if max_cost is not None:
            payload["settings"]["cost_limit"] = {"unit": "usd", "value": max_cost}

        resp = self._client.post("sessions", json_body=payload)
        self._session_id = resp["id"]

    # class-method shortcut
    @classmethod
    def from_id(cls, session_id: str, *, backend: IonQBackend) -> Session:
        """Create a Session object from an existing session ID."""
        return cls(backend=backend, session_id=session_id, create_new=False)

    def run(self, *args, **kwargs):
        """Run a job using the session."""
        return self._backend.run(*args, **kwargs, session_id=self._session_id)


__all__ = ["Session"]
