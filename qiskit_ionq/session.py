"""IonQ session for Qiskit, backed by ionq-core-python's SessionManager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ionq_core import SessionManager

from .sampler import IonQSampler

if TYPE_CHECKING:
    from .backend import IonQBackend


class IonQSession:
    def __init__(
        self,
        backend: IonQBackend,
        *,
        max_jobs: int | None = None,
        max_time: int | None = None,
        max_cost: float | None = None,
    ):
        self._backend = backend
        self._manager = SessionManager(
            backend._client, backend.name, max_jobs=max_jobs, max_time=max_time, max_cost=max_cost
        )

    @classmethod
    def from_id(cls, backend: IonQBackend, session_id: str) -> IonQSession:
        inst = cls.__new__(cls)
        inst._backend = backend
        inst._manager = SessionManager.from_id(backend._client, session_id)
        return inst

    @property
    def session_id(self) -> str | None:
        return self._manager.session_id

    def status(self) -> str:
        return self._manager.status()

    def close(self) -> None:
        self._manager.close()

    def __enter__(self):
        self._manager.open()
        return self

    def __exit__(self, *exc):
        self._manager.close()

    def sampler(self, **kwargs) -> IonQSampler:
        return IonQSampler(self._backend, session=self, **kwargs)
