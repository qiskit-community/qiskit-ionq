"""IonQ provider for Qiskit."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Literal

import qiskit
from ionq_core import ClientExtension, IonQClient
from ionq_core.api.backends import get_backends

from .backend import IonQBackend

try:
    __version__ = _pkg_version("qiskit-ionq")
except PackageNotFoundError:
    __version__ = "0.0.0"


class IonQProvider:
    def __init__(self, api_key: str | None = None, base_url: str = "https://api.ionq.co/v0.4"):
        self._client = IonQClient(
            api_key=api_key,
            base_url=base_url,
            extension=ClientExtension(
                user_agent_token=f"qiskit-ionq/{__version__}",
                default_headers={"X-Qiskit-Version": qiskit.__version__},
            ),
        )

    def backends(self, name: str | None = None, *, gateset: Literal["qis", "native"] = "qis") -> list[IonQBackend]:
        backend_list = get_backends.sync(client=self._client) or []
        results = [
            IonQBackend(provider=self, backend_info=b, client=self._client, gateset=gateset)
            for b in backend_list
            if name is None or b.backend == name
        ]
        return results

    def get_backend(self, name: str, *, gateset: Literal["qis", "native"] = "qis") -> IonQBackend:
        matches = self.backends(name=name, gateset=gateset)
        if not matches:
            raise ValueError(f"Backend {name!r} not found")
        return matches[0]
