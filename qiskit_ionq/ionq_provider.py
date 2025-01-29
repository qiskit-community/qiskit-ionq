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

"""Provider for interacting with IonQ backends"""

from __future__ import annotations

import logging

from typing import Callable, Literal, Optional

from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.providers.providerutils import filter_backends
from .helpers import resolve_credentials

from . import ionq_backend


logger = logging.getLogger(__name__)


class IonQProvider:
    """Provider for interacting with IonQ backends

    Attributes:
        credentials(dict[str, str]): A dictionary containing ``token`` and
            ``url`` keys, whose values are an IonQ API Access Token and
            IonQ API URL, respectively.
    """

    name = "ionq_provider"

    def __init__(
        self,
        token: Optional[str] = None,
        url: Optional[str] = None,
        custom_headers: Optional[dict] = None,
    ):
        super().__init__()
        self.custom_headers = custom_headers
        self.credentials = resolve_credentials(token, url)
        self.backends = BackendService(
            [
                ionq_backend.IonQSimulatorBackend(self),
                ionq_backend.IonQQPUBackend(self),
            ]
        )

    def get_backend(
        self,
        name: str,
        gateset: Literal["qis", "native"] = "qis",
        **kwargs,
    ) -> ionq_backend.Backend:
        """Return a single backend matching the specified filtering.
        Args:
            name (str): name of the backend.
            gateset (str): language used (QIS or native), defaults to QIS.
            **kwargs: dict used for filtering.
        Returns:
            Backend: a backend matching the filtering.
        Raises:
            QiskitBackendNotFoundError: if no backend could be found or
                more than one backend matches the filtering criteria.
        """
        name = "ionq_" + name if not name.startswith("ionq_") else name
        backends = self.backends(name, **kwargs)
        if len(backends) > 1:
            raise QiskitBackendNotFoundError("More than one backend matches criteria.")
        if not backends:
            raise QiskitBackendNotFoundError("No backend matches criteria.")

        return backends[0].with_name(name, gateset=gateset)


class BackendService:
    """A service class that allows for autocompletion
    of backends from provider.
    """

    def __init__(self, backends: list[ionq_backend.Backend]):
        """Initialize service

        Parameters:
            backends (list): List of backend instances.
        """
        self._backends = backends
        for backend in backends:
            setattr(self, backend.name(), backend)

    def __call__(
        self, name: Optional[str] = None, filters: Optional[Callable] = None, **kwargs
    ) -> list[ionq_backend.Backend]:
        """A listing of all backends from this provider.

        Parameters:
            name (str): The name of a given backend.
            filters (callable): A filter function.
            kwargs (dict): A dictionary of other keyword arguments.

        Returns:
            list: A list of backends, if any.

        Example:

        ..jupyter-execute::

            from qiskit_ionq import IonQProvider
            ionq = IonQProvider('TOKEN')
            sim = ionq.backends(filters=lambda x: x.configuration().simulator)
            print(sim)

        """
        # pylint: disable=arguments-differ
        backends = self._backends
        if name:
            backends = [b for b in self._backends if name.startswith(b.name())]
        return filter_backends(backends, filters, **kwargs)
