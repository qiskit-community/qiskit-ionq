# -*- coding: utf-8 -*-
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

import logging
import os

from qiskit.providers import BaseProvider
from qiskit.providers.providerutils import filter_backends
from qiskit.providers.exceptions import QiskitBackendNotFoundError

from . import ionq_backend

logger = logging.getLogger(__name__)


def resolve_credentials(token: str = None, url: str = None):
    """Resolve credentials for use in IonQ Client API calls.

    If the provided ``token`` and ``url`` are both ``None``, then these values
    are loaded from the ``QISKIT_IONQ_API_TOKEN`` and ``QISKIT_IONQ_API_URL``
    environment variables, respectively.

    If no url is discovered, then ``https://api.ionq.co/v0.1`` is used.

    Args:
        token (str): IonQ API access token.
        url (str, optional): IonQ API url. Defaults to ``None``.

    Returns:
        dict[str]: A dict with "token" and "url" keys, for use by a client.
    """
    env_token = os.environ.get("QISKIT_IONQ_API_TOKEN")
    env_url = os.environ.get("QISKIT_IONQ_API_URL")
    return {
        "token": token or env_token,
        "url": url or env_url or "https://api.ionq.co/v0.1",
    }


class IonQProvider(BaseProvider):
    """Provider for interacting with IonQ backends

    Attributes:
        credentials(dict[str, str]): A dictionary containing ``token`` and
            ``url`` keys, whose values are an IonQ API Access Token and
            IonQ API URL, respectively.
    """

    name = "ionq_provider"

    def __init__(self, token: str = None, url: str = None):
        super().__init__()
        self.credentials = resolve_credentials(token, url)
        self._backends = [
            ionq_backend.IonQSimulatorBackend(self),
            ionq_backend.IonQQPUBackend(self),
        ]

    def backends(self, name: str = None, **kwargs):
        """
        Return a list of available backends, filtered by provided options.

        Args:
            name (str, optional): Name of the backend. Defaults to None.
            **kwargs: dict of criteria.

        Returns:
            list[IonQBackend]: A filtered list of IonQ Provider Backends.
        """
        backends = self._backends
        if name:
            backends = [b for b in self._backends if b.name() == name]
        return filter_backends(backends, **kwargs)

    def get_backend(self, name=None, **kwargs):
        """Return a single backend matching the specified filtering.
        Args:
            name (str): name of the backend.
            **kwargs: dict used for filtering.
        Returns:
            Backend: a backend matching the filtering.
        Raises:
            QiskitBackendNotFoundError: if no backend could be found or
                more than one backend matches the filtering criteria.
        """
        backends = self.backends(name, **kwargs)
        if len(backends) > 1:
            raise QiskitBackendNotFoundError('More than one backend matches criteria.')
        if not backends:
            raise QiskitBackendNotFoundError('No backend matches criteria.')

        return backends[0]
