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

"""IonQ provider backends."""

from qiskit.providers import BaseBackend
from qiskit.providers.models import BackendConfiguration

from . import exceptions, ionq_client, ionq_job


class IonQBackend(BaseBackend):
    """IonQ Backend base class."""

    _client = None

    @property
    def client(self):
        """A lazily populated IonQ API Client.

        Returns:
            IonQClient: An instance of a REST API client
        """
        if self._client is None:
            self._client = self.create_client()
        return self._client

    def create_client(self):
        """Create an IonQ REST API Client using provider credentials.

        Raises:
            IonQCredentialsError: If the provider's
                :attr:`credentials <IonQProvider.credentials>` does not have
                a ``"token"`` or ``"url"`` key, or if their values are ``None``.

        Returns:
            IonQClient: An instance of a REST API client.
        """
        credentials = self._provider.credentials

        try:
            token = credentials["token"]
        except KeyError as ex:
            raise exceptions.IonQCredentialsError(
                "Credentials `token` not present in provider."
            ) from ex

        if token is None:
            raise exceptions.IonQCredentialsError(
                "Credentials `token` may not be None!"
            )

        try:
            url = credentials["url"]
        except KeyError as ex:
            raise exceptions.IonQCredentialsError(
                "Credentials `url` not present in provider."
            ) from ex

        if url is None:
            raise exceptions.IonQCredentialsError(
                "Credentials `url` may not be None!"
            )

        return ionq_client.IonQClient(token, url)

    def run(self, qobj):
        """Create and run a job on an IonQ Backend.

        Args:
            qobj (:mod:`qobj <qiskit.qobj>`): A Qiskit Quantum Job object.

        Returns:
            IonQJob: A reference to the job that was submitted.
        """
        job = ionq_job.IonQJob(self, None, self.client, qobj=qobj)
        job.submit()
        return job

    def retrieve_job(self, job_id):
        """get a job from a specific backend, by job id."""
        return ionq_job.IonQJob(self, job_id, self.client)

    def retrieve_jobs(self, job_ids):
        """get a list of jobs from a specific backend, job id """

        return [
            ionq_job.IonQJob(self, job_id, self.client) for job_id in job_ids
        ]

    # TODO: Implement backend status checks.
    def status(self):
        """Not yet implemented.

        Raises:
            NotImplementedError: This behavior is not currently supported.
        """
        raise NotImplementedError("Backend status check is not supported.")


class IonQSimulatorBackend(IonQBackend):
    """IonQ Backend for running simulated jobs."""

    def __init__(self, provider):
        """Base class for interfacing with an IonQ backend"""
        config = BackendConfiguration.from_dict(
            {
                "backend_name": "ionq_simulator",
                "backend_version": "0.0.1",
                "simulator": True,
                "local": False,
                "coupling_map": None,
                "description": "IonQ simulator",
                "basis_gates": [
                    "x",
                    "y",
                    "z",
                    "rx",
                    "ry",
                    "rz",
                    "h",
                    "not",
                    "cnot",
                    "cx",
                    "s",
                    "si",
                    "t",
                    "ti",
                    "v",
                    "vi",
                    "xx",
                    "yy",
                    "zz",
                    "swap",
                ],
                "memory": False,
                "n_qubits": 29,
                "conditional": False,
                "max_shots": 10000,
                "max_experiments": 1,
                "open_pulse": False,
                "gates": [
                    {"name": "TODO", "parameters": [], "qasm_def": "TODO"}
                ],
            }
        )
        super().__init__(configuration=config, provider=provider)


class IonQQPUBackend(IonQBackend):
    """IonQ Backend for running qpu-based jobs."""

    def __init__(self, provider):
        config = BackendConfiguration.from_dict(
            {
                "backend_name": "ionq_qpu",
                "backend_version": "0.0.1",
                "simulator": False,
                "local": False,
                "coupling_map": None,
                "description": "IonQ QPU",
                "basis_gates": [
                    "x",
                    "y",
                    "z",
                    "rx",
                    "ry",
                    "rz",
                    "h",
                    "not",
                    "cnot",
                    "cx",
                    "s",
                    "si",
                    "t",
                    "ti",
                    "v",
                    "vi",
                    "xx",
                    "yy",
                    "zz",
                    "swap",
                ],
                "memory": False,
                "n_qubits": 11,
                "conditional": False,
                "max_shots": 10000,
                "max_experiments": 1,
                "open_pulse": False,
                "gates": [
                    {"name": "TODO", "parameters": [], "qasm_def": "TODO"}
                ],
            }
        )
        super().__init__(configuration=config, provider=provider)


__all__ = ["IonQBackend", "IonQQPUBackend", "IonQSimulatorBackend"]
