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

import abc
from datetime import datetime
import warnings

from qiskit.providers import BackendV1 as Backend
from qiskit.providers.models import BackendConfiguration
from qiskit.providers.models.backendstatus import BackendStatus
from qiskit.providers import Options

from . import exceptions, ionq_client, ionq_job
from .helpers import GATESET_MAP


class Calibration:
    """
    IonQ backend calibration data.

    This class is a simple wrapper for IonQ hardware calibration data.
    """

    def __init__(self, data):
        self._data = data

    @property
    def uuid(self):
        """The ID of the calibration.

        Returns:
           str: The ID.
        """
        return self._data["id"]

    @property
    def num_qubits(self):
        """The number of qubits available.

        Returns:
            int: A number of qubits.
        """
        return int(self._data["qubits"])

    @property
    def target(self):
        """The target calibrated hardware.

        Returns:
            str: The name of the target hardware backend.
        """
        return self._data["backend"]

    @property
    def calibration_time(self):
        """Time of the measurement, in UTC.

        Returns:
            datetime.datetime: A datetime object with the time.
        """
        return datetime.fromtimestamp(self._data["date"])

    @property
    def fidelities(self):
        """Fidelity for single-qubit (1q) and two-qubit (2q) gates, and State
        Preparation and Measurement (spam) operations.

        Currently provides only mean fidelity; additional statistical data will
        be added in the future.

        Returns:
            dict: A dict containing fidelity data for 1a, 2q, and spam.
        """
        return self._data["fidelity"]

    @property
    def timings(self):
        """Various system property timings. All times expressed as seconds.

        Timings currently include::

            * ``t1``
            * ``t2``
            * ``1q``
            * ``2q``
            * ``readout``
            * ``reset``

        Returns:
            dict: A dictionary of timings.
        """
        return self._data["timing"]

    @property
    def connectivity(self):
        """Returns connectivity data.

        Returns:
            list[tuple[int, int]]: An array of valid, unordered tuples of
                possible qubits for executing two-qubit gates
        """
        return self._data["connectivity"]


class IonQBackend(Backend):
    """IonQ Backend base class."""

    _client = None

    @classmethod
    def _default_options(cls):
        return Options(
            shots=1024,
            job_settings=None,
            error_mitigation=None,
            extra_query_params={},
            extra_metadata={},
        )

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
            raise exceptions.IonQCredentialsError("Credentials `url` may not be None!")

        return ionq_client.IonQClient(token, url, self._provider.custom_headers)

    # pylint: disable=missing-type-doc,missing-param-doc,arguments-differ,arguments-renamed
    def run(self, circuit, **kwargs):
        """Create and run a job on an IonQ Backend.

        .. NOTE::

            IonQ backends do not support multi-experiment jobs.
            If ``circuit`` is provided as a list with more than one element
            then this method will raise out with a RuntimeError.

        Args:
            circuit (:class:`QuantumCircuit <qiskit.circuit.QuantumCircuit>`):
                A Qiskit QuantumCircuit object.

        Returns:
            IonQJob: A reference to the job that was submitted.

        Raises:
            RuntimeError: If a multi-experiment circuit was provided.
        """
        if isinstance(circuit, (list, tuple)):
            if len(circuit) > 1:
                raise RuntimeError("Multi-experiment jobs are not supported!")
            circuit = circuit[0]

        if not self.has_valid_mapping(circuit):
            warnings.warn(
                f"Circuit {circuit.name} is not measuring any qubits",
                UserWarning,
                stacklevel=2,
            )

        for kwarg in kwargs:
            if not hasattr(self.options, kwarg):
                warnings.warn(
                    f"Option {kwarg} is not used by this backend",
                    UserWarning,
                    stacklevel=2,
                )
        if "shots" not in kwargs:
            kwargs["shots"] = self.options.shots
        # TODO: Should we merge the two maps, or warn if both are set?
        if "job_settings" not in kwargs:
            kwargs["job_settings"] = self.options.job_settings
        elif self.options.job_settings is not None:
            warnings.warn(
                (
                    "Option job_settings is set on the backend, and on the request. "
                    "Ignoring the backend specified option."
                ),
                UserWarning,
                stacklevel=2,
            )
        passed_args = kwargs

        job = ionq_job.IonQJob(
            self,
            None,
            self.client,
            circuit=circuit,
            passed_args=passed_args,
        )
        job.submit()
        return job

    def retrieve_job(self, job_id):
        """get a job from a specific backend, by job id."""
        return ionq_job.IonQJob(self, job_id, self.client)

    def retrieve_jobs(self, job_ids):
        """get a list of jobs from a specific backend, job id"""

        return [ionq_job.IonQJob(self, job_id, self.client) for job_id in job_ids]

    def has_valid_mapping(self, circuit) -> bool:
        """checks if the circuit has at least one
        valid qubit -> bit measurement.

        Args:
            circuit (:class:`QuantumCircuit <qiskit.circuit.QuantumCircuit>`):
                A Qiskit QuantumCircuit object.

        Returns:
            boolean: if the circuit has valid mappings
        """
        # Check if a qubit is measured
        for instruction, _, cargs in circuit.data:
            if instruction.name == "measure" and len(cargs):
                return True
        # If no mappings are found, return False
        return False

    # TODO: Implement backend status checks.
    def status(self):
        """Return a backend status object to the caller.

        Returns:
            BackendStatus: the status of the backend.
        """
        return BackendStatus(
            backend_name=self.name(),
            backend_version="1",
            operational=True,
            pending_jobs=0,
            status_msg="",
        )

    def calibration(self):
        """Fetch the most recent calibration data for this backend.

        Returns:
            Calibration: A calibration data wrapper.
        """
        backend_name = self.name().replace("_", ".")
        calibration_data = self.client.get_calibration_data(backend_name)
        if calibration_data is None:
            return None
        return Calibration(calibration_data)

    @abc.abstractmethod
    def with_name(self, name, **kwargs):
        """Helper method that returns this backend with a more specific target system."""
        pass

    @abc.abstractmethod
    def gateset(self):
        """Helper method returning the gateset this backend is targeting."""
        pass

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name() == other.name() and self.gateset() == other.gateset()
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


class IonQSimulatorBackend(IonQBackend):
    """
    IonQ Backend for running simulated jobs.


    .. ATTENTION::

        When noise_model ideal is specified, the maximum shot-count for a state vector sim is
        always ``1``.

    .. ATTENTION::

        When noise_model ideal is specified, calling
        :meth:`get_counts <qiskit_ionq.ionq_job.IonQJob.get_counts>`
        on a job processed by this backend will return counts expressed as
        probabilites, rather than a multiple of shots.
    """

    @classmethod
    def _default_options(cls):
        return Options(
            shots=1024,
            job_settings=None,
            sampler_seed=None,
            noise_model="ideal",
            extra_query_params={},
            extra_metadata={},
        )

    # pylint: disable=missing-type-doc,missing-param-doc,arguments-differ,useless-super-delegation
    def run(self, circuit, **kwargs):
        """Create and run a job on IonQ's Simulator Backend.

        .. WARNING:

            The maximum shot-count for a state vector sim is always ``1``.
            As a result, the ``shots`` keyword argument in this method is ignored.

        Args:
            circuit (:class:`QuantumCircuit <qiskit.circuit.QuantumCircuit>`):
                A Qiskit QuantumCircuit object.

        Returns:
            IonQJob: A reference to the job that was submitted.
        """
        return super().run(circuit, **kwargs)

    def calibration(self):
        """Simulators have no calibration data.

        Returns:
            NoneType: None
        """
        return None

    def gateset(self):
        return self._gateset

    def __init__(self, provider, name="ionq_simulator", gateset="qis"):
        """Base class for interfacing with an IonQ backend"""
        self._gateset = gateset
        config = BackendConfiguration.from_dict(
            {
                "backend_name": name,
                "backend_version": "0.0.1",
                "simulator": True,
                "local": False,
                "coupling_map": None,
                "description": "IonQ simulator",
                "basis_gates": GATESET_MAP[gateset],
                "memory": False,
                # Varied based on noise model, but enforced server-side.
                "n_qubits": 32,
                "conditional": False,
                "max_shots": 1,
                "max_experiments": 1,
                "open_pulse": False,
                "gates": [{"name": "TODO", "parameters": [], "qasm_def": "TODO"}],
            }
        )
        super().__init__(configuration=config, provider=provider)

    def with_name(self, name, **kwargs):
        """Helper method that returns this backend with a more specific target system."""
        return IonQSimulatorBackend(self._provider, name, **kwargs)


class IonQQPUBackend(IonQBackend):
    """IonQ Backend for running qpu-based jobs."""

    def gateset(self):
        return self._gateset

    def __init__(self, provider, name="ionq_qpu", gateset="qis"):
        self._gateset = gateset
        config = BackendConfiguration.from_dict(
            {
                "backend_name": name,
                "backend_version": "0.0.1",
                "simulator": False,
                "local": False,
                "coupling_map": None,
                "description": "IonQ QPU",
                "basis_gates": GATESET_MAP[gateset],
                "memory": False,
                # This is a generic backend for all IonQ hardware, the server will do more specific
                # qubit count checks. In the future, dynamic backend configuration from the server
                # will be used in place of these hard-coded caps.
                "n_qubits": 32,
                "conditional": False,
                "max_shots": 10000,
                "max_experiments": 1,
                "open_pulse": False,
                "gates": [{"name": "TODO", "parameters": [], "qasm_def": "TODO"}],
            }
        )
        super().__init__(configuration=config, provider=provider)

    def with_name(self, name, **kwargs):
        """Helper method that returns this backend with a more specific target system."""
        return IonQQPUBackend(self._provider, name, **kwargs)


__all__ = ["IonQBackend", "IonQQPUBackend", "IonQSimulatorBackend"]
