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

from __future__ import annotations

import abc
from collections import defaultdict
from datetime import datetime
from typing import Literal, TYPE_CHECKING
import warnings

from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.providers import BackendV2 as Backend, Options
from qiskit.transpiler import Target

from . import exceptions, ionq_client, ionq_job, ionq_equivalence_library
from .helpers import GATESET_MAP, get_n_qubits

if TYPE_CHECKING:
    from .ionq_provider import IonQProvider


class Calibration:
    """
    IonQ backend calibration data.

    This class is a simple wrapper for IonQ hardware calibration data.
    """

    def __init__(self, data):
        self._data = data

    @property
    def uuid(self) -> str:
        """The ID of the calibration.

        Returns:
            str: The ID.
        """
        return self._data["id"]

    @property
    def num_qubits(self) -> int:
        """The number of qubits available.

        Returns:
            int: A number of qubits.
        """
        return int(self._data["qubits"])

    @property
    def target(self) -> str:
        """The target calibrated hardware.

        Returns:
            str: The name of the target hardware backend.
        """
        return self._data["backend"]

    @property
    def calibration_time(self) -> datetime:
        """Time of the measurement, in UTC.

        Returns:
            datetime.datetime: A datetime object with the time.
        """
        return datetime.fromtimestamp(self._data["date"])

    @property
    def fidelities(self) -> dict:
        """Fidelity for single-qubit (1q) and two-qubit (2q) gates, and State
        Preparation and Measurement (spam) operations.

        Currently provides only mean fidelity; additional statistical data will
        be added in the future.

        Returns:
            dict: A dict containing fidelity data for 1a, 2q, and spam.
        """
        return self._data["fidelity"]

    @property
    def timings(self) -> dict:
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
    def connectivity(self) -> list[tuple[int, int]]:
        """Returns connectivity data.

        Returns:
            list[tuple[int, int]]: An array of valid, unordered tuples of
                possible qubits for executing two-qubit gates
        """
        return self._data["connectivity"]


class IonQBackend(Backend):
    """IonQ Backend base class."""

    _client = None

    @property
    def target(self) -> Target:
        """Instruction-set description for the transpiler."""
        return self._target

    @property
    def num_qubits(self) -> int:
        return self._target.num_qubits

    @property
    def max_circuits(self) -> int:
        return self._max_circuits

    def __init__(
        self,
        *args,
        gateset: Literal["qis", "native"],
        simulator: bool,
        **kwargs,
    ) -> None:
        # Add IonQ equivalences.
        ionq_equivalence_library.add_equivalences()
        name = kwargs.get("name")
        self._target = Target(num_qubits=get_n_qubits(name))

        # Gate arity map -- all unknowns assumed single-qubit.
        _arity: dict[str, int] = defaultdict(lambda: 1, {"ms": 2, "zz": 2})

        for gate_name in GATESET_MAP[gateset]:
            instr = Instruction(name=gate_name, num_qubits=_arity[gate_name], num_clbits=0, params=[])
            self._target.add_instruction(instr, properties={None: None})

        self.gateset = gateset
        super().__init__(*args, **kwargs)

    @classmethod
    def _default_options(cls) -> Options:
        return Options(
            shots=1024,
            job_settings=None,
            error_mitigation=None,
            extra_query_params={},
            extra_metadata={},
        )

    @property
    def client(self) -> ionq_client.IonQClient:
        """A lazily populated IonQ API Client.

        Returns:
            IonQClient: An instance of a REST API client
        """
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> ionq_client.IonQClient:
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
    def run(self, circuit: QuantumCircuit, **kwargs) -> ionq_job.IonQJob:
        """Create and run a job on an IonQ Backend.

        Args:
            circuit (:class:`QuantumCircuit <qiskit.circuit.QuantumCircuit>`):
                A Qiskit QuantumCircuit object.

        Returns:
            IonQJob: A reference to the job that was submitted.
        """
        if not all(
            (
                self.has_valid_mapping(circ)
                for circ in (circuit if isinstance(circuit, list) else [circuit])
            )
        ):
            warnings.warn(
                "Circuit is not measuring any qubits",
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

    def retrieve_job(self, job_id: str) -> ionq_job.IonQJob:
        """get a job from a specific backend, by job id."""
        return ionq_job.IonQJob(self, job_id, self.client)

    def retrieve_jobs(self, job_ids: list[str]) -> list[ionq_job.IonQJob]:
        """get a list of jobs from a specific backend, job id"""
        return [ionq_job.IonQJob(self, job_id, self.client) for job_id in job_ids]

    def cancel_job(self, job_id: str) -> dict:
        """cancels a job from a specific backend, by job id."""
        return self.client.cancel_job(job_id)

    def cancel_jobs(self, job_ids: list[str]) -> list[dict]:
        """cancels a list of jobs from a specific backend, job id"""
        return [self.client.cancel_job(job_id) for job_id in job_ids]

    def status(self) -> dict:  # pylint: disable=arguments-differ
        """Return a backend status object to the caller.

        Returns:
            BackendStatus: the status of the backend.
        """
        return {
            "backend_name": self.name,
            "backend_version": "1",
            "operational": True,
            "pending_jobs": 0,
            "status_msg": "",
        }

    def calibration(self) -> Calibration | None:
        """Fetch the most recent calibration data for this backend.

        Returns:
            Calibration: A calibration data wrapper.
        """
        backend_name = self.name.replace("_", ".")
        calibration_data = self.client.get_calibration_data(backend_name)
        if calibration_data is None:
            return None
        return Calibration(calibration_data)

    @abc.abstractmethod
    def with_name(self, name, **kwargs) -> "IonQBackend":
        """Helper method that returns this backend with a more specific target system."""
        pass

    def has_valid_mapping(self, circuit: QuantumCircuit) -> bool:
        """checks if the circuit has at least one valid qubit -> bit measurement."""
        for instruction, _, cargs in circuit.data:
            if instruction.name == "measure" and len(cargs):
                return True
        return False

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.name == other.name and self.gateset == other.gateset
        return NotImplemented

    def __ne__(self, other) -> bool:
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
    def _default_options(cls) -> Options:
        return Options(
            shots=1024,
            job_settings=None,
            sampler_seed=None,
            noise_model="ideal",
            extra_query_params={},
            extra_metadata={},
        )

    def __init__(
        self,
        provider,
        name: str = "simulator",
        gateset: Literal["qis", "native"] = "qis",
    ):
        """Base class for interfacing with an IonQ backend"""
        full_name = "ionq_" + name if not name.startswith("ionq_") else name

        super().__init__(
            provider=provider,
            name=full_name,
            description="IonQ simulator",
            backend_version="0.0.1",
            gateset=gateset,
            simulator=True,
        )

    # pylint: disable=missing-type-doc,missing-param-doc,arguments-differ,useless-super-delegation
    def run(self, circuit: QuantumCircuit, **kwargs) -> ionq_job.IonQJob:
        """
        Create and run a job on IonQ's Simulator Backend.

        .. WARNING:

            The maximum shot-count for a state vector sim is always ``1``.
            As a result, the ``shots`` keyword argument in this method is ignored.

        Args:
            circuit (:class:`QuantumCircuit <qiskit.circuit.QuantumCircuit>`):
                A Qiskit QuantumCircuit object.

        Returns:
            IonQJob: A reference to the job that was submitted.
        """
        kwargs.pop("shots", None)
        return super().run(circuit, **kwargs)

    def calibration(self) -> None:
        """Simulators have no calibration data."""
        return None

    def with_name(self, name, **kwargs) -> "IonQSimulatorBackend":
        """Helper method that returns this backend with a more specific target system."""
        return IonQSimulatorBackend(self._provider, name, **kwargs)


class IonQQPUBackend(IonQBackend):
    """IonQ Backend for running qpu-based jobs."""

    def __init__(
        self,
        provider: IonQProvider,
        name: str = "ionq_qpu",
        gateset: Literal["qis", "native"] = "qis",
    ):
        super().__init__(
            provider=provider,
            name=name,
            description="IonQ QPU",
            backend_version="0.0.1",
            gateset=gateset,
            simulator=False,
        )

    def with_name(self, name: str, **kwargs) -> "IonQQPUBackend":
        """Helper method that returns this backend with a more specific target system."""
        return IonQQPUBackend(self._provider, name, **kwargs)


__all__ = ["IonQBackend", "IonQQPUBackend", "IonQSimulatorBackend"]
