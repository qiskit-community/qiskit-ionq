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

from typing import Literal, Sequence, TYPE_CHECKING
import warnings

from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2 as Backend, Options
from qiskit.transpiler import Target, CouplingMap

from . import ionq_equivalence_library, ionq_job, ionq_client, exceptions
from .helpers import GATESET_MAP, get_n_qubits
from .ionq_client import Characterization

if TYPE_CHECKING:  # pragma: no cover
    from .ionq_provider import IonQProvider


class IonQBackend(Backend):
    """Common functionality for all IonQ backends (simulator and QPU)."""

    _client: ionq_client.IonQClient | None = None

    def __init__(
        self,
        *,
        provider: IonQProvider,
        name: str,
        description: str,
        gateset: Literal["qis", "native"],
        num_qubits: int,
        simulator: bool,
        backend_version: str = "0.0.1",
        max_shots: int | None = None,
        max_experiments: int | None = None,
        **option_overrides,
    ):
        """Build a new IonQ backend instance."""
        # Register IonQ-specific gate equivalences once per process.
        ionq_equivalence_library.add_equivalences()

        # Feed mandatory metadata to BackendV2.
        super().__init__(
            provider=provider,
            name=name,
            description=description,
            backend_version=backend_version,
            **option_overrides,  # these must exist in _default_options()
        )

        # Immutable device facts
        self._gateset: Literal["qis", "native"] = gateset
        # Always advertise measure/reset so the transpiler can accept user circuits.
        self._basis_gates: Sequence[str] = tuple(GATESET_MAP[gateset])
        self._num_qubits: int = num_qubits
        self._simulator: bool = simulator
        self._max_experiments: int | None = max_experiments
        self._max_shots: int | None = max_shots

        # A bare-bones transpiler Target.
        target = Target(num_qubits=self._num_qubits, dt=None)
        self._target: Target = target

    @classmethod
    def _default_options(cls) -> Options:
        """Dynamic (user-tuneable) backend options."""
        return Options(
            shots=1024,
            job_settings=None,
            error_mitigation=None,
            extra_query_params={},
            extra_metadata={},
            sampler_seed=None,  # simulator-only; harmless default for QPU
            noise_model="ideal",  # simulator-only
        )

    @property
    def target(self) -> Target | None:
        return self._target

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def max_circuits(self) -> int | None:
        return self._max_experiments

    @property
    def basis_gates(self) -> Sequence[str]:
        """Return the basis gates for this backend."""
        return self._basis_gates

    def gateset(self) -> Literal["qis", "native"]:
        """Return the active gate-set (``"qis"`` or ``"native"``)."""
        return self._gateset

    @property
    def client(self) -> ionq_client.IonQClient:
        """Return the IonQ client for this backend."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> ionq_client.IonQClient:
        creds = self._provider.credentials

        if "token" not in creds:
            raise exceptions.IonQCredentialsError(
                "Credentials `token` not present in provider."
            )
        token = creds["token"]
        if token is None:
            raise exceptions.IonQCredentialsError(
                "Credentials `token` may not be None!"
            )

        if "url" not in creds:
            raise exceptions.IonQCredentialsError(
                "Credentials `url` not present in provider."
            )
        url = creds["url"]
        if url is None:
            raise exceptions.IonQCredentialsError("Credentials `url` may not be None!")
        return ionq_client.IonQClient(token, url, self._provider.custom_headers)

    def run(self, run_input: QuantumCircuit | Sequence[QuantumCircuit], **options):
        """Create and run a job on an IonQ Backend.

        Args:
            run_input (:class:`QuantumCircuit <qiskit.circuit.QuantumCircuit>`):
                A Qiskit QuantumCircuit object.

        Returns:
            IonQJob: A reference to the job that was submitted.
        """
        # Mid-circuit measurement sanity check.
        if not all(
            (
                self._has_measurements(circ)
                for circ in (run_input if isinstance(run_input, list) else [run_input])
            )
        ):
            warnings.warn(
                "Circuit is not measuring any qubits",
                UserWarning,
                stacklevel=2,
            )

        # Merge default & user-supplied options
        run_opts = {**self.options.__dict__, **options}

        job = ionq_job.IonQJob(
            backend=self,
            job_id=None,
            client=self.client,
            circuit=run_input,
            passed_args=run_opts,
        )
        job.submit()
        return job

    def retrieve_job(self, job_id: str):
        """Retrieve a job by its ID."""
        return ionq_job.IonQJob(self, job_id, self.client)

    def retrieve_jobs(self, job_ids: Sequence[str]):
        """Retrieve multiple jobs by their IDs."""
        return [ionq_job.IonQJob(self, jid, self.client) for jid in job_ids]

    def calibration(self) -> Characterization | None:
        """Return the characterization data for this backend."""
        if self._simulator:
            return None
        name_for_api = self.name.replace("ionq_qpu", "qpu")
        return self.client.get_calibration_data(name_for_api, limit=1)

    def status(self) -> bool:
        """Return True if the backend is available, False otherwise."""
        cal = self.calibration()
        return bool(cal and getattr(cal, "status", "available") == "available")

    @property
    def coupling_map(self) -> CouplingMap:
        """IonQ hardware is fully connected."""
        return CouplingMap.from_full(self._num_qubits)

    @staticmethod
    def _has_measurements(circ: QuantumCircuit) -> bool:
        return any(inst.operation.name == "measure" for inst in circ.data)

    def __eq__(self, other):
        if not isinstance(other, IonQBackend):
            return NotImplemented
        return (self.name, self._gateset) == (other.name, other._gateset)

    def __hash__(self):
        return hash((self.name, self._gateset))


class IonQSimulatorBackend(IonQBackend):
    """IonQ state-vector simulator."""

    def __init__(
        self,
        provider: IonQProvider,
        name: str = "simulator",
        gateset: Literal["qis", "native"] = "qis",
    ):
        backend_name = name if name.startswith("ionq_") else f"ionq_{name}"
        num_qubits = get_n_qubits(name)
        super().__init__(
            provider=provider,
            name=backend_name,
            description="IonQ cloud simulator",
            gateset=gateset,
            num_qubits=num_qubits,
            simulator=True,
            max_shots=1,
            max_experiments=None,
        )

    def with_name(self, name: str, **kwargs):
        """Helper method that returns this backend with a more specific target system."""
        return IonQSimulatorBackend(self._provider, name, **kwargs)


class IonQQPUBackend(IonQBackend):
    """IonQ trapped-ion hardware back-ends."""

    def __init__(
        self,
        provider: IonQProvider,
        name: str = "ionq_qpu",
        gateset: Literal["qis", "native"] = "qis",
    ):
        num_qubits = get_n_qubits(name)
        super().__init__(
            provider=provider,
            name=name,
            description="IonQ trapped-ion QPU",
            gateset=gateset,
            num_qubits=num_qubits,
            simulator=False,
            max_shots=10_000,
            max_experiments=None,
        )

    def with_name(self, name: str, **kwargs) -> IonQQPUBackend:
        """Helper method that returns this backend with a more specific target system."""
        return IonQQPUBackend(self._provider, name, **kwargs)


__all__ = ["IonQBackend", "IonQSimulatorBackend", "IonQQPUBackend"]
