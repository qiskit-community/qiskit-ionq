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

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import (
    Measure,
    Reset,
    CHGate,
    XGate,
    CPhaseGate,
    CRXGate,
    CRYGate,
    CRZGate,
    CSXGate,
    CXGate,
    CYGate,
    CZGate,
    HGate,
    IGate,
    MCPhaseGate,
    MCXGate,
    PhaseGate,
    RXGate,
    RXXGate,
    RYGate,
    RYYGate,
    RZGate,
    RZZGate,
    SGate,
    SdgGate,
    SwapGate,
    SXGate,
    SXdgGate,
    TGate,
    TdgGate,
    YGate,
    ZGate,
    PauliEvolutionGate,
)
from qiskit.providers import BackendV2 as Backend, Options
from qiskit.transpiler import Target, CouplingMap

from qiskit_ionq.ionq_gates import GPIGate, GPI2Gate, MSGate, ZZGate
from . import ionq_equivalence_library, ionq_job, ionq_client, exceptions
from .helpers import GATESET_MAP, get_n_qubits, warn_bad_transpile_level
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
        **initial_options,
    ):
        """Build a new IonQ backend instance."""
        # Register IonQ-specific gate equivalences once per process.
        ionq_equivalence_library.add_equivalences()

        super().__init__(
            provider=provider,
            name=name,
            description=description,
            backend_version=backend_version,
        )

        # Immutable facts
        self._gateset: Literal["qis", "native"] = gateset
        self._basis_gates: Sequence[str] = tuple(GATESET_MAP[gateset])
        self._num_qubits: int = num_qubits
        self._simulator: bool = simulator
        self._max_experiments: int | None = max_experiments
        self._max_shots: int | None = max_shots

        # Target (basis & connectivity)
        self._target = self._make_target()

        # Apply initial options if any
        if initial_options:
            self.options.update_options(**initial_options)

        # Warn if optimization_level is set to a bad value (for IonQ)
        warn_bad_transpile_level()

    @classmethod
    def _default_options(cls) -> Options:
        """Dynamic (user-tuneable) backend options."""
        return Options(
            shots=1024,
            job_settings=None,
            error_mitigation=None,
            extra_query_params={},
            extra_metadata={},
            sampler_seed=None,  # simulator-only (harmless on QPU)
            noise_model="ideal",  # simulator-only
        )

    @property
    def target(self) -> Target | None:
        return self._target

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def basis_gates(self) -> Sequence[str]:
        """Return the basis gates for this backend."""
        return self._basis_gates

    @property
    def coupling_map(self) -> CouplingMap:
        """IonQ hardware is fully connected."""
        return CouplingMap.from_full(self._num_qubits)

    @property
    def max_circuits(self) -> int | None:
        return self._max_experiments

    def gateset(self) -> Literal["qis", "native"]:
        """Active gateset (``"qis"`` or ``"native"``)."""
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

    @property
    def _api_backend_name(self) -> str:
        """Backend name used by the IonQ API (e.g., `qpu.aria-1`)."""
        # QPU names are `ionq_qpu.*` locally; API expects `qpu.*`
        return self.name.replace("ionq_qpu", "qpu")

    def run(
        self, run_input: QuantumCircuit | Sequence[QuantumCircuit], **options
    ) -> ionq_job.IonQJob:
        """Create and run a job on an IonQ Backend.

        Args:
            run_input: A single or list of Qiskit QuantumCircuit object(s).
            **options: Additional options for the job.

        Returns:
            IonQJob: A reference to the job that was submitted.
        """
        circuits = run_input if isinstance(run_input, (list, tuple)) else [run_input]

        if not all(self._has_measurements(c) for c in circuits):
            warnings.warn(
                "Circuit is not measuring any qubits", UserWarning, stacklevel=2
            )

        # Merge default options with user overrides
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

    def retrieve_job(self, job_id: str) -> ionq_job.IonQJob:
        """Retrieve a job by its ID."""
        return ionq_job.IonQJob(self, job_id, self.client)

    def retrieve_jobs(self, job_ids: Sequence[str]) -> Sequence[ionq_job.IonQJob]:
        """Retrieve multiple jobs by their IDs."""
        return [ionq_job.IonQJob(self, jid, self.client) for jid in job_ids]

    def cancel_job(self, job_id: str) -> dict:
        """Cancel a job by its ID."""
        return self.client.cancel_job(job_id)

    def cancel_jobs(self, job_ids: Sequence[str]) -> Sequence[dict]:
        """Cancel a list of jobs by their IDs."""
        return [self.client.cancel_job(job_id) for job_id in job_ids]

    def calibration(self) -> Characterization | None:
        """Return the latest characterization data (None for simulator)."""
        if self._simulator:
            return None
        return self.client.get_calibration_data(self._api_backend_name, limit=1)

    def status(self) -> bool:
        """True if the backend is currently available."""
        cal = self.calibration()
        return bool(cal and getattr(cal, "status", "available") == "available")

    def __eq__(self, other):
        if not isinstance(other, IonQBackend):
            return NotImplemented
        return (self.name, self._gateset) == (other.name, other._gateset)

    def __hash__(self):
        return hash((self.name, self._gateset))

    def _make_target(self) -> Target:
        """Build a Target exposing either QIS or IonQ-native gates."""
        tgt = Target(num_qubits=self._num_qubits)

        if self._gateset == "qis":
            theta = Parameter("θ")
            for gate in (
                # 1-qubit (fixed)
                IGate(),
                XGate(),
                YGate(),
                ZGate(),
                HGate(),
                SGate(),
                SdgGate(),
                SXGate(),
                SXdgGate(),
                TGate(),
                TdgGate(),
                # 1-qubit (parameterized)
                RXGate(theta),
                RYGate(theta),
                RZGate(theta),
                PhaseGate(theta),
                # 2-qubit (fixed)
                CXGate(),
                CYGate(),
                CZGate(),
                CHGate(),
                CSXGate(),
                SwapGate(),
                # 2-qubit (parameterized)
                CRXGate(theta),
                CRYGate(theta),
                CRZGate(theta),
                CPhaseGate(theta),
                RXXGate(theta),
                RYYGate(theta),
                RZZGate(theta),
            ):
                tgt.add_instruction(gate)

            tgt.add_instruction(MCXGate, name="mcx")
            tgt.add_instruction(MCPhaseGate, name="mcphase")
            tgt.add_instruction(PauliEvolutionGate, name="PauliEvolution")

        else:
            # 1q native
            phi = Parameter("φ")
            for gate in (GPIGate(phi), GPI2Gate(phi)):
                tgt.add_instruction(gate)

            # 2q native
            if "forte" in self.name.lower():
                theta = Parameter("θ")
                tgt.add_instruction(ZZGate(theta))
            else:
                phi0, phi1, theta = Parameter("φ0"), Parameter("φ1"), Parameter("θ")
                tgt.add_instruction(MSGate(phi0, phi1, theta))

        # Always allow measure/reset
        for cls in (Measure, Reset):
            tgt.add_instruction(cls())

        return tgt

    @staticmethod
    def _has_measurements(circ: QuantumCircuit) -> bool:
        return any(inst.operation.name == "measure" for inst in circ.data)


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
        probabilities, rather than a multiple of shots.
    """

    def __init__(
        self,
        provider: IonQProvider,
        name: str = "simulator",
        gateset: Literal["qis", "native"] = "qis",
        **initial_options,
    ):
        backend_name = name if name.startswith("ionq_") else f"ionq_{name}"
        super().__init__(
            provider=provider,
            name=backend_name,
            description="IonQ cloud simulator",
            gateset=gateset,
            num_qubits=get_n_qubits(name),
            simulator=True,
            max_shots=1,
            max_experiments=None,
            **initial_options,
        )

    def with_name(self, name: str, **kwargs) -> IonQSimulatorBackend:
        """Helper method that returns this backend with a more specific target system."""
        return IonQSimulatorBackend(self._provider, name, **kwargs)


class IonQQPUBackend(IonQBackend):
    """IonQ trapped-ion hardware back-ends (Aria/Alpine: MS; Forte: ZZ)."""

    def __init__(
        self,
        provider: IonQProvider,
        name: str = "ionq_qpu",
        gateset: Literal["qis", "native"] = "qis",
        **initial_options,
    ):
        super().__init__(
            provider=provider,
            name=name,
            description="IonQ trapped-ion QPU",
            gateset=gateset,
            num_qubits=get_n_qubits(name),
            simulator=False,
            max_shots=10_000,
            max_experiments=None,
            **initial_options,
        )

    def with_name(self, name: str, **kwargs) -> IonQQPUBackend:
        """Helper method that returns this backend with a more specific target system."""
        return IonQQPUBackend(self._provider, name, **kwargs)


__all__ = ["IonQBackend", "IonQSimulatorBackend", "IonQQPUBackend"]
