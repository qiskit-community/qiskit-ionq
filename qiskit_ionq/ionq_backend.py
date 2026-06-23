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
from .helpers import GATESET_MAP, get_backend_config, warn_bad_transpile_level
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
        simulator: bool,
        num_qubits: int | None = None,
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
        self._simulator: bool = simulator
        self._max_experiments: int | None = max_experiments
        self._max_shots: int | None = max_shots

        # Static backend configuration (qubits, supported/native gates, error
        # mitigations) from a single GET /backends/{id} via the provider's
        # credentials. ``{}`` when offline -> callers fall back to defaults.
        # Skipped when ``num_qubits`` is given explicitly (e.g. test mocks) so
        # construction stays network-free.
        if num_qubits is None:
            creds = self._provider.credentials
            self._config = get_backend_config(
                name, creds.get("token"), creds.get("url")
            )
            num_qubits = int(self._config.get("qubits", 4))
        else:
            self._config = {}
        self._num_qubits: int = num_qubits

        # Target & coupling map are resolved lazily (on first ``.target`` /
        # ``.coupling_map`` access) so backend construction stays network-free.
        # QPU connectivity is then read from the latest characterization and
        # cached; simulators and any fetch failure fall back to all-to-all.
        self._target: Target | None = None
        self._coupling_map: CouplingMap | None = None
        self._restricted_coupling: bool = False
        # Track noise_model for native simulator target caching
        self._cached_noise_model: str | None = None

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
            memory=False,
            dry_run=False,  # if True, the API compiles but does not execute
        )

    @property
    def target(self) -> Target | None:
        # The native simulator target depends on the selected noise model (it
        # picks the matching 2q gate), so rebuild it when that changes.
        if self._simulator and self._gateset == "native":
            current_noise_model = getattr(self.options, "noise_model", "ideal")
            if self._target is None or self._cached_noise_model != current_noise_model:
                self._target = self._make_target()
                self._cached_noise_model = current_noise_model
        elif self._target is None:
            self._target = self._make_target()
        return self._target

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def basis_gates(self) -> Sequence[str]:
        """Return the basis gates for this backend."""
        return self._basis_gates

    @property
    def supported_gates(self) -> list[str]:
        """QIS gate names the backend accepts (API ``supported_gates``)."""
        return list(self._config.get("supported_gates") or [])

    @property
    def supported_native_gates(self) -> list[str]:
        """Native gate names the backend accepts (API ``supported_native_gates``)."""
        return list(self._config.get("supported_native_gates") or [])

    @property
    def supported_error_mitigations(self) -> list[str]:
        """Error-mitigation methods the backend supports (API field)."""
        return list(self._config.get("supported_error_mitigations") or [])

    @property
    def coupling_map(self) -> CouplingMap:
        """Physical qubit connectivity.

        Built from the latest characterization's ``connectivity`` (the set of
        valid two-qubit pairs) for QPUs that report a restricted topology --
        e.g. Tempo's multi-parcel layout. Falls back to all-to-all for
        simulators, the generic ``ionq_qpu`` meta-backend, and IonQ's current
        fully-connected trapped-ion systems (Aria, Forte).
        """
        return self._resolve_coupling_map()

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
        """Backend name used by the IonQ API (e.g., `qpu.forte-1`)."""
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
        return self.client.get_latest_calibration(self._api_backend_name)

    def status(self) -> bool:
        """True if the backend is currently available.

        Simulators short-circuit to ``True``: they have no characterization
        endpoint but are reachable whenever the API is.
        """
        if self._simulator:
            return True
        cal = self.calibration()
        return bool(cal and getattr(cal, "status", "available") == "available")

    def __eq__(self, other):
        if not isinstance(other, IonQBackend):
            return NotImplemented
        return (self.name, self._gateset) == (other.name, other._gateset)

    def __hash__(self):
        return hash((self.name, self._gateset))

    def _fetch_connectivity(self) -> list[tuple[int, int]] | None:
        """Valid two-qubit pairs from the latest characterization.

        Returns ``None`` for simulators, when the backend reports no
        connectivity, or when the lookup fails for any reason (offline,
        missing credentials, unknown system) -- callers then default to
        all-to-all.
        """
        if self._simulator:
            return None
        try:
            cal = self.calibration()
        except Exception as exc:  # pylint: disable=broad-except
            warnings.warn(
                f"Could not fetch connectivity for {self.name}: {exc}. "
                "Falling back to all-to-all coupling.",
                stacklevel=2,
            )
            return None
        if cal is not None and cal.connectivity:
            return cal.connectivity
        return None

    def _resolve_coupling_map(self) -> CouplingMap:
        """Coupling map for this backend, cached after first resolution.

        Restricted topologies come from the characterization's ``connectivity``
        (a list of undirected ``[i, j]`` pairs). IonQ two-qubit gates are
        symmetric, so each pair is added in both directions; pairs outside
        ``num_qubits`` are ignored defensively. A *complete* graph (which IonQ's
        all-to-all systems report in full) and anything missing resolve to an
        all-to-all map left implicit in the Target -- cheaper to build and
        identical for routing. Only a genuine subset marks the Target
        restricted (e.g. Tempo's multi-parcel layout).
        """
        if self._coupling_map is not None:
            return self._coupling_map

        n = self._num_qubits
        pairs = self._fetch_connectivity()
        if pairs:
            edges: set[tuple[int, int]] = set()
            for left, right in pairs:
                left, right = int(left), int(right)
                for src, dst in ((left, right), (right, left)):
                    if src != dst and 0 <= src < n and 0 <= dst < n:
                        edges.add((src, dst))
            # n*(n-1) directed edges == complete graph -> treat as all-to-all.
            if edges and len(edges) < n * (n - 1):
                cmap = CouplingMap()
                for qubit in range(n):
                    cmap.add_physical_qubit(qubit)
                for src, dst in sorted(edges):
                    cmap.add_edge(src, dst)
                self._coupling_map = cmap
                self._restricted_coupling = True
                return self._coupling_map

        self._coupling_map = CouplingMap.from_full(n)
        self._restricted_coupling = False
        return self._coupling_map

    def _make_target(self) -> Target:
        """Build a Target exposing either QIS or IonQ-native gates.

        Two-qubit gates are constrained to the backend's coupling map when the
        characterization reports a restricted topology; otherwise they are
        added as globally available (all-to-all), matching IonQ's fully
        connected trapped-ion systems.
        """
        tgt = Target(num_qubits=self._num_qubits)

        # Resolve connectivity first so 2q gates can be scoped to valid pairs.
        self._resolve_coupling_map()
        two_q_props = (
            {edge: None for edge in self._coupling_map.get_edges()}
            if self._restricted_coupling and self._coupling_map is not None
            else None
        )

        def add(instruction) -> None:
            """Add a gate, scoping 2q gates to the coupling map when restricted."""
            if two_q_props is not None and getattr(instruction, "num_qubits", 0) == 2:
                tgt.add_instruction(instruction, dict(two_q_props))
            else:
                tgt.add_instruction(instruction)

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
                add(gate)

            tgt.add_instruction(MCXGate, name="mcx")
            tgt.add_instruction(MCPhaseGate, name="mcphase")
            tgt.add_instruction(PauliEvolutionGate, name="PauliEvolution")

        else:
            # 1q native
            phi = Parameter("φ")
            for gate in (GPIGate(phi), GPI2Gate(phi)):
                add(gate)

            # 2q native: ms (Aria) or zz (Forte/Tempo), from the API's
            # supported_native_gates -- see _two_q_native_gate().
            if self._two_q_native_gate() == "zz":
                theta = Parameter("θ")
                add(ZZGate(theta))
            else:
                phi0, phi1, theta = Parameter("φ0"), Parameter("φ1"), Parameter("θ")
                add(MSGate(phi0, phi1, theta))

        # Always allow measure/reset
        for cls in (Measure, Reset):
            tgt.add_instruction(cls())

        return tgt

    def _two_q_native_gate(self) -> Literal["ms", "zz"]:
        """The device's native two-qubit gate, from ``supported_native_gates``.

        QPUs use their own config. The simulator follows its active
        ``noise_model`` (which names the emulated device); with no/``ideal``
        noise model it has no real device, so it defaults to ``"ms"`` -- also
        the fallback whenever the API is unreachable.
        """
        config = self._config
        if self._simulator:
            noise_model = getattr(self.options, "noise_model", None)
            if not noise_model or noise_model == "ideal":
                return "ms"
            creds = self._provider.credentials
            config = get_backend_config(
                noise_model, creds.get("token"), creds.get("url")
            )
        return "zz" if "zz" in (config.get("supported_native_gates") or []) else "ms"

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
            simulator=True,
            max_shots=1,
            max_experiments=None,
            **initial_options,
        )

    def with_name(self, name: str, **kwargs) -> IonQSimulatorBackend:
        """Helper method that returns this backend with a more specific target system."""
        return IonQSimulatorBackend(self._provider, name, **kwargs)


class IonQQPUBackend(IonQBackend):
    """IonQ trapped-ion hardware back-ends (Aria: MS; Forte/Tempo: ZZ)."""

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
            simulator=False,
            max_shots=10_000,
            max_experiments=None,
            **initial_options,
        )

    def with_name(self, name: str, **kwargs) -> IonQQPUBackend:
        """Helper method that returns this backend with a more specific target system."""
        return IonQQPUBackend(self._provider, name, **kwargs)


__all__ = ["IonQBackend", "IonQSimulatorBackend", "IonQQPUBackend"]
