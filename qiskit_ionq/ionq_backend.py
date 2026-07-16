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
from .helpers import GATESET_MAP, api_backend_id, warn_bad_transpile_level
from .ionq_client import Characterization

if TYPE_CHECKING:  # pragma: no cover
    from .ionq_provider import IonQProvider

# Fallback qubit count when the /backends catalog has no config for a backend
# (offline, unknown id), so a Target can still be built; real counts come from
# the API.
_DEFAULT_NUM_QUBITS = 4


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

        # Static config (qubits, gates, capabilities) comes from the provider's
        # cached /backends catalog. An explicit num_qubits pins the caller's
        # own config and bypasses the catalog; otherwise a missing entry
        # yields {} and we fall back to a small default qubit count.
        self._config: dict = {}
        if num_qubits is None:
            self._config = self._provider.backend_config(name)
            num_qubits = int(self._config.get("qubits", _DEFAULT_NUM_QUBITS))
        self._num_qubits: int = num_qubits

        # Target and coupling map are resolved lazily on first access, keeping
        # construction network-free.
        self._target: Target | None = None
        self._coupling_map: CouplingMap | None = None
        self._restricted_coupling: bool = False
        # noise_model the native-simulator Target was last built for, so it
        # gets rebuilt when the user switches noise models.
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
            debiasing=None,
            symmetry_verification=None,
            extra_query_params={},
            extra_metadata={},
            sampler_seed=None,  # simulator-only (harmless on QPU)
            noise_model="ideal",  # simulator-only
            memory=False,
            dry_run=False,  # if True, the API compiles but does not execute
        )

    @property
    def target(self) -> Target | None:
        # A native-gateset simulator's target depends on the active noise model
        # (it selects the 2q gate), so rebuild it when that changes.
        native_sim = self._simulator and self._gateset == "native"
        current_noise_model = getattr(self.options, "noise_model", "ideal")
        if self._target is None or (
            native_sim and self._cached_noise_model != current_noise_model
        ):
            self._target = self._make_target()
            self._cached_noise_model = current_noise_model
        return self._target

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def basis_gates(self) -> Sequence[str]:
        """Return the basis gates for this backend."""
        return self._basis_gates

    def _config_list(self, key: str) -> list[str]:
        """Read a list-valued key from the API config, warning when the config
        is unavailable so an empty result is distinguishable from "none"."""
        if not self._config:
            warnings.warn(
                f"No API config available for backend {self.name!r}; "
                f"{key} is unknown (returning an empty list)."
            )
        return list(self._config.get(key) or [])

    @property
    def supported_gates(self) -> list[str]:
        """QIS gate names the API accepts; empty (with a warning) if the config
        was unavailable."""
        return self._config_list("supported_gates")

    @property
    def supported_native_gates(self) -> list[str]:
        """Native gate names the API accepts; empty (with a warning) if the config
        was unavailable."""
        return self._config_list("supported_native_gates")

    @property
    def supported_error_mitigations(self) -> list[str]:
        """Supported error-mitigation techniques; empty (with a warning) if the
        config was unavailable."""
        return self._config_list("supported_error_mitigations")

    @property
    def coupling_map(self) -> CouplingMap:
        """Qubit connectivity from the characterization, else all-to-all."""
        return self._get_coupling_map()

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
        """Backend name used by the IonQ API (e.g. ``qpu.forte-1``)."""
        return api_backend_id(self.name)

    def run(
        self, run_input: QuantumCircuit | Sequence[QuantumCircuit], **options
    ) -> ionq_job.IonQJob:
        """Create and run a job on an IonQ Backend.

        Args:
            run_input: A single or list of Qiskit QuantumCircuit object(s).
            **options: Additional options for the job, overriding the backend
                defaults (see :meth:`_default_options`). Notable options:

                - ``shots`` (int): Number of shots (default 1024).
                - ``debiasing`` (bool): Enable debiasing error mitigation, which
                  runs the circuit as multiple symmetrized variants to suppress
                  systematic hardware biases. Requires at least 500 shots. When
                  unset, the IonQ platform default for the target applies.
                - ``symmetry_verification`` (bool): Enable symmetry verification,
                  discarding measurement outcomes that violate the circuit's
                  symmetries. When unset, the platform default applies.
                - ``job_settings`` (dict): Raw ``settings`` payload passed through
                  to the API for options without a dedicated kwarg.

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
        """Connectivity pairs (valid two-qubit pairs) from the characterization.

        Returns ``None`` -- meaning all-to-all -- for simulators, a backend
        that reports no connectivity, or any failed lookup (offline, missing
        credentials, unknown system).
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

    def _get_coupling_map(self) -> CouplingMap:
        """Backend coupling map, resolved once and cached.

        Uses the characterization's connectivity when it describes a genuine
        *subset* of pairs (e.g. Tempo's multi-parcel layout), symmetrizing each
        pair since IonQ two-qubit gates are direction-agnostic. A complete or
        missing graph -- what IonQ's all-to-all systems report -- resolves to
        an implicit all-to-all map instead.
        """
        if self._coupling_map is not None:
            return self._coupling_map

        n = self._num_qubits
        pairs = self._fetch_connectivity()
        if pairs:
            # Collect into a set (symmetrize + dedupe + bounds-check) so we can
            # compare the edge count against a complete graph below.
            edges = {
                (src, dst)
                for left, right in pairs
                for src, dst in ((int(left), int(right)), (int(right), int(left)))
                if src != dst and 0 <= src < n and 0 <= dst < n
            }
            # n*(n-1) directed edges == a complete graph (all-to-all); only a
            # strict subset is treated as a restricted topology.
            if edges and len(edges) < n * (n - 1):
                cmap = CouplingMap()
                for qubit in range(n):  # keep isolated qubits as nodes
                    cmap.add_physical_qubit(qubit)
                for src, dst in sorted(edges):  # sorted for deterministic order
                    cmap.add_edge(src, dst)
                self._coupling_map = cmap
                self._restricted_coupling = True
                return self._coupling_map

        self._coupling_map = CouplingMap.from_full(n)
        self._restricted_coupling = False
        return self._coupling_map

    def _make_target(self) -> Target:
        """Build a Target of QIS or native gates, scoping 2q gates to the coupling
        map on restricted topologies (else use all-to-all connectivity)."""
        tgt = Target(num_qubits=self._num_qubits)

        self._get_coupling_map()
        two_q_props = (
            {edge: None for edge in self._coupling_map.get_edges()}
            if self._restricted_coupling and self._coupling_map is not None
            else None
        )

        def add_gate(instruction, name: str | None = None) -> None:
            """Add a gate, scoping 2q gates to valid pairs on restricted topologies."""
            if two_q_props is not None and getattr(instruction, "num_qubits", 0) == 2:
                tgt.add_instruction(instruction, dict(two_q_props), name=name)
            else:
                tgt.add_instruction(instruction, name=name)

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
                add_gate(gate)

            add_gate(MCXGate, name="mcx")
            add_gate(MCPhaseGate, name="mcphase")
            add_gate(PauliEvolutionGate, name="PauliEvolution")

        else:
            # 1q native
            phi = Parameter("φ")
            for gate in (GPIGate(phi), GPI2Gate(phi)):
                add_gate(gate)

            # 2q native: ms (Aria) or zz (Forte/Tempo), per the API config.
            if self._two_q_native_gate() == "zz":
                theta = Parameter("θ")
                add_gate(ZZGate(theta))
            else:
                phi0, phi1, theta = Parameter("φ0"), Parameter("φ1"), Parameter("θ")
                add_gate(MSGate(phi0, phi1, theta))

        # Always allow measure/reset
        for cls in (Measure, Reset):
            add_gate(cls())

        return tgt

    def _two_q_native_gate(self) -> Literal["ms", "zz"]:
        """Native 2q gate from ``supported_native_gates`` (the simulator follows
        its noise model); defaults to ``"ms"`` for ideal-sim and offline."""
        config = self._config
        if self._simulator:
            noise_model = getattr(self.options, "noise_model", None)
            if not noise_model or noise_model == "ideal":
                return "ms"
            config = self._provider.backend_config(noise_model)
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
