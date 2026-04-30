# Migrating from `qiskit-ionq` 1.x to 2.0

`qiskit-ionq` 2.0 is a from-scratch reimplementation on top of the official low-level [`ionq-core`](https://github.com/ionq/ionq-core-python) Python client. It replaces the hand-rolled `requests`-based transport, adds a [V2 Sampler primitive](https://quantum.cloud.ibm.com/docs/migration-guides/v2-primitives), and rebuilds the `Target` from live IonQ characterization data each time a backend is constructed.

The 2.0 public surface is **not source-compatible** with 1.x. This document covers every observable change.

`1.1.0` ships the deprecation warnings listed below. Pin to `qiskit-ionq<2` if you need more time to migrate.

## Python version floor

`qiskit-ionq` 1.1.0 and 2.0.0 both require **Python ≥ 3.12**. Python 3.10 and 3.11 are no longer supported. (The floor is set by `ionq-core`.) If you are on 3.10 or 3.11, pin `qiskit-ionq==1.0.2` until you can upgrade Python.

## Authentication and provider construction

```python
# 1.x
from qiskit_ionq import IonQProvider
provider = IonQProvider(token="...", url="https://api.ionq.co/v0.4", custom_headers={"X-Foo": "bar"})

# 2.0
from qiskit_ionq import IonQProvider
provider = IonQProvider(api_key="...", base_url="https://api.ionq.co/v0.4")
```

- `token=` → `api_key=`.
- `url=` → `base_url=`.
- `custom_headers=` is removed. The provider sets internal headers (`X-Qiskit-Version`, user-agent suffix `qiskit-ionq/<version>`) through `ionq_core.ClientExtension`. If you need additional headers, build the `ionq_core.IonQClient` yourself and pass a custom `ClientExtension`.
- The `IONQ_API_KEY` environment variable still works (read by `ionq-core`).

## Backends are dynamic; legacy stub names are gone

```python
# 1.x
backend = provider.get_backend("ionq_qpu.aria-1")
sim     = provider.get_backend("ionq_simulator")

# 2.0
backend = provider.get_backend("qpu.aria-1")
sim     = provider.get_backend("simulator")
```

- The `ionq_` prefix is gone. Names match what the IonQ API returns.
- `IonQQPUBackend` and `IonQSimulatorBackend` no longer exist. `provider.backends()` calls `/backends` live and returns a list of `IonQBackend` instances. There is one class.
- Use `provider.backends()` (no `name=`) to list everything your API key can reach.

## Job submission

```python
# 1.x — supports many extra kwargs
job = backend.run(qc, shots=2000,
                  error_mitigation=ErrorMitigation.DEBIASING,
                  noise_model="aria-1",
                  sampler_seed=42,
                  extra_query_params={...},
                  extra_metadata={...},
                  job_settings={...})

# 2.0 — narrower, typed kwargs
job = backend.run(qc, shots=2000,
                  error_mitigation={"debiasing": True},
                  noise_model=NoiseModel.ARIA_1,
                  noise_seed=42)
```

- `error_mitigation=ErrorMitigation.DEBIASING` → `error_mitigation={"debiasing": True}`. The `ErrorMitigation` enum is removed. Pass any settings dict the API accepts.
- `noise_model="aria-1"` → `noise_model=ionq_core.models.NoiseModel.ARIA_1`. The string-keyed form is removed.
- `sampler_seed=` → `noise_seed=`. The seed now drives the simulator's noise sampling, not a client-side counts-from-probabilities resampler.
- `extra_query_params=` and `extra_metadata=` are removed. There is no escape hatch into the API request body.
- `job_settings=` is removed. Options that used to live there now live on the typed payload (`error_mitigation`, `noise_model`, `noise_seed`).

## Sampler primitive (new)

2.0 ships a real V2 `BaseSamplerV2` implementation. Use it for parameterized sweeps and multi-circuit batching.

```python
from qiskit_ionq import IonQProvider, IonQSampler
backend = IonQProvider().get_backend("simulator")

sampler = IonQSampler(backend)
job = sampler.run([(qc1,), (qc2, params)], shots=4096)
result = job.result()              # PrimitiveResult[SamplerPubResult]
counts = result[0].data.meas.get_counts()
```

There is **no `Estimator` in 2.0**. Build it on top of `IonQSampler` if you need expectation values, or call `ionq_core` directly.

## Sessions

```python
# 1.x
from qiskit_ionq import Session
with Session(backend, max_jobs=10) as sess:
    backend.run(qc)               # session_id auto-injected via monkey-patch
    backend.run(qc)

# 2.0
from qiskit_ionq import IonQSession
with IonQSession(backend, max_jobs=10) as sess:
    sampler = sess.sampler()      # explicit; binds session to the sampler
    sampler.run([(qc,)])
```

- `Session` → `IonQSession`. The class is now a wrapper over `ionq_core.SessionManager`. **Already available in 1.1.0** — you can migrate today on Python >=3.12.
- The monkey-patch on `backend.run` is gone. In 1.1.0 you can pass `session_id=sess.session_id` explicitly. In 2.0 you call `session.sampler().run(...)` (or pass the session into `IonQSampler(backend, session=sess)`).
- `IonQSession.from_id(backend, "session-uuid")` reattaches to an existing session.

## Native gates

```python
# 1.x and 2.0 both export these names; same `phi`/`angle` semantics.
from qiskit_ionq import GPIGate, GPI2Gate, MSGate, ZZGate
```

- Phase parameters (`phi`, `phi0`, `phi1`) are still in **turns** (cycles of 2π).
- Interaction parameters (`angle`) are still in **units of π**.
- The 2.0 versions back `__array__()` with `ionq_core`'s native-gate matrix helpers (`gpi_matrix`, `gpi2_matrix`, `ms_matrix`, `zz_matrix`).
- `gateset="native"` on `provider.get_backend()` still selects the native basis.

## Mid-circuit measurement

**1.x raised `IonQMidCircuitMeasurementError`.** **2.0 silently drops** any `measure` instruction during translation (`_translate.py: _SKIP`), and the API returns probabilities over the full register at the end. **Move all measurements to the end of your circuit.** If you submit a circuit with mid-circuit measures, you will get results that look correct but ignore those mid-circuit operations.

This is a behavior change worth flagging in your test suite before upgrading.

## Transpiler passes

The IonQ-specific optimizer plugin and rewrite-rule passes are not in 2.0:

- `TrappedIonOptimizerPlugin`, `TrappedIonOptimizerPluginSimpleRules`, `TrappedIonOptimizerPluginCompactGates`, `TrappedIonOptimizerPluginCommuteGpi2ThroughMs` — removed.
- `add_equivalences` and the `IonQEquivalenceLibrary` — removed.
- `IonQTranspileLevelWarning` — removed; no warning is issued for `optimization_level=2/3` on QIS submissions.

2.0 calls Qiskit's stock `transpile(circuit, target=backend.target)` with no IonQ-specific passes. Expect more gates after transpilation than 1.x produced for the same circuit. If you depend on the legacy passes, either:

- pin to `qiskit-ionq<2`, or
- copy the rewrite rules from 1.x's `qiskit_ionq/rewrite_rules.py` into your own `PassManager`.

The `Target` is built fresh on every `IonQBackend` construction by querying `/backends/<name>/characterizations/<uuid>`, so per-qubit gate errors and SPAM fidelity reflect live calibration.

## Job lifecycle

```python
# 1.x
job = backend.run(qc)
job.status(detailed=True)         # multi-circuit child status dict
job.queue_position()               # never existed; raised
result = job.result(sharpen=True, callback=cb, wait=5, timeout=300, extra_query_params={...})
backend.retrieve_job(job_id)       # rebuild from API
backend.calibration()              # Characterization wrapper
backend.status()                   # bool

# 2.0
job = backend.run(qc)
job.status()                       # JobStatus enum
result = job.result(timeout=300.0) # plain Result
job.cancel()
# retrieve_job / calibration / status are gone — wire ionq_core directly if you need them
```

- `result(sharpen=...)`, `result(callback=...)`, `result(extra_query_params=...)`, `result(wait=...)` — all removed. `result(timeout=300.0)` is the only kwarg.
- `job.status(detailed=True)` is removed; `IonQJob.status()` returns a `JobStatus` enum and nothing else.
- `backend.retrieve_job(job_id)` and `backend.retrieve_jobs(...)` are removed. Build an `IonQJob(backend, job_id, client, num_qubits, shots)` manually if you need to reattach.
- `backend.calibration()` is removed. The characterization is consumed internally to build `Target`. Call `ionq_core.api.characterizations.get_characterization.sync(...)` directly if you want the raw object.
- `backend.status()` is removed. There is no health-check method.
- `IonQJobError`, `IonQJobFailureError`, `IonQJobStateError`, `IonQJobTimeoutError` are removed. Catch `RuntimeError`, `ValueError`, or `ionq_core.exceptions.APIError` and subclasses.

## Exceptions

```python
# 1.x
from qiskit_ionq import IonQAPIError, IonQRetriableError, IonQGateError
try:
    backend.run(qc)
except IonQAPIError as exc:
    log.error("API error %s: %s", exc.status_code, exc.message)

# 2.0
from ionq_core.exceptions import APIError, RateLimitError, ServerError, AuthenticationError
try:
    backend.run(qc)
except APIError as exc:
    log.error("API error %s: %s (request_id=%s)", exc.status_code, exc.message, exc.request_id)
```

The entire `qiskit_ionq.exceptions` module is removed. All of these go away: `IonQError`, `IonQClientError`, `IonQAPIError`, `IonQRetriableError`, `IonQBackendError`, `IonQCredentialsError`, `IonQGateError`, `IonQMidCircuitMeasurementError`, `IonQPauliExponentialError`, `IonQTranspileLevelWarning`, `IonQJobError`, `IonQJobFailureError`, `IonQJobStateError`, `IonQJobTimeoutError`.

`ionq-core`'s typed exceptions carry `.status_code`, `.body`, `.message`, and `.request_id`. `RateLimitError` has `.retry_after`.

## Helpers and `IonQClient`

```python
# 1.x — the package shipped a hand-rolled REST client
from qiskit_ionq.ionq_client import IonQClient
from qiskit_ionq.helpers import qiskit_to_ionq, get_n_qubits
client = IonQClient(token, url)
job_id = client.submit_job(...)["id"]
nq = get_n_qubits("aria-1")

# 2.0 — drop to ionq-core directly
from ionq_core import IonQClient
from ionq_core.api.default import create_job, get_job
from ionq_core.api.backends import get_backend
client = IonQClient(api_key=...)
nq = get_backend.sync(client=client, name="qpu.aria-1").qubits
```

- `qiskit_ionq.IonQClient` is removed. Use `ionq_core.IonQClient` (a factory returning `AuthenticatedClient`).
- `qiskit_ionq.helpers` is removed entirely. `qiskit_to_ionq`, `qiskit_circ_to_ionq_circ`, `get_n_qubits`, `compress_to_metadata_string`, `decompress_metadata_string`, `SafeEncoder`, `retry`, `resolve_credentials`, `get_user_agent` — all gone.
- `qiskit_ionq.constants.APIJobStatus` and `JobStatusMap` — removed. The mapping is a 7-line dict in `qiskit_ionq.job._STATUS_MAP`.

## Result format

Counts are still little-endian Qiskit-style bitstring → int. Two behavioral differences:

1. The "ideal" simulator path no longer client-samples counts via `np.random.RandomState(sampler_seed)`. 2.0 uses deterministic largest-remainder rounding so counts always sum to exactly `shots`.
2. `IonQResult.get_probabilities()` is removed. The branch returns a plain `qiskit.result.Result`. Read counts via `result.get_counts()`.

For the new sampler primitive, the result type is `PrimitiveResult[SamplerPubResult]` and counts come from `result[i].data.meas.get_counts()`.

## CI / tooling

- Linting: pylint → ruff (`ruff check`, `ruff format`).
- Type checking: a new `ty check` pass.
- Build: `setup.py` → `pyproject.toml` + `hatchling`. Use `uv build` / `uv publish`.
- Lockfile: `uv.lock` is committed.

## Quick checklist

- [ ] Python ≥ 3.12.
- [ ] `IonQProvider(api_key=...)` (not `token=`).
- [ ] Backend names without `ionq_` prefix.
- [ ] `error_mitigation` is a dict, not the enum.
- [ ] `noise_model` is `ionq_core.models.NoiseModel`, not a string.
- [ ] No `extra_query_params`, `extra_metadata`, `sampler_seed`, `job_settings` in `backend.run`.
- [ ] All measurements at the end of every circuit.
- [ ] Catch `ionq_core.exceptions.APIError`, not `IonQAPIError`.
- [ ] Replace `Session` with `IonQSession` and call `.sampler()` explicitly.
- [ ] Drop any direct use of `qiskit_ionq.helpers`, `qiskit_ionq.IonQClient`, or `qiskit_ionq.exceptions.*`.
- [ ] Stop relying on `TrappedIonOptimizerPlugin` or `add_equivalences()`.
