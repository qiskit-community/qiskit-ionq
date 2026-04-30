# Qiskit IonQ Provider

[![License](https://img.shields.io/github/license/ionq/qiskit-ionq.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/ionq/qiskit-ionq/actions/workflows/ci.yml/badge.svg)](https://github.com/ionq/qiskit-ionq/actions/workflows/ci.yml)
[![Python versions](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/qiskit-%E2%89%A52.0-6929C4)](https://github.com/Qiskit/qiskit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://docs.astral.sh/ruff)

[Qiskit](https://github.com/Qiskit/qiskit) provider for [IonQ](https://ionq.com/) trapped-ion quantum computers. Run circuits on IonQ Aria, Forte, and the cloud simulator through the IonQ API, with support for native trapped-ion gates, multi-circuit batching, and priority sessions.

## Installation

```bash
pip install qiskit-ionq
```

## Quick Start

```python
from qiskit.circuit import QuantumCircuit
from qiskit_ionq import IonQProvider

provider = IonQProvider()  # uses IONQ_API_KEY environment variable
backend = provider.get_backend("simulator")

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

job = backend.run(qc, shots=1024)
result = job.result()
print(result.get_counts())  # {'00': 512, '11': 512}
```

## Features

- **Qiskit 2.0+ compatible** - implements BackendV2 and BaseSamplerV2
- **Multi-circuit batching** - submit multiple circuits in a single API call via `IonQSampler`
- **Priority sessions** - group related jobs for priority queuing on hardware
- **Native gate access** - program directly with GPI, GPI2, MS, and ZZ trapped-ion gates
- **Error mitigation** - enable debiasing for improved results on hardware
- **Noise models** - configure simulator noise to match specific QPU characteristics
- **Automatic transpilation** - Qiskit's transpiler decomposes circuits to the IonQ gate set using backend characterization data

## Authentication

Get an API key from the [IonQ Cloud Console](https://cloud.ionq.com/).

Set it as an environment variable (recommended):

```bash
export IONQ_API_KEY="your-api-key"
```

Or pass it directly:

```python
provider = IonQProvider(api_key="your-api-key")
```

## Backends

List all available backends:

```python
for backend in provider.backends():
    print(f"{backend.name}: {backend.num_qubits} qubits")
```

Get a specific backend:

```python
simulator = provider.get_backend("simulator")
aria = provider.get_backend("qpu.aria-1")
```

## Sampler

Use `IonQSampler` for the modern Qiskit primitives interface and multi-circuit batching:

```python
from qiskit_ionq import IonQSampler

sampler = IonQSampler(backend)
job = sampler.run([circuit1, circuit2, circuit3], shots=1024)
result = job.result()

for i, pub_result in enumerate(result):
    print(f"Circuit {i}: {pub_result.data.meas.get_counts()}")
```

## Sessions

Group jobs for priority execution on hardware:

```python
from qiskit_ionq import IonQSession

with IonQSession(backend, max_jobs=10, max_time=60, max_cost=100.0) as session:
    sampler = session.sampler()
    job1 = sampler.run([circuit1], shots=1024)
    job2 = sampler.run([circuit2], shots=1024)
    r1, r2 = job1.result(), job2.result()
```

Reconnect to an existing session:

```python
session = IonQSession.from_id(backend, "existing-session-id")
print(session.status())
```

## Native Gates

Access IonQ's trapped-ion native gates for lower-level control:

```python
from qiskit.circuit import QuantumCircuit
from qiskit_ionq import GPIGate, GPI2Gate, MSGate, ZZGate

backend = provider.get_backend("simulator", gateset="native")

qc = QuantumCircuit(2)
qc.append(GPI2Gate(0.5), [0])
qc.append(MSGate(0.0, 0.0), [0, 1])
qc.measure_all()

job = backend.run(qc)
```

All gate parameters use the IonQ convention: phase parameters (`phi`) are in turns (fractions of 2pi), interaction parameters (`angle`) are in units of pi.

| Gate | Qubits | Description |
|------|--------|-------------|
| `GPIGate(phi)` | 1 | Pi rotation about axis at angle phi |
| `GPI2Gate(phi)` | 1 | Pi/2 rotation about axis at angle phi |
| `MSGate(phi0, phi1, angle=0.25)` | 2 | Molmer-Sorensen entangling gate |
| `ZZGate(angle)` | 2 | ZZ interaction gate |

## Error Mitigation and Noise Models

Enable debiasing for improved results on QPU hardware:

```python
job = backend.run(qc, error_mitigation={"debiasing": True}, shots=1024)
```

Configure noise models for the simulator:

```python
simulator = provider.get_backend("simulator")
job = simulator.run(qc, noise_model="aria-1", noise_seed=42)
```

## Contributing

We welcome contributions. To set up a development environment:

```bash
git clone https://github.com/ionq/qiskit-ionq.git
cd qiskit-ionq
uv sync --group dev
```

### Linting and formatting

```bash
uv run ruff check qiskit_ionq/ tests/
uv run ruff format qiskit_ionq/ tests/
uv run ty check
```

### Running tests

Unit tests (95% coverage enforced):

```bash
uv run pytest
```

Integration tests against the real IonQ API (requires `IONQ_API_KEY`):

```bash
uv run python tests/integration_test.py
```

## License

[Apache License 2.0](LICENSE)
