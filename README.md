# Qiskit IonQ Provider

<img src="https://ionq.com/images/ionq-logo-dark.png" alt="IonQ Logo" width="350px"/>

[![License](https://img.shields.io/github/license/qiskit-community/qiskit-aqt-provider.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)

**Qiskit** is an open-source SDK for working with quantum computers at the level of circuits, algorithms, and application modules.

This project contains a provider that allows access to **[IonQ]** ion trap quantum
systems.

The example python notebook (in `/example`) should help you understand basic usage.

## API Access

The IonQ Provider uses IonQ's REST API, and using the provider requires an API access token from IonQ. If you would like to use IonQ as a Qiskit provider, please visit <https://cloud.ionq.com/settings/keys> to generate an IonQ API key.

## Installation

You can install the provider using pip:

```bash
pip install qiskit-ionq
```

## Provider Setup

To instantiate the provider, make sure you have an access token then create a provider:

```python
from qiskit_ionq import IonQProvider

provider = IonQProvider("token")
```

### Credential Environment Variables

Alternatively, the IonQ Provider can discover your access token from environment variables. It checks `QISKIT_IONQ_API_TOKEN`, then `IONQ_API_KEY`, then `IONQ_API_TOKEN`:

```bash
export IONQ_API_KEY="token"
```

Then invoke instantiate the provider without any arguments:

```python
from qiskit_ionq import IonQProvider

provider = IonQProvider()
```

Once the provider has been instantiated, it may be used to access supported backends:

```python
# Show all current supported backends:
print(provider.backends())

# Get IonQ's simulator backend:
simulator_backend = provider.get_backend("ionq_simulator")
```

### Submitting a Circuit

Once a backend has been specified, it may be used to submit circuits.
For example, running a Bell State:

```python
from qiskit import QuantumCircuit

# Create a basic Bell State circuit:
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

# Run the circuit on IonQ's platform with debiasing:
job = simulator_backend.run(qc, shots=1000, debiasing=True)

# Print the results.
print(job.result().get_counts())

# The simulator specifically provides the ideal probabilities and creates
# counts by sampling from these probabilities. The raw probabilities are also accessible:
print(job.result().get_probabilities())
```

### Error mitigation

#### Debiasing and aggregation

Debiasing is a compiler-level error-mitigation technique.
It creates physically different but logically equivalent variants of a circuit and divides the requested shots among them.
The variants have the same ideal output but different error profiles, allowing systematic hardware biases to be suppressed when their results are combined.

Enable debiasing when submitting the job. It requires at least 500 shots:

```python
job = backend.run(qc, shots=1000, debiasing=True)
```

Leaving `debiasing` unset defers to the IonQ platform default for the target; passing `False` explicitly disables it.
When a job is debiased, there are multiple options for how to combine variant results.
This is achieved by `job.result(aggregation=...)` where the options are:

1. `average` (default) takes the component-wise mean of the variant
   distributions. It preserves arbitrary distribution shapes and is the safest choice for broad or uneven output distributions.
2. `voting` performs plurality voting across variants, also called sharpening.
   It emphasizes outcomes that appear consistently across variants and is best suited to distributions with one or a few roughly equal peaks.
   It can distort broad or uneven distributions.
3. `dnl` applies debiasing with non-linear filtering, suppressing outcomes that are not observed consistently across variants.
   See [arXiv:2506.05757](https://arxiv.org/abs/2506.05757) for details.

For example:

```python
result = job.result(aggregation="voting")
print(result.get_counts())
```

The `aggregation` argument has no effect on a job that ran without debiasing.

#### Symmetry verification

Symmetry verification is a post-processing technique that uses circuit symmetries (such as conserved particle number, Hamming weight, or parity) to discard outcomes that violate those conserved quantities.
During server-side compilation, IonQ analyzes the circuit to find a conservative set of reachable computational-basis states using a primitive simulation method.
That set is available through `job.reachable_states` and can be passed to `result()` to discard outcomes outside it:

```python
job = backend.run(qc, shots=1000)
result = job.result(postselect_on=job.reachable_states)
```

For a multi-circuit job, `job.reachable_states` contains one set of bitstrings per circuit.
If a circuit cannot be analyzed, its entry is `None` and its result is left unchanged.
**Post-selection does not renormalize aggregate probabilities**.

Because symmetry verification is applied after aggregation, it can be combined with debiasing and any aggregation method:

```python
job = backend.run(qc, shots=1000, debiasing=True)
result = job.result(
    aggregation="dnl",
    postselect_on=job.reachable_states,
)
```

The `postselect_on` kwarg can be used with custom post-selection as well by passing any collection of bitstrings.

### Compilation as a service (`dry_run`)

IonQ Cloud can compile a circuit and return the result _without_ executing it on a QPU. This is useful for inspecting the post-compilation circuit, estimating gate counts, or validating native-gate output before paying for shots.

Set `dry_run=True` on `backend.run(...)` and then read the compiled circuit back via `job.compiled_circuit(...)`:

```python
backend = provider.get_backend("ionq_qpu.forte-1")

job = backend.run(qc, dry_run=True, job_settings={"compilation": {"service_version": "v0.4"}})
job.wait_for_final_state()

native = job.compiled_circuit(lang="native")   # IonQ-native gate JSON (dict)
```

The compiled circuit is fetched from the job's published artifacts (`output.compilation.compiled_circuits`); `lang` is matched against the available format keys (e.g. `"native"` → `ionq.native.v1`). Compiled-circuit artifacts come from the v0.4 compiler stack, which is rolling out as the prod default — until then, pass `service_version="v0.4"` as above, otherwise no compiled circuit is published and `compiled_circuit()` raises listing the available formats (`none`). Dry-run jobs produce no measurement results, so calling `job.result()` on one raises `IonQJobError` directing you to `compiled_circuit(...)`.

### Per-shot memory (`memory`)

QPU and noisy-simulator jobs can return per-shot measurement outcomes. Pass `memory=True` on `backend.run(...)` to opt in (the default is `False`, matching the `qiskit` and `qiskit-aer` `backend.run` convention), then call `job.get_memory()` to retrieve the per-shot bitstrings:

```python
job = backend.run(qc, shots=1000, memory=True)
memory = job.get_memory()       # ['11', '00', '11', '00', ...]
```

The ideal simulator does not produce per-shot data; calling `get_memory()` on a job submitted without `memory=True` raises `IonQBackendError`.

### Mid-circuit measurements

The IonQ provider supports mid-circuit measurements, qubit reuse, and mid-circuit `reset`. Results are reported per declared classical register, like Qiskit's usual register-split counts. Single-circuit only; the `debiasing` kwarg works here as well.

These run automatically as OpenQASM 3 (`ionq.qasm3.v1`); no extra flags needed. Today they execute on the simulator (mid-circuit-measurement QPU support is rolling out); other targets are rejected server-side. Register names that are OpenQASM 3 reserved words (`output`, `input`, `measure`, …) are rejected at submission.

Per-register results require sampling, so use a noisy simulator (`noise_model=...`) or a QPU; the ideal simulator returns only the aggregate distribution and `result()` raises there.

```python
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

qr = QuantumRegister(1, "q")
mid = ClassicalRegister(1, "mid")
result = ClassicalRegister(2, "result")
qc = QuantumCircuit(qr, mid, result)

qc.h(0)
qc.measure(0, mid[0])          # mid-circuit measurement
qc.x(0)
qc.measure(0, result[0])       # qubit reused after measurement
qc.x(0)
qc.measure(0, result[1])

job = backend.run(qc, shots=100, memory=True, noise_model="aria-1")
result = job.result()
result.get_counts()        # split across registers, e.g. {'01 0': 96, '10 1': 104}
result.get_memory()        # per-shot, e.g. ['01 0', '10 1', ...]
```

### Basis gates and transpilation

The IonQ provider provides access to the full IonQ Cloud backend, which includes its own transpilation and compilation pipeline. As such, IonQ provider backends have a broad set of "basis gates" that they will accept — effectively anything the IonQ API will accept. The current supported gates can be found [on our docs site](https://docs.ionq.com/#tag/quantum_programs).

If you have circuits that you'd like to run on IonQ backends that use other gates than this (`u` or `iswap` for example), you will either need to manually rewrite the circuit to only use the above list, or use the Qiskit transpiler, per the example below. Please note that not all circuits can be automatically transpiled.

If you'd like lower-level access—the ability to program in native gates and skip our compilation/transpilation pipeline—please reach out to your IonQ contact for further information.

```python
from qiskit import QuantumCircuit, transpile
from math import pi

qc2 = QuantumCircuit(1, 1)
qc2.u(pi, pi/2, pi/4, 0)
qc2.measure(0,0)
transpiled_circuit = transpile(qc2, simulator_backend)
```

## Contributing

If you'd like to contribute to the IonQ Provider, please take a look at the [contribution guidelines](CONTRIBUTING.md). This project adheres the Qiskit Community code of conduct. By participating, you are agreeing to uphold this code.

If you have an enhancement request or bug report, we encourage you to open an issue in [this repo's issues tracker](https://github.com/qiskit-partners/qiskit-ionq/issues). If you have a support question or general discussion topic, we recommend instead asking on the [Qiskit community slack](https://qiskit.slack.com/) (you can join using [this link](https://ibm.co/joinqiskitslack)) or the [Quantum Computing StackExchange](https://quantumcomputing.stackexchange.com/questions/tagged/qiskit).

## Running Tests

This package uses the [pytest](https://docs.pytest.org/en/stable/) test runner, and other packages
for mocking interfactions, reporting coverage, etc.
These can be installed with `pip install -r requirements-test.txt`.

To use pytest directly, just run:

```bash
pytest [pytest-args]
```

Alternatively, you may use the setuptools integration by running tests through `setup.py`, e.g.:

```bash
python setup.py test --addopts="[pytest-args]"
```

### Fixtures

Global pytest fixtures for the test suite can be found in the top-level [test/conftest.py](./test/conftest.py) file.

## SSL certificate issues

If you receive the error `SSLError(SSLCertVerificationError)` or otherwise are unable to connect succesfully, there are a few possible resolutions:

1. Try accessing <https://api.ionq.co/v0.4/health> in your browser; if this does not load, you need to contact an IT administrator about allowing IonQ API access.
2. `pip install pip_system_certs` instructs python to use the same certificate roots of trust as your local browser - install this if the first step succeeded but qiskit-ionq continues to have issues.
3. You can debug further by running `res = requests.get('https://api.ionq.co/v0.4/health', timeout=30)` and inspecting `res`, you should receive a 200 response with the content `{"status": "pass"}`. If you see a corporate or ISP login page, you will need to contact a local IT administrator to debug further.

## Documentation

To build the API reference and quickstart docs, run:

```bash
pip install -r requirements-docs.txt
make html
open build/html/index.html
```

## License

[Apache License 2.0].

The IonQ logo and Q mark are copyright IonQ, Inc. All rights reserved.

[ionq]: https://www.ionq.com/
[apache license 2.0]: https://github.com/qiskit-partners/qiskit-ionq/blob/master/LICENSE.txt
