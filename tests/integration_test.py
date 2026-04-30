"""Comprehensive integration tests against the real IonQ API.

Run with: uv run python tests/integration_test.py
Requires IONQ_API_KEY environment variable.
"""

from __future__ import annotations

import math
import sys
import traceback

from qiskit.circuit import QuantumCircuit
from qiskit.providers import JobStatus

from qiskit_ionq import (
    IonQBackend,
    IonQProvider,
    IonQSampler,
    IonQSamplerJob,
    IonQSession,
)

passed = 0
failed = 0
errors = []


def test(name):
    def decorator(fn):
        global passed, failed
        print(f"\n{'=' * 60}")
        print(f"TEST: {name}")
        print(f"{'=' * 60}")
        try:
            fn()
            passed += 1
            print("  PASSED")
        except Exception as e:
            failed += 1
            tb = traceback.format_exc()
            errors.append((name, tb))
            print(f"  FAILED: {e}")
            print(tb)

    return decorator


# 1. Provider basics
print("\n" + "#" * 60)
print("# PROVIDER TESTS")
print("#" * 60)

provider = IonQProvider()


@test("Provider: list all backends")
def _():
    backends = provider.backends()
    assert len(backends) > 0, "No backends returned"
    print(f"  Found {len(backends)} backends:")
    for b in backends:
        print(f"    - {b.name} ({b._backend_info.qubits}q, status={b._backend_info.status})")


@test("Provider: get simulator backend")
def _():
    backend = provider.get_backend("simulator")
    assert backend.name == "simulator"
    assert isinstance(backend, IonQBackend)
    print(f"  Simulator: {backend.num_qubits} qubits")


@test("Provider: get_backend raises for nonexistent")
def _():
    try:
        provider.get_backend("nonexistent-backend-xyz")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        print(f"  Correctly raised: {e}")


@test("Provider: filter backends by name")
def _():
    results = provider.backends(name="simulator")
    assert len(results) == 1
    assert results[0].name == "simulator"
    print(f"  Filtered to 1 backend: {results[0].name}")


# 2. Backend properties
print("\n" + "#" * 60)
print("# BACKEND TESTS")
print("#" * 60)

sim = provider.get_backend("simulator")


@test("Backend: target has QIS gates")
def _():
    target = sim.target
    expected_gates = [
        "h",
        "x",
        "y",
        "z",
        "cx",
        "rx",
        "ry",
        "rz",
        "s",
        "sdg",
        "t",
        "tdg",
        "sx",
        "sxdg",
        "swap",
        "rxx",
        "ryy",
        "rzz",
        "measure",
    ]
    for gate in expected_gates:
        assert gate in target.operation_names, f"Missing gate: {gate}"
    print(f"  All {len(expected_gates)} expected QIS gates present in target")
    print(f"  Total operations: {len(target.operation_names)}")


@test("Backend: num_qubits")
def _():
    assert sim.num_qubits is not None
    assert sim.num_qubits > 0
    print(f"  Simulator has {sim.num_qubits} qubits")


@test("Backend: max_circuits is None")
def _():
    assert sim.max_circuits is None
    print("  max_circuits = None (unlimited)")


@test("Backend: default options")
def _():
    assert sim.options.shots == 1024
    print(f"  Default shots: {sim.options.shots}")


@test("Backend: gateset property")
def _():
    assert sim.gateset == "qis"
    print(f"  Gateset: {sim.gateset}")


@test("Backend: native gateset target")
def _():
    native_sim = IonQBackend(
        provider=provider,
        backend_info=sim._backend_info,
        client=sim._client,
        gateset="native",
    )
    assert native_sim.gateset == "native"
    assert "gpi" in native_sim.target.operation_names
    assert "gpi2" in native_sim.target.operation_names
    assert "ms" in native_sim.target.operation_names
    assert "measure" in native_sim.target.operation_names
    assert "h" not in native_sim.target.operation_names
    print(f"  Native target operations: {sorted(native_sim.target.operation_names)}")


# 3. QPU backend (if available)
print("\n" + "#" * 60)
print("# QPU BACKEND TESTS (read-only)")
print("#" * 60)


@test("QPU: list QPU backends and check targets")
def _():
    all_backends = provider.backends()
    qpus = [b for b in all_backends if b.name.startswith("qpu.")]
    if not qpus:
        print("  No QPU backends available, skipping")
        return
    for qpu in qpus:
        print(f"  {qpu.name}: {qpu.num_qubits}q, status={qpu._backend_info.status}")
        target = qpu.target
        assert "h" in target.operation_names
        assert "cx" in target.operation_names
        assert "measure" in target.operation_names
        # QPU targets should have per-qubit properties (not global None)
        h_props = target["h"]
        if (0,) in h_props:
            error = h_props[(0,)].error
            print(f"    1Q error rate (qubit 0): {error}")
        else:
            print("    Using global properties (no characterization)")


# 4. Circuit submission via backend.run()
print("\n" + "#" * 60)
print("# CIRCUIT SUBMISSION TESTS (simulator)")
print("#" * 60)


@test("Backend.run(): Bell state circuit")
def _():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    job = sim.run(qc, shots=100)
    print(f"  Job ID: {job.job_id()}")

    status = job.status()
    print(f"  Initial status: {status}")
    assert isinstance(status, JobStatus)

    result = job.result(timeout=120)
    print(f"  Result success: {result.success}")
    counts = result.get_counts(0)
    print(f"  Counts: {counts}")
    total = sum(counts.values())
    assert total == 100, f"Expected 100 total shots, got {total}"
    # Bell state should only produce 00 and 11
    for bitstring in counts:
        assert bitstring in ("00", "11"), f"Unexpected state: {bitstring}"
    print("  Bell state verified: only |00> and |11> observed")


@test("Backend.run(): single qubit H gate")
def _():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.measure_all()

    job = sim.run(qc, shots=1000)
    result = job.result(timeout=120)
    counts = result.get_counts(0)
    print(f"  H gate counts: {counts}")
    assert sum(counts.values()) == 1000
    # Should see roughly 50/50 split
    assert "0" in counts and "1" in counts, "Expected both |0> and |1>"
    ratio = counts.get("0", 0) / 1000
    assert 0.3 < ratio < 0.7, f"H gate ratio too skewed: {ratio}"
    print(f"  |0> ratio: {ratio:.2f} (expected ~0.5)")


@test("Backend.run(): parametric RX gate")
def _():
    qc = QuantumCircuit(1)
    qc.rx(math.pi, 0)  # RX(pi) = X gate, should flip |0> to |1>
    qc.measure_all()

    job = sim.run(qc, shots=100)
    result = job.result(timeout=120)
    counts = result.get_counts(0)
    print(f"  RX(pi) counts: {counts}")
    assert counts.get("1", 0) == 100, f"Expected all |1>, got {counts}"
    print("  RX(pi) correctly produces |1> with 100% probability")


@test("Backend.run(): GHZ state (3 qubits)")
def _():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()

    job = sim.run(qc, shots=200)
    result = job.result(timeout=120)
    counts = result.get_counts(0)
    print(f"  GHZ counts: {counts}")
    assert sum(counts.values()) == 200
    for bitstring in counts:
        assert bitstring in ("000", "111"), f"Unexpected GHZ state: {bitstring}"
    print("  GHZ state verified: only |000> and |111> observed")


@test("Backend.run(): multi-gate circuit with rotations")
def _():
    qc = QuantumCircuit(2)
    qc.ry(math.pi / 3, 0)
    qc.rz(math.pi / 4, 1)
    qc.cx(0, 1)
    qc.rxx(math.pi / 6, 0, 1)
    qc.measure_all()

    job = sim.run(qc, shots=500)
    result = job.result(timeout=120)
    counts = result.get_counts(0)
    print(f"  Multi-gate counts: {counts}")
    assert sum(counts.values()) == 500
    print(f"  Multi-gate circuit executed successfully with {len(counts)} distinct outcomes")


@test("Backend.run(): custom shot count via options")
def _():
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.measure_all()

    job = sim.run(qc, shots=42)
    result = job.result(timeout=120)
    counts = result.get_counts(0)
    assert sum(counts.values()) == 42
    assert counts.get("1", 0) == 42
    print("  Custom shots=42 works correctly")


@test("Backend.run(): job status transitions")
def _():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.measure_all()

    job = sim.run(qc, shots=100)
    # Check initial status is valid
    status = job.status()
    assert status in (JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.DONE), f"Unexpected status: {status}"
    print(f"  Initial status: {status}")

    # Wait for completion
    job.result(timeout=120)
    final_status = job.status()
    assert final_status == JobStatus.DONE
    print(f"  Final status: {final_status}")


@test("Backend.run(): result caching")
def _():
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.measure_all()

    job = sim.run(qc, shots=50)
    r1 = job.result(timeout=120)
    r2 = job.result(timeout=120)
    assert r1 is r2, "Result should be cached"
    print("  Result caching verified (same object returned)")


# 5. Sampler tests
print("\n" + "#" * 60)
print("# SAMPLER TESTS (simulator)")
print("#" * 60)


@test("Sampler: single circuit PUB")
def _():
    sampler = IonQSampler(sim)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    job = sampler.run([qc], shots=100)
    assert isinstance(job, IonQSamplerJob)
    print(f"  Sampler job ID: {job.job_id()}")

    result = job.result()
    assert len(result) == 1
    pub_result = result[0]
    ba = pub_result.data.meas
    print(f"  BitArray: num_bits={ba.num_bits}, num_shots={ba.num_shots}")
    assert ba.num_bits == 2
    assert ba.num_shots == 100
    counts = ba.get_counts()
    print(f"  Sampler counts: {counts}")
    assert sum(counts.values()) == 100
    for bitstring in counts:
        assert bitstring in ("00", "11"), f"Unexpected Bell state: {bitstring}"
    print("  Sampler Bell state verified")


@test("Sampler: bare circuit as PUB (no tuple wrapping)")
def _():
    sampler = IonQSampler(sim)
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.measure_all()

    result = sampler.run([qc], shots=50).result()
    counts = result[0].data.meas.get_counts()
    assert counts.get("1", 0) == 50
    print(f"  Bare circuit PUB works: {counts}")


@test("Sampler: multi-circuit batching")
def _():
    sampler = IonQSampler(sim)

    qc1 = QuantumCircuit(2)
    qc1.h(0)
    qc1.cx(0, 1)
    qc1.measure_all()

    qc2 = QuantumCircuit(1)
    qc2.x(0)
    qc2.measure_all()

    qc3 = QuantumCircuit(1)
    qc3.h(0)
    qc3.measure_all()

    job = sampler.run([qc1, qc2, qc3], shots=100)
    result = job.result()
    assert len(result) == 3, f"Expected 3 results, got {len(result)}"

    # Circuit 1: Bell state
    counts1 = result[0].data.meas.get_counts()
    print(f"  Circuit 1 (Bell): {counts1}")
    for bs in counts1:
        assert bs in ("00", "11")

    # Circuit 2: X gate
    counts2 = result[1].data.meas.get_counts()
    print(f"  Circuit 2 (X): {counts2}")
    assert counts2.get("1", 0) == 100

    # Circuit 3: H gate
    counts3 = result[2].data.meas.get_counts()
    print(f"  Circuit 3 (H): {counts3}")
    assert sum(counts3.values()) == 100
    assert "0" in counts3 and "1" in counts3

    print("  Multi-circuit batching verified: 3 circuits, all correct")


@test("Sampler: result caching")
def _():
    sampler = IonQSampler(sim)
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.measure_all()

    job = sampler.run([qc], shots=50)
    r1 = job.result()
    r2 = job.result()
    assert r1 is r2, "SamplerJob result should be cached"
    print("  Sampler result caching verified")


# 6. Session tests
print("\n" + "#" * 60)
print("# SESSION TESTS (simulator)")
print("#" * 60)


@test("Session: context manager lifecycle")
def _():
    with IonQSession(sim) as session:
        assert session.session_id is not None
        print(f"  Session ID: {session.session_id}")

        sampler = session.sampler()
        assert sampler._session is session

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()

        result = sampler.run([qc], shots=100).result()
        counts = result[0].data.meas.get_counts()
        print(f"  Session sampler counts: {counts}")
        assert sum(counts.values()) == 100

    print("  Session closed successfully")


@test("Session: multiple jobs in same session")
def _():
    with IonQSession(sim) as session:
        sampler = session.sampler()
        print(f"  Session ID: {session.session_id}")

        qc1 = QuantumCircuit(1)
        qc1.x(0)
        qc1.measure_all()

        qc2 = QuantumCircuit(1)
        qc2.h(0)
        qc2.measure_all()

        r1 = sampler.run([qc1], shots=50).result()
        r2 = sampler.run([qc2], shots=50).result()

        c1 = r1[0].data.meas.get_counts()
        c2 = r2[0].data.meas.get_counts()
        print(f"  Job 1 (X): {c1}")
        print(f"  Job 2 (H): {c2}")
        assert c1.get("1", 0) == 50
        assert sum(c2.values()) == 50

    print("  Multiple jobs in session verified")


# 7. Transpilation tests
print("\n" + "#" * 60)
print("# TRANSPILATION TESTS (simulator)")
print("#" * 60)


@test("Transpilation: circuit with non-basis gates gets transpiled")
def _():
    # Use gates that aren't directly in IonQ's QIS set
    # The transpiler should decompose them
    qc = QuantumCircuit(2)
    qc.ch(0, 1)  # Controlled-H, not directly in IonQ gate set
    qc.measure_all()

    job = sim.run(qc, shots=200)
    result = job.result(timeout=120)
    counts = result.get_counts(0)
    print(f"  Controlled-H counts: {counts}")
    assert sum(counts.values()) == 200
    print("  Non-basis gate transpilation works")


@test("Transpilation: Toffoli gate decomposition")
def _():
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.x(1)
    qc.ccx(0, 1, 2)  # Toffoli
    qc.measure_all()

    job = sim.run(qc, shots=100)
    result = job.result(timeout=120)
    counts = result.get_counts(0)
    print(f"  Toffoli counts: {counts}")
    # |110> -> CCX flips qubit 2 -> |111>
    assert counts.get("111", 0) == 100, f"Expected all |111>, got {counts}"
    print("  Toffoli decomposition verified")


@test("Transpilation: SWAP gate")
def _():
    qc = QuantumCircuit(2)
    qc.x(0)  # qubit 0 = |1>, qubit 1 = |0>
    qc.swap(0, 1)  # swap -> qubit 0 = |0>, qubit 1 = |1>
    qc.measure_all()

    job = sim.run(qc, shots=100)
    result = job.result(timeout=120)
    counts = result.get_counts(0)
    print(f"  SWAP counts: {counts}")
    # After swap: qubit 0=|0>, qubit 1=|1>
    # Qiskit uses little-endian: rightmost bit is qubit 0 -> "01"
    assert counts.get("01", 0) == 100, f"Expected all |01>, got {counts}"
    print("  SWAP gate verified")


# 8. Edge cases
print("\n" + "#" * 60)
print("# EDGE CASE TESTS")
print("#" * 60)


@test("Edge: single qubit identity circuit")
def _():
    qc = QuantumCircuit(1)
    qc.measure_all()

    job = sim.run(qc, shots=100)
    result = job.result(timeout=120)
    counts = result.get_counts(0)
    print(f"  Identity counts: {counts}")
    assert counts.get("0", 0) == 100
    print("  Identity circuit: all |0> as expected")


@test("Edge: maximum qubit circuit (small)")
def _():
    n = 10
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
    qc.measure_all()

    job = sim.run(qc, shots=100)
    result = job.result(timeout=120)
    counts = result.get_counts(0)
    assert sum(counts.values()) == 100
    print(f"  {n}-qubit H circuit: {len(counts)} unique outcomes out of 100 shots")


@test("Edge: circuit with barriers (should be ignored)")
def _():
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.barrier()
    qc.x(0)  # X twice = identity
    qc.measure_all()

    job = sim.run(qc, shots=100)
    result = job.result(timeout=120)
    counts = result.get_counts(0)
    assert counts.get("0", 0) == 100
    print("  Barriers correctly ignored: X.X = I, all |0>")


# Summary
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
print("=" * 60)

if errors:
    print("\nFAILED TESTS:")
    for name, tb in errors:
        print(f"\n  {name}:")
        for line in tb.strip().split("\n"):
            print(f"    {line}")

sys.exit(1 if failed > 0 else 0)
