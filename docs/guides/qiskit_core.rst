=====================
Qiskit core workflows
=====================

The IonQ provider can be used with Qiskit's core circuit, transpiler, and
``quantum_info`` tools before a job is submitted to IonQ Cloud. This is useful
when you want to validate a circuit locally, inspect the operations that Qiskit
will send to an IonQ backend, or request an IonQ compilation-only dry run.

The repository includes a runnable script at
``example/qiskit_core_workflow.py``. By default it does not submit to IonQ Cloud:

.. code-block:: bash

   python example/qiskit_core_workflow.py

The script demonstrates three steps:

* build an unmeasured GHZ circuit with :class:`qiskit.QuantumCircuit`;
* use :class:`qiskit.quantum_info.Statevector` and
  :class:`qiskit.quantum_info.SparsePauliOp` to compute a local ``<ZZI...>``
  expectation value;
* add measurements and call :func:`qiskit.transpile` against an IonQ backend
  target.

The core workflow is:

.. code-block:: python

   from qiskit import QuantumCircuit, transpile
   from qiskit.quantum_info import SparsePauliOp, Statevector
   from qiskit_ionq import IonQProvider

   provider = IonQProvider()
   backend = provider.get_backend("ionq_simulator")

   state_prep = QuantumCircuit(3)
   state_prep.h(0)
   state_prep.cx(0, 1)
   state_prep.cx(0, 2)

   observable = SparsePauliOp.from_list([("ZZI", 1.0)])
   expectation = Statevector.from_instruction(state_prep).expectation_value(
       observable
   )

   measured = QuantumCircuit(3, 3)
   measured.compose(state_prep, inplace=True)
   measured.measure(range(3), range(3))

   transpiled = transpile(measured, backend=backend, optimization_level=1)

Use ``--submit`` only when you intentionally want to create an IonQ Cloud job.
For compilation without shot execution, combine ``--submit`` with ``--dry-run``:

.. code-block:: bash

   IONQ_API_TOKEN=your-token python example/qiskit_core_workflow.py \
       --backend ionq_qpu.forte-1 --submit --dry-run

For a dry-run job, the example prints the compiled OpenQASM 3 output returned by
``job.compiled_circuit(lang="qasm3")``. Without ``--dry-run``, it prints the
regular result counts from ``job.result().get_counts()``.
