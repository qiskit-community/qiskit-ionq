====================================
Submitting circuits to IonQ backends
====================================

Once a a backend has been specified, it may be used to submit circuits.
For example, running a Bell State:

.. code-block:: python

   from qiskit import QuantumCircuit

   # Create a basic Bell State circuit:
   qc = QuantumCircuit(2, 2)
   qc.h(0)
   qc.cx(0, 1)
   qc.measure([0, 1], [0, 1])

   # Run the circuit on IonQ's platform:
   job = simulator_backend.run(qc)

   # Print the results:
   print(job.get_counts())


Basis gates and transpilation
=============================

The IonQ provider provides access to the full IonQ Cloud backend, which includes
its own transpilation and compilation pipeline. As such, IonQ provider backends
have a broad set of "basis gates" that they will accept — effectively anything
the IonQ API will accept:

.. jupyter-execute::
   :hide-code:

   from qiskit_ionq import IonQProvider
   ionq = IonQProvider('TOKEN')
   print(ionq.backends.ionq_qpu.configuration().basis_gates)


If you have circuits that you'd like to run on IonQ backends that use other gates
than this (``u`` or ``iswap`` for example), you will either need to manually rewrite
the circuit to only use the above list, or use the Qiskit transpiler, per the
example below. Please note that not all circuits can be automatically transpiled.

If you'd like lower-level access—the ability to program in native gates and skip
our compilation/transpilation pipeline—please reach out to your IonQ contact for
further information.
