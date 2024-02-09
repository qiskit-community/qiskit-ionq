from math import pi
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import HGate, XGate, YGate
from qiskit_ionq import GPIGate, GPI2Gate, MSGate, ZZGate

# TODO: This code is not yet complete, more equivalence rules will be added: 
# see the exhaustive list of gates here: https://github.com/Qiskit/qiskit/blob/main/qiskit/circuit/library/standard_gates/equivalence_library.py

# TODO: I do not know yet how to make the difference between situations where the user is targeting Aria or Forte IonQ devices.

q = QuantumRegister(1, "q") 
x_gate = QuantumCircuit(q)
x_gate.append(GPIGate(0), [0])
SessionEquivalenceLibrary.add_equivalence(XGate(), x_gate)

q = QuantumRegister(1, "q") 
y_gate = QuantumCircuit(q)
y_gate.append(GPIGate(pi / 2), [0])
SessionEquivalenceLibrary.add_equivalence(YGate(), y_gate)

# q = QuantumRegister(1, "q")
# h_gate = QuantumCircuit(q)
# h_gate.append(GPIGate(1.25 * pi), [0])
# h_gate.append(GPI2Gate(0), [0])
# h_gate.append(VirtualZ(1.5 * pi), [0])
# SessionEquivalenceLibrary.add_equivalence(HGate(), h_gate)







