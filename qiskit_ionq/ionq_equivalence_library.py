import numpy as np
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit import QuantumRegister, QuantumCircuit, Parameter
from qiskit.circuit.library import CXGate, RXGate, RYGate, RZGate, UGate, XGate, GMS
from qiskit_ionq import GPIGate, GPI2Gate, MSGate

q = QuantumRegister(1, "q")
theta_param = Parameter("theta_param")
phi_param = Parameter("phi_param")
lambda_param = Parameter("lambda_param")
u3_gate = QuantumCircuit(q)
# this sequence can be compacted if virtual-z gates will be introduced
u3_gate.append(GPI2Gate(0.5 - lambda_param/(2*np.pi)), [0])
u3_gate.append(GPIGate(theta_param/(4*np.pi) + phi_param/(4*np.pi) - lambda_param/(4*np.pi)), [0])
u3_gate.append(GPI2Gate(0.5 + phi_param/(2*np.pi)), [0])
SessionEquivalenceLibrary.add_equivalence(UGate(theta_param, phi_param, lambda_param), u3_gate)

q = QuantumRegister(2, "q")
cx_gate = QuantumCircuit(q)
cx_gate.append(GPI2Gate(1/4), [0])
cx_gate.append(MSGate(0, 0), [0, 1])
cx_gate.append(GPI2Gate(1/2), [0])
cx_gate.append(GPI2Gate(1/2), [1])
cx_gate.append(GPI2Gate(-1/4), [0])
SessionEquivalenceLibrary.add_equivalence(CXGate(), cx_gate)

# Below are the rules needed for Aer simulator to simulate circuits containing IonQ native gates

q = QuantumRegister(1, "q")
phi_param = Parameter("phi_param")
gpi_gate = QuantumCircuit(q)
gpi_gate.append(XGate(), [0])
gpi_gate.append(RZGate(4 * phi_param * np.pi), [0])
SessionEquivalenceLibrary.add_equivalence(GPIGate(phi_param), gpi_gate)

q = QuantumRegister(1, "q")
phi_param = Parameter("phi_param")
gpi2_gate = QuantumCircuit(q)
gpi2_gate.append(RZGate(-2 * phi_param * np.pi), [0])
gpi2_gate.append(RXGate(np.pi/2), [0])
gpi2_gate.append(RZGate(2 * phi_param * np.pi), [0])
SessionEquivalenceLibrary.add_equivalence(GPI2Gate(phi_param), gpi2_gate)

q = QuantumRegister(2, "q")
phi0_param = Parameter("phi0_param")
phi1_param = Parameter("phi1_param")
theta_param = Parameter("theta_param")
ms_gate = QuantumCircuit(q)
ms_gate.h(0)
ms_gate.h(1)
ms_gate.append(CXGate(), [0, 1])
ms_gate.append(RZGate(2 * theta_param * np.pi), [1])
ms_gate.append(CXGate(), [0, 1])
ms_gate.h(0)
ms_gate.h(1)
ms_gate.append(RZGate(0 * phi0_param), [0])
ms_gate.append(RZGate(0 * phi1_param), [0])
SessionEquivalenceLibrary.add_equivalence(MSGate(phi0_param, phi1_param, theta_param), ms_gate)




