import numpy as np
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit import QuantumRegister, QuantumCircuit, Parameter
from qiskit.circuit.library import HGate, CXGate, RXGate, RYGate, RZGate,  SGate, SdgGate, SXGate, SXdgGate, TGate, TdgGate, UGate, U1Gate, U2Gate, U3Gate, XGate, YGate, ZGate, IGate, PhaseGate, GlobalPhaseGate
from qiskit_ionq import GPIGate, GPI2Gate, MSGate

# TODO: This code is not yet complete, more equivalence rules will be added: 
# see the exhaustive list of gates here: https://github.com/Qiskit/qiskit/blob/main/qiskit/circuit/library/standard_gates/equivalence_library.py

q = QuantumRegister(1, "q")
h_gate = QuantumCircuit(q)
h_gate.append(GPI2Gate(1), [0])
h_gate.append(GPIGate(0.375), [0])
h_gate.append(GPI2Gate(0.5), [0])
SessionEquivalenceLibrary.add_equivalence(HGate(), h_gate)

q = QuantumRegister(1, "q")
lambda_param = Parameter("lambda_param")
phase_gate = QuantumCircuit(q)
phase_gate.append(GPI2Gate(0.5), [0])
phase_gate.append(GPIGate(-lambda_param/(4*np.pi)), [0])
phase_gate.append(GPI2Gate(0.5 - lambda_param/(2*np.pi)), [0])
SessionEquivalenceLibrary.add_equivalence(PhaseGate(lambda_param), phase_gate)

q = QuantumRegister(1, "q")
theta_param = Parameter("theta_param")
rx_gate = QuantumCircuit(q)
rx_gate.append(GPI2Gate(0.25), [0])
# MINUS, IS THIS RIGHT?
rx_gate.append(GPIGate(-theta_param/(4*np.pi) + 1/4), [0])
rx_gate.append(GPI2Gate(0.25), [0])
SessionEquivalenceLibrary.add_equivalence(RXGate(theta_param), rx_gate)

q = QuantumRegister(1, "q")
theta_param = Parameter("theta_param")
ry_gate = QuantumCircuit(q)
ry_gate.append(GPI2Gate(0.5), [0])
ry_gate.append(GPIGate(theta_param/(4*np.pi)), [0])
ry_gate.append(GPI2Gate(0.5), [0])
SessionEquivalenceLibrary.add_equivalence(RYGate(theta_param), ry_gate)

q = QuantumRegister(1, "q")
theta_param = Parameter("theta_param")
rz_gate = QuantumCircuit(q)
rz_gate.append(GPIGate(theta_param/(4*np.pi)), [0])
rz_gate.append(GPIGate(0), [0])
SessionEquivalenceLibrary.add_equivalence(RZGate(theta_param), rz_gate)

q = QuantumRegister(1, "q")
s_gate = QuantumCircuit(q)
s_gate.append(GPI2Gate(0.75), [0])
s_gate.append(GPIGate(0.125), [0])
s_gate.append(GPI2Gate(0.5), [0])
SessionEquivalenceLibrary.add_equivalence(SGate(), s_gate)

q = QuantumRegister(1, "q")
s_dag_gate = QuantumCircuit(q)
s_dag_gate.append(GPI2Gate(1.25), [0])
s_dag_gate.append(GPIGate(0.375), [0])
s_dag_gate.append(GPI2Gate(0.5), [0])
SessionEquivalenceLibrary.add_equivalence(SdgGate(), s_dag_gate)

q = QuantumRegister(1, "q")
sx_gate = QuantumCircuit(q)
sx_gate.append(GPI2Gate(0.75), [0])
sx_gate.append(GPIGate(0.375), [0])
sx_gate.append(GPI2Gate(0.75), [0])
SessionEquivalenceLibrary.add_equivalence(SXGate(), sx_gate)

q = QuantumRegister(1, "q")
sx_dag_gate = QuantumCircuit(q)
sx_dag_gate.append(GPI2Gate(1.25), [0])
sx_dag_gate.append(GPIGate(0.375), [0])
sx_dag_gate.append(GPI2Gate(0.25), [0])
SessionEquivalenceLibrary.add_equivalence(SXdgGate(), sx_dag_gate)

q = QuantumRegister(1, "q")
t_gate = QuantumCircuit(q)
t_gate.append(GPI2Gate(0.625), [0])
t_gate.append(GPIGate(0.0625), [0])
t_gate.append(GPI2Gate(0.5), [0])
SessionEquivalenceLibrary.add_equivalence(TGate(), t_gate)

q = QuantumRegister(1, "q")
t_dag_gate = QuantumCircuit(q)
t_dag_gate.append(GPI2Gate(1.375), [0])
t_dag_gate.append(GPIGate(0.43750000000000006), [0])
t_dag_gate.append(GPI2Gate(0.5), [0])
SessionEquivalenceLibrary.add_equivalence(TdgGate(), t_dag_gate)

# q = QuantumRegister(1, "q")
# theta_param = Parameter("theta_param")
# phi_param = Parameter("phi_param")
# lambda_param = Parameter("lambda_param")
# u_gate = QuantumCircuit(q)
# u_gate.append(GPI2Gate(0.5 + phi_param/2), [0])
# u_gate.append(GPIGate(theta_param/4 + phi_param/4 - lambda_param/4), [0])
# u_gate.append(GPI2Gate(0.5 - lambda_param/2), [0])
# SessionEquivalenceLibrary.add_equivalence(UGate(theta_param, phi_param, lambda_param), u_gate)

# q = QuantumRegister(1, "q")
# theta_param = Parameter("theta_param")
# phi_param = Parameter("phi_param")
# lambda_param = Parameter("lambda_param")
# u3_gate = QuantumCircuit(q)
# u3_gate.append(GPI2Gate(0.5 + phi_param/2), [0])
# u3_gate.append(GPIGate(theta_param/4 + phi_param/4 - lambda_param/4), [0])
# u3_gate.append(GPI2Gate(0.5 - lambda_param/2), [0])
# SessionEquivalenceLibrary.add_equivalence(U3Gate(theta_param, phi_param, lambda_param), u3_gate)

q = QuantumRegister(1, "q")
lambda_param = Parameter("lambda_param")
u1_gate = QuantumCircuit(q)
u1_gate.append(GPI2Gate(0.5), [0])
u1_gate.append(GPIGate(-lambda_param/4), [0])
u1_gate.append(GPI2Gate(0.5 - lambda_param/2), [0])
SessionEquivalenceLibrary.add_equivalence(U1Gate(lambda_param), u1_gate)

# q = QuantumRegister(1, "q")
# phi_param = Parameter("phi_param")
# lambda_param = Parameter("lambda_param")
# u2_gate = QuantumCircuit(q)
# u2_gate.append(GPI2Gate(0.5 + phi_param/2), [0])
# u2_gate.append(GPIGate(phi_param/4 - lambda_param/4), [0])
# u2_gate.append(GPI2Gate(0.5 - lambda_param/2), [0])
# SessionEquivalenceLibrary.add_equivalence(U2Gate(phi_param, lambda_param), u2_gate)

q = QuantumRegister(1, "q") 
x_gate = QuantumCircuit(q)
x_gate.append(GPIGate(0), [0])
SessionEquivalenceLibrary.add_equivalence(XGate(), x_gate)

# q = QuantumRegister(2, "q")
# cx_gate = QuantumCircuit(q)
# cx_gate.append(GPI2Gate(1/4), [0])
# cx_gate.append(MSGate(0, 0), [0, 1])
# cx_gate.append(GPI2Gate(1/2), [0])
# cx_gate.append(GPI2Gate(1/2), [1])
# cx_gate.append(GPI2Gate(-1/4), [0])
# SessionEquivalenceLibrary.add_equivalence(CXGate(), cx_gate)

q = QuantumRegister(1, "q") 
y_gate = QuantumCircuit(q)
y_gate.append(GPIGate(0.25), [0])
SessionEquivalenceLibrary.add_equivalence(YGate(), y_gate)

q = QuantumRegister(1, "q") 
z_gate = QuantumCircuit(q)
z_gate.append(GPIGate(0.25), [0])
z_gate.append(GPIGate(0), [0])
SessionEquivalenceLibrary.add_equivalence(ZGate(), z_gate)

q = QuantumRegister(1, "q") 
i_gate = QuantumCircuit(q)
i_gate.append(GPI2Gate(0.5), [0])
i_gate.append(GPIGate(0), [0])
i_gate.append(GPI2Gate(0.5), [0])
SessionEquivalenceLibrary.add_equivalence(IGate(), i_gate)

# Below are the rules needed for Aer simulator to simulate circuits containing IonQ native gates

#TODO: verify!!!
q = QuantumRegister(1, "q")
phi_param = Parameter("phi_param")
gpi_gate = QuantumCircuit(q)
gpi_gate.append(RZGate(4 * phi_param * np.pi), [0])
gpi_gate.append(RXGate(np.pi), [0])
SessionEquivalenceLibrary.add_equivalence(GPIGate(phi_param), gpi_gate)

q = QuantumRegister(1, "q")
phi_param = Parameter("phi_param")
gpi2_gate = QuantumCircuit(q)
gpi2_gate.append(RZGate(2 * phi_param * np.pi), [0])
gpi2_gate.append(RXGate(np.pi/2), [0])
gpi2_gate.append(RZGate(-2 * phi_param * np.pi), [0])
SessionEquivalenceLibrary.add_equivalence(GPI2Gate(phi_param), gpi2_gate)

# CHGate,,
# CPhaseGate,
# RGate,
# RCCXGate,
# CRXGate,
# RXXGate,
# CRYGate,
# CRZGate,
# RZZGate,
# RZXGate,
# CSGate,
# CSdgGate,
# SwapGate
# CSwapGate,
# iSwapGate,
# CSXGate,
# DCXGate,
# CUGate,
# CU1Gate,
# U3Gate,
# CU3Gate,
# CCXGate,
# CYGate,
# RYYGate,
# ECRGate,
# CZGate, TODO
# CCZGate, TODO
# XXPlusYYGate,
# XXMinusYYGate,






