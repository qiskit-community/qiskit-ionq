import math

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGOpNode


class GPI2_Adjoint(TransformationPass):
    """GPI2 times GPI2 adjoint should cancel."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        nodes_to_remove = []

        for node in dag.topological_op_nodes():
            if isinstance(node, DAGOpNode) and node.op.name == "gpi2":
                successors = [
                    succ
                    for succ in dag.quantum_successors(node)
                    if isinstance(succ, DAGOpNode)
                ]
                for next_node in successors:
                    if next_node.op.name == "gpi2" and node.qargs == next_node.qargs:
                        phi1 = node.op.params[0]
                        phi2 = next_node.op.params[0]
                        if math.isclose((phi2 + 0.5) % 1, phi1 % 1) or math.isclose(
                            (phi1 + 0.5) % 1, phi2 % 1
                        ):
                            nodes_to_remove.extend([node, next_node])

        for node in nodes_to_remove:
            dag.remove_op_node(node)

        return dag


class CancelFourGPI2(TransformationPass):
    """Four GPI2 adjoint should cancel."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        nodes_to_remove = []
        gpi2_streak = []

        for node in dag.topological_op_nodes():
            if node.op.name == "gpi2" and math.isclose(
                node.op.params[0], 0.5
            ):  # GPI2(pi)
                if gpi2_streak and node.qargs != gpi2_streak[-1].qargs:
                    gpi2_streak = []
                gpi2_streak.append(node)

                if len(gpi2_streak) == 4:
                    # Found four consecutive gpi2 nodes on the same qubits
                    nodes_to_remove.extend(gpi2_streak)
                    gpi2_streak = []
            else:
                gpi2_streak = []

        for node in nodes_to_remove:
            dag.remove_op_node(node)

        return dag


class GPI_Adjoint(TransformationPass):
    """GPI times GPI adjoint should cancel."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        nodes_to_remove = []

        for node in dag.topological_op_nodes():
            if isinstance(node, DAGOpNode) and node.op.name == "gpi":
                successors = [
                    succ
                    for succ in dag.quantum_successors(node)
                    if isinstance(succ, DAGOpNode)
                ]
                for next_node in successors:
                    if next_node.op.name == "gpi" and node.qargs == next_node.qargs:
                        phi1 = node.op.params[0]
                        phi2 = next_node.op.params[0]
                        if math.isclose(phi1, phi2):
                            nodes_to_remove.extend([node, next_node])

        for node in nodes_to_remove:
            dag.remove_op_node(node)

        return dag


class CommuteGPI2MS(TransformationPass):
    """GPI2 * MS is replaced by MS * GPI2."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        nodes_to_remove = set()

        for node in dag.topological_op_nodes():
            if node in nodes_to_remove:
                continue

            if node.op.name == "gpi2" and math.isclose(
                node.op.params[0], 0.5
            ):  # GPI2(pi/2)
                successors = [
                    succ for succ in dag.successors(node) if isinstance(succ, DAGOpNode)
                ]
                for next_node in successors:
                    if (
                        next_node.op.name == "ms"
                        and math.isclose(next_node.op.params[0], 0)
                        and math.isclose(next_node.op.params[1], 0)
                        and math.isclose(next_node.op.params[2], 0.25)
                        and node.qargs[0] in next_node.qargs
                    ):

                        sub_dag = DAGCircuit()
                        for qreg in dag.qregs.values():
                            sub_dag.add_qreg(qreg)

                        # map the ops to the qubits in the sub-DAG
                        ms_qubits = [next_node.qargs[0], next_node.qargs[1]]
                        gpi2_qubit = [node.qargs[0]]

                        sub_dag.apply_operation_back(next_node.op, ms_qubits)
                        sub_dag.apply_operation_back(node.op, gpi2_qubit)

                        wire_mapping = {qubit: qubit for qubit in ms_qubits}
                        wire_mapping[node.qargs[0]] = node.qargs[0]

                        dag.substitute_node_with_dag(
                            next_node, sub_dag, wires=wire_mapping
                        )
                        nodes_to_remove.add(node)
                        break

        for node in nodes_to_remove:
            dag.remove_op_node(node)

        return dag


class GPI2TwiceIsGPITimesI(TransformationPass):
    """GPI2 times GPI2 is plus i times identity matrix."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:

        for node in dag.topological_op_nodes():
            if isinstance(node, DAGOpNode) and node.op.name == "gpi2":
                successors = [
                    succ
                    for succ in dag.quantum_successors(node)
                    if isinstance(succ, DAGOpNode)
                ]
                for next_node in successors:
                    if next_node.op.name == "gpi2" and node.qargs == next_node.qargs:
                        phi1 = node.op.params[0]
                        phi2 = next_node.op.params[0]
                        if math.isclose(phi1, phi2):
                            node.op.name = "gpi"
                            next_node.op.name = "+iId"
                            next_node.op.params = None

        return dag


class SimplifyIds(TransformationPass):
    """Simplify chains of identity matrices."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        nodes_to_remove = []

        for node in dag.topological_op_nodes():
            if isinstance(node, DAGOpNode) and node.op.name in ["+iId", "-iId", "-Id"]:
                successors = [
                    succ
                    for succ in dag.quantum_successors(node)
                    if isinstance(succ, DAGOpNode)
                ]
                for next_node in successors:
                    if node.qargs == next_node.qargs:
                        if node.op.name == "+iId" and next_node.op.name == "-iId":
                            nodes_to_remove.append(node)
                            nodes_to_remove.append(next_node)
                        elif node.op.name == "-iId" and next_node.op.name == "+iId":
                            nodes_to_remove.append(node)
                            nodes_to_remove.append(next_node)
                        elif node.op.name == "-iId" and next_node.op.name == "-iId":
                            nodes_to_remove.append(node)
                            next_node.op.name = "-Id"
                        elif node.op.name == "+iId" and next_node.op.name == "+iId":
                            nodes_to_remove.append(node)
                            next_node.op.name = "-Id"
                        elif node.op.name == "-Id" and next_node.op.name == "+iId":
                            nodes_to_remove.append(node)
                            next_node.op.name = "-iId"
                        elif node.op.name == "+iId" and next_node.op.name == "-Id":
                            nodes_to_remove.append(node)
                            next_node.op.name = "-iId"
                        elif node.op.name == "-Id" and next_node.op.name == "-iId":
                            nodes_to_remove.append(node)
                            next_node.op.name = "+iId"
                        elif node.op.name == "-iId" and next_node.op.name == "-Id":
                            nodes_to_remove.append(node)
                            next_node.op.name = "+iId"

        for node in nodes_to_remove:
            dag.remove_op_node(node)

        return dag


class CommuteGPisAndIds(TransformationPass):
    """Move identity matrices towards the end of the circuit when a GPI or GPI2 gate is found."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:

        for node in dag.topological_op_nodes():
            if isinstance(node, DAGOpNode) and node.op.name in ["+iId", "-iId", "-Id"]:
                successors = [
                    succ
                    for succ in dag.quantum_successors(node)
                    if isinstance(succ, DAGOpNode)
                ]
                for next_node in successors:
                    if (
                        next_node.op.name in ["gpi", "gpi2"]
                        and node.qargs == next_node.qargs
                    ):
                        node.op.name, next_node.op.name = (
                            next_node.op.name,
                            node.op.name,
                        )
                        node.op.params == next_node.op.params
                        next_node.op.params = None
        return dag


class CommuteMSAndIds(TransformationPass):
    """Move identity matrices towards the end of the circuit when an MS gate is found."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        nodes_to_remove = set()

        for node in dag.topological_op_nodes():
            if node in nodes_to_remove:
                continue

            if node.op.name in ["+iId", "-iId", "-Id"]:
                successors = [
                    succ for succ in dag.successors(node) if isinstance(succ, DAGOpNode)
                ]
                for next_node in successors:
                    if (
                        next_node.op.name == "ms"
                        and math.isclose(next_node.op.params[0], 0)
                        and math.isclose(next_node.op.params[1], 0)
                        and math.isclose(next_node.op.params[2], 0.25)
                        and node.qargs[0] in next_node.qargs
                    ):

                        sub_dag = DAGCircuit()
                        for qreg in dag.qregs.values():
                            sub_dag.add_qreg(qreg)

                        # map the ops to the qubits in the sub-DAG
                        ms_qubits = [next_node.qargs[0], next_node.qargs[1]]
                        id_qubit = [node.qargs[0]]

                        sub_dag.apply_operation_back(next_node.op, ms_qubits)
                        sub_dag.apply_operation_back(node.op, id_qubit)

                        wire_mapping = {qubit: qubit for qubit in ms_qubits}
                        wire_mapping[node.qargs[0]] = node.qargs[0]

                        dag.substitute_node_with_dag(
                            next_node, sub_dag, wires=wire_mapping
                        )
                        nodes_to_remove.add(node)
                        break

        for node in nodes_to_remove:
            dag.remove_op_node(node)

        return dag
