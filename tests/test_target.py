from __future__ import annotations

from unittest.mock import patch

from qiskit_ionq._target import build_target


class TestSimulatorTarget:
    def test_qis_gateset(self, client, simulator_backend_info):
        target = build_target(simulator_backend_info, client, "qis")
        assert target.num_qubits == 29
        assert {"h", "cx", "rx", "measure"} <= target.operation_names
        assert None in target["h"]  # simulator uses global (None) qubit properties

    def test_native_gateset(self, client, simulator_backend_info):
        target = build_target(simulator_backend_info, client, "native")
        assert {"gpi", "gpi2", "ms", "zz", "measure"} <= target.operation_names
        assert "h" not in target.operation_names


class TestQPUTarget:
    @patch("qiskit_ionq._target.get_characterization")
    def test_per_qubit_properties(
        self, mock_get_char, client, qpu_backend_info, characterization
    ):
        mock_get_char.sync.return_value = characterization
        target = build_target(qpu_backend_info, client, "qis")
        assert target.num_qubits == 25
        assert (0,) in target["h"] and target["h"][(0,)].error is not None
        assert (0, 1) in target["cx"] and (1, 0) in target["cx"]

    @patch("qiskit_ionq._target.get_characterization")
    def test_no_characterization(self, mock_get_char, client, qpu_backend_info):
        mock_get_char.sync.return_value = None
        target = build_target(qpu_backend_info, client, "qis")
        assert target.num_qubits == 25 and (0,) in target["h"]
