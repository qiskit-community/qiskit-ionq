from __future__ import annotations

from unittest.mock import patch

import pytest
from ionq_core.models.backend import Backend

from qiskit_ionq.provider import IonQProvider

_COMMON = dict(degraded=False, last_updated="2026-01-01T00:00:00Z")


def _make_backends():
    return [
        Backend(backend="simulator", status="available", qubits=29, average_queue_time=0.0, **_COMMON),
        Backend(backend="qpu.aria-1", status="available", qubits=25, average_queue_time=120.0, **_COMMON),
        Backend(backend="qpu.forte-1", status="unavailable", qubits=36, average_queue_time=0.0, **_COMMON),
    ]


class TestIonQProvider:
    @patch("qiskit_ionq.provider.get_backends")
    def test_backends_returns_all(self, mock):
        mock.sync.return_value = _make_backends()
        assert len(IonQProvider(api_key="test-key").backends()) == 3

    @patch("qiskit_ionq.provider.get_backends")
    def test_backends_filter_by_name(self, mock):
        mock.sync.return_value = _make_backends()
        backends = IonQProvider(api_key="test-key").backends(name="simulator")
        assert len(backends) == 1 and backends[0].name == "simulator"

    @patch("qiskit_ionq.provider.get_backends")
    def test_get_backend(self, mock):
        mock.sync.return_value = _make_backends()
        assert IonQProvider(api_key="test-key").get_backend("qpu.aria-1").name == "qpu.aria-1"

    @patch("qiskit_ionq.provider.get_backends")
    def test_get_backend_not_found(self, mock):
        mock.sync.return_value = _make_backends()
        with pytest.raises(ValueError, match="not found"):
            IonQProvider(api_key="test-key").get_backend("nonexistent")
