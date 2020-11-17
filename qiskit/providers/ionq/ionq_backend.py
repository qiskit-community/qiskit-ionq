import requests
from qiskit.providers import BaseBackend
from qiskit.providers.models import BackendConfiguration

from . import qobj_to_ionq
from .exceptions import *
from .ionq_job import IonQJob


class IonQBackend(BaseBackend):
    def __init__(self, name, configuration, provider, client):
        """Base class for interfacing with an IonQ backend"""
        self._name = name
        self._client = client
        super().__init__(
            configuration=BackendConfiguration.from_dict(configuration),
            provider=provider,
        )

    def run(self, qobj):
        """create and run a job on a backend from a qobj"""
        job = IonQJob(self, None, self._client, qobj=qobj)
        job.submit()
        return job

    def retrieve_job(self, job_id):
        """get a job from a specific backend, by job id."""
        return IonQJob(self, job_id, self._client)

    def retrieve_jobs(self, job_ids):
        """get a list of jobs from a specific backend, job id """

        return [IonQJob(self, job_id, self._client) for job_id in job_ids]

    def status(self):
        raise IonQBackendError(
            "IonQ backends don't currently expose a status endpoint. This method will be updated when they do"
        )

    def name(self):
        return self._name


def generate_backends(provider, client):
    "generate and return IonQ backends (simulator, qpu) from base backend class"
    simulator_config = {
        "backend_name": "ionq_simulator",
        "backend_version": "0.0.1",
        "simulator": True,
        "local": False,
        "coupling_map": None,
        "description": "IonQ simulator",
        "basis_gates": [
            "x",
            "y",
            "z",
            "rx",
            "ry",
            "rz",
            "h",
            "not",
            "cnot",
            "s",
            "si",
            "t",
            "ti",
            "v",
            "vi",
            "xx",
            "yy",
            "zz",
            "swap",
        ],
        "memory": False,
        "n_qubits": 29,
        "conditional": False,
        "max_shots": 10000,
        "max_experiments": 1,
        "open_pulse": False,
        "gates": [{"name": "TODO", "parameters": [], "qasm_def": "TODO"}],
    }
    qpu_config = {
        "backend_name": "ionq_qpu",
        "backend_version": "0.0.1",
        "simulator": False,
        "local": False,
        "coupling_map": None,
        "description": "IonQ QPU",
        "basis_gates": [
            "x",
            "y",
            "z",
            "rx",
            "ry",
            "rz",
            "h",
            "not",
            "cnot",
            "s",
            "si",
            "t",
            "ti",
            "v",
            "vi",
            "xx",
            "yy",
            "zz",
            "swap",
        ],
        "memory": False,
        "n_qubits": 11,
        "conditional": False,
        "max_shots": 10000,
        "max_experiments": 1,
        "open_pulse": False,
        "gates": [{"name": "TODO", "parameters": [], "qasm_def": "TODO"}],
    }
    IonQSimulator = IonQBackend("ionq_simulator", simulator_config, provider, client)
    IonQQPU = IonQBackend("ionq_qpu", qpu_config, provider, client)
    return [IonQSimulator, IonQQPU]
