"""Exceptions for the IonQ Provider."""

from qiskit.exceptions import QiskitError
from qiskit.providers import JobError, JobTimeoutError


class IonQError(QiskitError):
    """Base class for errors raised by the IonQ provider."""


class IonQCredentialsError(IonQError):
    """Errors generated from bad credentials or config"""


class IonQAPIError(IonQError):
    """Errors generated from API trouble

    Attributes:
    code -- API error code
    message -- API error message
    """

    @classmethod
    def from_response(cls, response):
        status_code = response.status_code
        response_json = response.json()
        error = response_json.get("error") or {}
        raise cls(status_code, error.get("message"), error.get("type"))

    def __init__(self, code, message, errorType):
        self.code = code
        self.message = message
        self.errorType = errorType
        super().__init__(self.message)

    def __str__(self):
        return "{} ({}): {}".format(self.code, self.errorType, self.message)


class IonQBackendError(IonQError):
    """Errors generated from backend issues"""


class IonQJobError(IonQError, JobError):
    """Errors generated from job issues"""


class IonQGateError(IonQError, JobError):
    """Errors generated from invalid gate defs

    Attributes:
    code -- API error code
    message -- API error message
    """

    def __init__(self, gate_name):
        self.gate_name = gate_name
        self.message = "gate not supported"
        super().__init__(self.message)

    def __str__(self):
        return "{}: {}".format(self.message, self.gate_name)


class IonQJobTimeoutError(IonQError, JobTimeoutError):
    """Errors generated from job issues"""


__all__ = [
    "IonQError",
    "IonQCredentialsError",
    "IonQAPIError",
    "IonQBackendError",
    "IonQJobError",
    "IonQGateError",
    "IonQJobTimeoutError",
]
