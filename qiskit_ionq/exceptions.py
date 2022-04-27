# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Copyright 2020 IonQ, Inc. (www.ionq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exceptions for the IonQ Provider."""
import warnings

from qiskit.exceptions import QiskitError
from qiskit.providers import JobError, JobTimeoutError


class IonQError(QiskitError):
    """Base class for errors raised by an IonQProvider."""

    def __str__(self):
        return f"{self.__class__.__name__}({self.message!r})"

    def __repr__(self):
        return repr(str(self))


class IonQCredentialsError(IonQError):
    """Errors generated from bad credentials or config"""


class IonQClientError(IonQError):
    """Errors that arise from unexpected behavior while using IonQClient."""


class IonQAPIError(IonQError):
    """Base exception for fatal API errors.

    Attributes:
        status_code(int): An HTTP response status code.
        error_type(str): An error type string from the IonQ REST API.
    """

    @classmethod
    def from_response(cls, response):
        """Raise an instance of the exception class from an API response object.

        Args:
            response (:class:`Response <requests.Response>`): An IonQ REST API response.

        Returns:
            IonQAPIError: instance of `cls` with error detail from `response`.

        """
        # TODO: Pending API changes will cleanup this error logic:
        status_code = response.status_code
        response_json = response.json()

        # Defaults, if items cannot be extracted from the response.
        error_type = "internal_error"
        message = "No error details provided."
        if "code" in response_json:
            # { "code": <int>, "message": <str> }
            message = response_json.get("message") or message
        elif "statusCode" in response_json:
            # { "statusCode": <int>, "error": <str>, "message": <str> }
            message = response_json.get("message") or message
            error_type = response_json.get("error") or error_type
        elif "error" in response_json:
            # { "error": { "type": <str>, "message: <str> } }
            error_data = response_json.get("error")
            message = error_data.get("message") or message
            error_type = error_data.get("type") or error_type
        return cls(message, status_code, error_type)

    def __init__(self, message, status_code, error_type):
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(message)

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r},"
            f"status_code={self.status_code},"
            f"error_type={self.error_type!r})"
        )


class IonQBackendError(IonQError):
    """Errors generated from improper usage of IonQBackend objects."""


class IonQJobError(IonQError, JobError):
    """Errors generated from improper usage of IonQJob objects."""


class IonQJobFailureError(IonQError, JobError):
    """Errors generated from jobs that fail on the API side."""


class IonQJobStateError(IonQError, JobError):
    """Errors generated from attempting to do something to a job that its state would not allow"""


class IonQGateError(IonQError, JobError):
    """Errors generated from invalid gate defs

    Attributes:
        gate_name: The name of the gate which caused this error.
    """

    def __init__(self, gate_name, lang):
        self.gate_name = gate_name
        self.lang = lang
        super().__init__(
            (f"gate '{gate_name}' is not supported on the '{lang}' IonQ backends. "
              "Please use the qiskit.transpile method, manually rewrite to remove the gate, "
              "or change the language selection as appropriate."
            )
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(gate_name={self.gate_name!r}, lang={self.lang!r})"


class IonQMidCircuitMeasurementError(IonQError, JobError):
    """Errors generated from attempting mid-circuit measurement, which is not supported.
    Measurement must come after all instructions.

    Attributes:
        qubit_index: The qubit index to be measured mid-circuit
    """

    def __init__(self, qubit_index, gate_name):
        self.qubit_index = qubit_index
        self.gate_name = gate_name
        super().__init__(
            f"Attempting to put '{gate_name}' after a measurement on qubit {qubit_index}. "
            "Mid-circuit measurement is not supported."
        )

    def __str__(self):
        kwargs = f"qubit_index={self.qubit_index!r}, gate_name={self.gate_name!r}"
        return f"{self.__class__.__name__}({kwargs})"


class IonQJobTimeoutError(IonQError, JobTimeoutError):
    """Errors generated from job timeouts"""


class IonQMetadataStringError(IonQError, JobError):
    """Errors generated from metadata strings being too long

    Attributes:
        string_length: The length of the metadata string that's too long.
    """

    def __init__(self, string_length):
        self.string_length = string_length
        super().__init__(
            f"attempting to serialize circuit metadata, got length '{string_length}'. "
            "Must be under 400."
        )
        warnings.warn(
            """
            This error is a limitation of the IonQ API, not something you did wrong.
            To submit this circuit we recommend trying the following: shorten the circuit name,
            Use fewer qubit or cbit registers (i.e. combine them), or give them shorter names.
            Please file a ticket at support.ionq.co if you repeatedly see this error.
            """
        )

    def __str__(self):
        return f"{self.__class__.__name__}(string_length={self.string_length!r})"


__all__ = [
    "IonQError",
    "IonQCredentialsError",
    "IonQClientError",
    "IonQAPIError",
    "IonQBackendError",
    "IonQJobError",
    "IonQGateError",
    "IonQMidCircuitMeasurementError",
    "IonQJobTimeoutError",
    "IonQMetadataStringError",
]
