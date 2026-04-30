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

from __future__ import annotations

from typing import Literal

import json.decoder as jd

import requests

from qiskit.exceptions import QiskitError
from qiskit.providers import JobError, JobTimeoutError


class IonQError(QiskitError):
    """Base class for errors raised by an IonQProvider."""

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r})"

    def __repr__(self) -> str:
        return repr(str(self))


class IonQCredentialsError(IonQError):
    """Errors generated from bad credentials or config"""


class IonQClientError(IonQError):
    """Errors that arise from unexpected behavior while using IonQClient."""


class IonQRetriableError(IonQError):
    """Errors that do not indicate a failure related to the request, and can be retried."""

    def __init__(self, cause):
        self._cause = cause
        super().__init__(getattr(cause, "message", "Unknown error"))


# pylint: disable=no-member


# https://support.cloudflare.com/hc/en-us/articles/115003014512-4xx-Client-Error
# "Cloudflare will generate and serve a 409 response for a Error 1001: DNS Resolution Error."
# We may want to condition on the body as well, to allow for some GET requests to return 409 in
# the future.
_RETRIABLE_FOR_GETS = {requests.codes.conflict}
# Retriable regardless of the source
# Handle 52x responses from cloudflare.
# See https://support.cloudflare.com/hc/en-us/articles/115003011431/
_RETRIABLE_STATUS_CODES = {
    requests.codes.too_many_requests,  # 429
    requests.codes.bad_gateway,  # 502
    requests.codes.service_unavailable,  # 503
    requests.codes.gateway_timeout,  # 504
    *list(range(520, 530)),  # 520–529 (Cloudflare)
}
# pylint: enable=no-member


def _is_retriable(method, code):
    return code in _RETRIABLE_STATUS_CODES or (
        method == "GET" and code in _RETRIABLE_FOR_GETS
    )


class IonQAPIError(IonQError):
    """Base exception for fatal API errors.

    Attributes:
        status_code(int): An HTTP response status code.
        error_type(str): An error type string from the IonQ REST API.
    """

    def __init__(self, message, status_code, headers, body, error_type):  # pylint: disable=too-many-positional-arguments
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers
        self.body = body
        self.error_type = error_type
        # Populated by :meth:`from_ionq_core` when the source exception is an
        # ``ionq_core.exceptions.APIError``; ``None`` for the legacy
        # ``requests``-based path. Transitional: removed alongside the rest
        # of this exception hierarchy when callers migrate to
        # ``ionq_core.exceptions.APIError``.
        self.request_id: str | None = None

    @classmethod
    def raise_for_status(cls, response) -> IonQAPIError | None:
        """Raise an instance of the exception class from an API response object if needed.
        Args:
            response (:class:`Response <requests.Response>`): An IonQ REST API response.

        Raises:
            IonQAPIError: instance of `cls` with error detail from `response`.
            IonQRetriableError:  instance of `cls` with error detail from `response`."""
        status_code = response.status_code
        if status_code < 400:
            # No error, return None
            return None
        res = cls.from_response(response)
        if _is_retriable(response.request.method, status_code):
            raise IonQRetriableError(res)
        raise res

    @classmethod
    def from_response(cls, response: requests.Response) -> IonQAPIError:
        """Raise an instance of the exception class from an API response object.

        Args:
            response (:class:`Response <requests.Response>`): An IonQ REST API response.

        Returns:
            IonQAPIError: instance of `cls` with error detail from `response`.

        """
        # TODO: Pending API changes will cleanup this error logic:
        status_code = response.status_code
        headers = response.headers
        body = response.text
        try:
            response_json = response.json()
        except jd.JSONDecodeError:
            response_json = {"invalid_json": response.text}
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
        return cls(message, status_code, headers, body, error_type)

    @classmethod
    def from_ionq_core(cls, exc) -> IonQAPIError:
        """Translate an ``ionq_core.exceptions.APIError`` into an ``IonQAPIError``.

        ``ionq-core`` raises a typed exception hierarchy keyed on HTTP status
        (``BadRequestError``/``AuthenticationError``/``NotFoundError``/
        ``RateLimitError``/``ServerError``/etc.). Code paths that route HTTP
        through ``ionq-core`` can call this classmethod to surface those
        failures as an ``IonQAPIError`` with the same ``message``/``status_code``/
        ``headers``/``body``/``error_type`` shape every other ``qiskit-ionq``
        call already produces - so ``except IonQAPIError`` keeps catching.

        TODO(ionq-core): transitional bridge. Remove when callers migrate to
        ``ionq_core.exceptions.APIError`` directly and the rest of
        ``qiskit_ionq.exceptions`` is dismantled.
        """
        # Re-parse the raw body through the same three-shape parser
        # ``from_response`` uses, so legacy and ionq-core code paths produce
        # identical error_type/message attributes for the same HTTP body.
        raw_body = getattr(exc, "body", None) or ""
        try:
            response_json = jd.JSONDecoder().decode(raw_body) if raw_body else {}
        except jd.JSONDecodeError:
            response_json = {"invalid_json": raw_body}
        error_type = "internal_error"
        message = getattr(exc, "message", None) or "No error details provided."
        if "code" in response_json:
            message = response_json.get("message") or message
        elif "statusCode" in response_json:
            message = response_json.get("message") or message
            error_type = response_json.get("error") or error_type
        elif "error" in response_json:
            error_data = response_json.get("error") or {}
            message = error_data.get("message") or message
            error_type = error_data.get("type") or error_type
        instance = cls(
            message,
            getattr(exc, "status_code", 0),
            {},  # ionq-core APIError doesn't expose headers
            raw_body,
            error_type,
        )
        # Surface the IonQ Cloud request id for support escalation.
        instance.request_id = getattr(exc, "request_id", None)
        return instance

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r},"
            f"status_code={self.status_code!r},"
            f"headers={self.headers},"
            f"body={self.body},"
            f"error_type={self.error_type!r})"
        )

    def __reduce__(self):
        return (
            self.__class__,
            (self.message, self.status_code, self.headers, self.body, self.error_type),
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

    def __init__(self, gate_name: str, gateset: Literal["qis", "native"]):
        self.gate_name = gate_name
        self.gateset = gateset
        super().__init__(
            (
                f"gate '{gate_name}' is not supported on the '{gateset}' IonQ backends. "
                "Please use the qiskit.transpile method, manually rewrite to remove the gate, "
                "or change the gateset selection as appropriate."
            )
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(gate_name={self.gate_name!r}, gateset={self.gateset!r})"


class IonQMidCircuitMeasurementError(IonQError, JobError):
    """Errors generated from attempting mid-circuit measurement, which is not supported.
    Measurement must come after all instructions.

    Attributes:
        qubit_index: The qubit index to be measured mid-circuit
    """

    def __init__(self, qubit_index: int, gate_name: str):
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


class IonQPauliExponentialError(IonQError):
    """Errors generated from improper usage of Pauli exponentials."""


class IonQTranspileLevelWarning(UserWarning):
    """Transpiler optimization level may degrade IonQ QIS results."""


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
    "IonQPauliExponentialError",
    "IonQTranspileLevelWarning",
]
