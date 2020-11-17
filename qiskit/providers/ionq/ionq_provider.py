"""Provider for interacting with IonQ backends"""

import logging

from qiskit.providers import BaseProvider
from qiskit.providers.models import BackendConfiguration
from qiskit.providers.providerutils import filter_backends

from .constants import DEFAULT_CLIENT_URL, DEFAULT_QISKITRC_FILE
from .credentials import Credentials
from .exceptions import *
from .ionq_backend import generate_backends
from .ionq_client import IonQClient

logger = logging.getLogger(__name__)


class IonQProvider(BaseProvider):
    """Provider for interacting with IonQ backends"""

    name = "ionq_provider"

    def __init__(self):
        super().__init__()
        self.credentials = None
        self._client = None
        self._backends = None

    def load_account(self, filepath: str = DEFAULT_QISKITRC_FILE):
        """Init client with stored credentials"""
        self.credentials = Credentials(filepath=filepath)
        if not self.credentials.token:
            raise IonQCredentialsError(
                "not able to load a token. do you have stored credentials?"
            )
        if not self.credentials.url:
            raise IonQCredentialsError(
                "not able to load a url. do you have stored credentials?"
            )
        self.client = IonQClient(self.credentials.token, self.credentials.url)

    def save_account(
        self,
        token: str,
        url: str = DEFAULT_CLIENT_URL,
        overwrite: bool = False,
        filepath: str = DEFAULT_QISKITRC_FILE,
    ):
        """Save credentials to qiskitrc (local disk).
        If no credentials are provided, saves the currently active credentials"""
        self.credentials = Credentials(token, url)
        self.client = IonQClient(self.credentials.token, self.credentials.url)
        self.credentials.save_credentials(overwrite, filepath)

    def enable_account(self, token: str, url: str = DEFAULT_CLIENT_URL):
        """Init provider with ephemeral credentials. can subsequently be saved."""
        self.credentials = Credentials(token, url)
        self.client = IonQClient(self.credentials.token, self.credentials.url)

    def delete_account(self, filepath: str = DEFAULT_QISKITRC_FILE):
        """Delete stored credentials"""
        self.credentials.remove_credentials(filepath)

    def backends(self, name=None, filters=None, **kwargs):
        """Return a list of available backends, filtered by filter options"""
        if not self.credentials:
            self.load_account()

        if not self._backends:
            self._backends = self._get_backends()

        backends = self._backends
        if name:
            backends = [
                backend for backend in backends if backend.name() == name
            ]

        return filter_backends(backends, filters=filters, **kwargs)

    def _get_backends(self):
        """get IonQ backends for provider"""
        return generate_backends(provider=self, client=self.client)
