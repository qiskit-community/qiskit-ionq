"""module for managing IonQ credentials in Qiskit"""
import json
import logging
import os
from configparser import ConfigParser, ParsingError
from pathlib import Path

from .exceptions import *

logger = logging.getLogger(__name__)

from .constants import DEFAULT_CLIENT_URL, DEFAULT_QISKITRC_FILE, QISKITRC_SECTION_NAME


class Credentials:
    """ manager for storing, accessing credentials for IonQ client."""

    TOKEN_ENVVAR = "IONQ_API_TOKEN"
    URL_ENVVAR = "IONQ_API_URL"

    def __init__(self, token: str = None, url: str = None, filepath: str = None):
        discovered_token, discovered_url = self._load_credentials(DEFAULT_QISKITRC_FILE)
        self.token = token or discovered_token
        self.url = url or discovered_url or DEFAULT_CLIENT_URL

    def _load_from_env_var(self):
        """Get credentials from environment varibles """
        token = os.environ.get(self.TOKEN_ENVVAR)
        url = os.environ.get(self.URL_ENVVAR)
        return token, url

    def _load_from_qiskitrc(self, filepath: str = None):
        """ Get credentials from qiskitrc file"""
        if not Path(filepath).exists():
            logger.warning(
                "qiskitrc path {} does not exist. Not loading credentials from file.".format(
                    filepath
                )
            )
            return None, None

        config_parser = ConfigParser()
        try:
            config_parser.read(filepath)
        except ParsingError as exception:
            raise IonQCredentialsError(str(exception))

        token = config_parser.get(QISKITRC_SECTION_NAME, "token")
        url = config_parser.get(QISKITRC_SECTION_NAME, "url")
        return token, url

    def _load_credentials(self, filepath: str = None):
        """Get credentials. first try env vars then try qiskitrc.
        If both are set, qiskitrc takes precedence"""
        token, url = self._load_from_env_var()
        rc_token, rc_url = self._load_from_qiskitrc(filepath)

        if rc_token and rc_url:
            token, url = rc_token, rc_url

        return token, url

    def save_credentials(
        self, overwrite: bool = False, filepath: str = DEFAULT_QISKITRC_FILE
    ):
        """save credentials to qiskitrc after making sure there are creds to save"""

        if self.token is None:
            raise IonQCredentialsError(
                "No token to save. Please enable an account before attempting to save one."
            )
        if self.url is None:
            raise IonQCredentialsError(
                "No url to save. Please enable an account before attempting to save one."
            )
        self._save_qiskitrc(overwrite, filepath)

    def remove_credentials(self, filepath: str = DEFAULT_QISKITRC_FILE):
        # TODO: check that there are creds to remove?
        self._remove_creds_from_qiskitrc(filepath)

    def _save_qiskitrc(
        self, overwrite: bool = False, filepath: str = DEFAULT_QISKITRC_FILE
    ):
        """Save credentials to qiskitrc configuration file
        The default qiskitrc location is ``$HOME/.qiskitrc/qiskitrc``
        """
        config_parser = ConfigParser()
        try:
            config_parser.read(filepath)
        except ParsingError as ex:
            raise Exception(str(ex))

        if not config_parser.has_section(QISKITRC_SECTION_NAME):
            config_parser[QISKITRC_SECTION_NAME] = {}
        for k, v in {"token": self.token, "url": self.url}.items():
            if k not in config_parser[QISKITRC_SECTION_NAME] or not overwrite:
                if isinstance(v, dict):
                    config_parser[QISKITRC_SECTION_NAME].update({k: json.dumps(v)})
                else:
                    config_parser[QISKITRC_SECTION_NAME].update({k: v})

        (Path(filepath).parent).mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as conf_file:
            config_parser.write(conf_file)

    def _remove_creds_from_qiskitrc(self, filepath: str = DEFAULT_QISKITRC_FILE):
        """Remove  credentials from qiskitrc configuration file
        The default qiskitrc location is ``$HOME/.qiskitrc/qiskitrc``
        """
        config_parser = ConfigParser()
        try:
            config_parser.read(filepath)
        except ParsingError as exception:
            raise IonQCredentialsError(str(exception))
        if not config_parser.has_section(QISKITRC_SECTION_NAME):
            return
        if config_parser.has_option(QISKITRC_SECTION_NAME, "token"):
            config_parser.remove_option(QISKITRC_SECTION_NAME, "token")
        if config_parser.has_option(QISKITRC_SECTION_NAME, "url"):
            config_parser.remove_option(QISKITRC_SECTION_NAME, "url")

        (Path(filepath).parent).mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as conf_file:
            config_parser.write(conf_file)
