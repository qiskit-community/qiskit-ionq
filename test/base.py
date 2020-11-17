import unittest
from unittest import mock

from qiskit.providers.ionq.credentials import Credentials


class MockCredentialsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cred_mock = mock.patch.dict(
            "os.environ",
            {Credentials.TOKEN_ENVVAR: "token", Credentials.URL_ENVVAR: "url"},
        )
        cls.cred_mock.start()

    @classmethod
    def tearDownClass(cls):
        cls.cred_mock.stop()
