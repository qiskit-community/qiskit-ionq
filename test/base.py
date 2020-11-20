# -*- coding: utf-8 -*-
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

import unittest
from unittest import mock


class MockCredentialsTestCase(unittest.TestCase):
    """Base test case for ensuring mock credentials are in the test environment
    prior to importing or initializing an IonQ provider.
    """

    @classmethod
    def setUpClass(cls):
        cls.cred_mock = mock.patch.dict(
            "os.environ",
            {
                "QISKIT_IONQ_API_TOKEN": "token",
                "QISKIT_IONQ_API_URL": "url",
            },
        )
        cls.cred_mock.start()

    @classmethod
    def tearDownClass(cls):
        cls.cred_mock.stop()
