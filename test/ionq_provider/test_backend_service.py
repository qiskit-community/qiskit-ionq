# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import unittest

from qiskit_ionq_provider import IonQProvider


class TestProvider(unittest.TestCase):

    def test_provider_autocomplete(self):
        """Verifies that provider.backends autocomplete works.
        """
        pro = IonQProvider('123456')

        for backend in pro.backends():
            self.assertTrue(hasattr(pro.backends, backend.name()))

    def test_provider_getbackend(self):
        """Verifies that provider.get_backend works.
        """
        pro = IonQProvider('123456')

        for backend in pro.backends():
            self.assertTrue(backend == pro.get_backend(backend.name()))
