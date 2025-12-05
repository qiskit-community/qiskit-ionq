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

"""Tests for the IonQ primitives (IonQSampler and IonQEstimator)."""

from unittest.mock import MagicMock

import pytest
from qiskit.primitives import BaseSamplerV2, BackendEstimatorV2

from qiskit_ionq.ionq_primitives import IonQSampler, IonQEstimator


def test_sampler_requires_keyword_backend(simulator_backend):
    """IonQSampler should require `backend` as a keyword-only argument."""
    # Positional use should fail because the signature is `*, backend`.
    with pytest.raises(TypeError):
        IonQSampler(simulator_backend)  # type: ignore[call-arg]

    # But the keyword form should work.
    sampler = IonQSampler(backend=simulator_backend)
    assert isinstance(sampler, BaseSamplerV2)


def test_sampler_uses_backend_default_shots(simulator_backend):
    """IonQSampler default_shots should default to backend.options.shots."""
    sampler = IonQSampler(backend=simulator_backend)

    # The sampler should expose the backend it is wrapping.
    assert sampler.backend is simulator_backend

    # And it should pick up the backend's default shot count if none is given.
    backend_default = getattr(simulator_backend.options, "shots", None)
    assert sampler.default_shots == backend_default


def test_sampler_explicit_default_shots_override(simulator_backend):
    """An explicit default_shots argument should override the backend default."""
    explicit_shots = 2048
    sampler = IonQSampler(backend=simulator_backend, default_shots=explicit_shots)

    assert sampler.backend is simulator_backend
    assert sampler.default_shots == explicit_shots


def test_estimator_is_backend_estimator(simulator_backend):
    """IonQEstimator should be a BackendEstimatorV2."""
    estimator = IonQEstimator(backend=simulator_backend)
    assert isinstance(estimator, BackendEstimatorV2)


def test_estimator_applies_run_options_to_backend(simulator_backend, monkeypatch):
    """IonQEstimator should pull run_options out and pass them to backend.set_options."""
    backend = simulator_backend

    # Spy on set_options so we don't actually mutate backend options for real.
    set_options_spy = MagicMock()
    monkeypatch.setattr(backend, "set_options", set_options_spy)

    run_options = {"shots": 123, "job_settings": {"foo": "bar"}}
    IonQEstimator(backend=backend, options={"run_options": run_options})

    # IonQEstimator.__init__ should have called set_options with the run_options
    # extracted from the user options.
    set_options_spy.assert_called_once_with(**run_options)
