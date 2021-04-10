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

"""Constants used by components of the provider."""
import enum

from qiskit.providers import jobstatus


class APIJobStatus(enum.Enum):
    """Job status values.

    Happy path job flow is
    SUBMITTED -> READY -> RUNNING -> COMPLETED
    """

    SUBMITTED = "submitted"
    READY = "ready"
    RUNNING = "running"
    CANCELED = "canceled"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatusMap(enum.Enum):
    """Enum to map IonQ job statuses to a qiskit JobStatus.

    IonQ Status Transition happy path:
    SUBMITTED -> READY -> RUNNING -> COMPLETED

    Qiskit Status Transition happy path:
    INITIALIZING -> QUEUED -> RUNNING -> DONE
    """

    SUBMITTED = jobstatus.JobStatus.INITIALIZING.name
    READY = jobstatus.JobStatus.QUEUED.name
    RUNNING = jobstatus.JobStatus.RUNNING.name
    CANCELED = jobstatus.JobStatus.CANCELLED.name
    COMPLETED = jobstatus.JobStatus.DONE.name
    FAILED = jobstatus.JobStatus.ERROR.name


__all__ = ["APIJobStatus", "JobStatusMap"]
