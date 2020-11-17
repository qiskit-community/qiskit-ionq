from enum import Enum
from pathlib import Path
from urllib import parse as urlparse


class APIJobStatus(Enum):
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


class JobStatus(Enum):
    """friendly output for job statuses"""

    INITIALIZING = "job is being initialized"
    SUBMITTED = "job has been submitted and is in preflight"
    READY = "job has been validated and is in queue"
    RUNNING = "job is actively running"
    CANCELED = "job has been canceled"
    COMPLETED = "job has been run successfully"
    FAILED = "job has failed"


JOB_FINAL_STATES = (JobStatus.COMPLETED, JobStatus.CANCELED, JobStatus.FAILED)
"""states that are considered 'final' by provider logic"""

DEFAULT_QISKITRC_FILE = str(Path.home() / ".qiskit" / "qiskitrc")
"""default file to use for credential storage/retrieval"""

QISKITRC_SECTION_NAME = "ionq"
"""heading in qiskitrc to reference the ionq creds specifically """

DEFAULT_CLIENT_URL = urlparse.urlunsplit(
    (
        "https",  # protocol/scheme
        "api.ionq.co",  # domain
        "v0.1",  # path
        None,  # query
        None,  # fragment
    )
)
"""api domain and version"""
