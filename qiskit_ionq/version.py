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

"""Contains the package version."""

import os
import pathlib
import subprocess
from typing import List

pkg_parent = pathlib.Path(__file__).parent.parent.absolute()

# major, minor, micro
VERSION_INFO = ".".join(map(str, (0, 4, 4)))


def _minimal_ext_cmd(cmd: List[str]) -> bytes:
    # construct minimal environment
    env = {
        "LANGUAGE": "C",  # LANGUAGE is used on win32
        "LANG": "C",
        "LC_ALL": "C",
    }
    for k in ["SYSTEMROOT", "PATH"]:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(pkg_parent),
    )
    out = proc.communicate()[0]
    if proc.returncode > 0:
        raise OSError
    return out


def git_version() -> str:
    """Get the current git head sha1."""
    # Determine if we're at master
    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        git_revision = out.strip().decode("ascii")
    except OSError:
        git_revision = "Unknown"

    return git_revision


def get_version_info() -> str:
    """Get the full version string."""
    # Adding the git rev number needs to be done inside
    # write_version_py(), otherwise the import of scipy.version messes
    # up the build under Python 3.
    git_dir = pkg_parent / ".git"
    if not git_dir.exists():
        return VERSION_INFO

    full_version = VERSION_INFO
    try:
        release = _minimal_ext_cmd(["git", "tag", "-l", "--points-at", "HEAD"])
    except Exception:  # pylint: disable=broad-except
        return full_version

    if not release:
        git_revision = git_version()
        if ".dev" not in full_version:
            full_version += ".dev0"
        full_version += "+" + git_revision[:7]

    return full_version


__version__ = get_version_info()

__all__ = ["__version__"]
