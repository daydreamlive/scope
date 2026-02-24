"""Format version migration for workflow files.

Ensures older workflow formats can be upgraded to the current version before
Pydantic validation.  Rejects workflow formats *newer* than the running Scope
can understand.
"""

from __future__ import annotations

from typing import Any

from packaging.version import Version

from .schema import WORKFLOW_FORMAT_VERSION


def migrate_workflow(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate a raw workflow dict to the current format version.

    Returns *data* unchanged if it is already at the current version.
    Raises :class:`ValueError` if the format version is newer than supported.
    """
    version = Version(data.get("format_version", "1.0"))
    current = Version(WORKFLOW_FORMAT_VERSION)

    if version > current:
        raise ValueError(
            f"Workflow format version {version} is newer than supported ({current})"
        )

    if version == current:
        return data

    # Future: add migration steps for older versions here.
    # e.g. if version < Version("1.1"): data = _migrate_1_0_to_1_1(data)
    return data
