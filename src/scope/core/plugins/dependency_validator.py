"""Dependency validation for plugin installation.

Validates that plugin dependencies are compatible with the project's
declared dependencies before installation, preventing venv corruption.
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class InstallValidationResult:
    """Result of dependency validation."""

    is_valid: bool
    error_message: str | None = None  # Raw stderr from uv on conflict


class DependencyValidator:
    """Validates plugin dependencies before installation.

    Uses uv pip compile to check if plugin dependencies can be resolved
    together with the project's existing dependencies.
    """

    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path.cwd()

    def validate_install(self, packages: list[str]) -> InstallValidationResult:
        """Check if plugin can be installed without breaking project deps.

        Compiles the project's pyproject.toml along with the plugin packages
        to verify everything can be resolved together. This respects the
        project's [tool.uv.sources] configuration for custom package indexes.

        Args:
            packages: List of package specifiers to validate (e.g., ["some-plugin>=1.0"])

        Returns:
            InstallValidationResult with is_valid=True if resolution succeeds,
            or is_valid=False with error_message containing uv's conflict explanation.
        """
        pyproject_path = self.project_root / "pyproject.toml"

        if not pyproject_path.exists():
            # No pyproject.toml to conflict with
            return InstallValidationResult(is_valid=True)

        # Create temp file with plugin packages to add
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for pkg in packages:
                f.write(f"{pkg}\n")
            plugin_requirements = f.name

        try:
            # Run uv pip compile with pyproject.toml + plugin requirements
            # This respects [tool.uv.sources] from pyproject.toml
            # Set PYTHONUTF8=1 in subprocess env for proper Unicode handling
            env = {**os.environ, "PYTHONUTF8": "1"}
            result = subprocess.run(
                [
                    "uv",
                    "pip",
                    "compile",
                    str(pyproject_path),
                    plugin_requirements,
                    "--torch-backend",
                    "cu128",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=self.project_root,
                env=env,
            )

            if result.returncode != 0:
                return InstallValidationResult(
                    is_valid=False, error_message=result.stderr.strip()
                )

            return InstallValidationResult(is_valid=True)
        finally:
            # Clean up temp file
            Path(plugin_requirements).unlink(missing_ok=True)
