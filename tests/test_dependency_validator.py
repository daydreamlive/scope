"""Unit tests for dependency_validator module."""

from pathlib import Path

from scope.core.plugins.dependency_validator import (
    DependencyValidator,
    InstallValidationResult,
)


class TestInstallValidationResult:
    """Tests for InstallValidationResult dataclass."""

    def test_valid_result(self):
        """Should create valid result with no error message."""
        result = InstallValidationResult(is_valid=True)

        assert result.is_valid is True
        assert result.error_message is None

    def test_invalid_result_with_error(self):
        """Should create invalid result with error message."""
        result = InstallValidationResult(
            is_valid=False, error_message="Conflict detected"
        )

        assert result.is_valid is False
        assert result.error_message == "Conflict detected"


class TestDependencyValidator:
    """Tests for DependencyValidator class."""

    def test_init_with_default_project_root(self):
        """Should use current working directory as default project root."""
        validator = DependencyValidator()

        assert validator.project_root == Path.cwd()

    def test_init_with_custom_project_root(self, tmp_path):
        """Should use provided project root."""
        validator = DependencyValidator(project_root=tmp_path)

        assert validator.project_root == tmp_path

    def test_validate_no_pyproject_returns_valid(self, tmp_path):
        """Should return valid when no pyproject.toml exists."""
        validator = DependencyValidator(project_root=tmp_path)

        result = validator.validate_install(["requests"])

        assert result.is_valid is True
        assert result.error_message is None

    def test_validate_compatible_package(self, tmp_path):
        """Should return valid for compatible packages."""
        # Create a minimal pyproject.toml
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            """
[project]
name = "test-project"
version = "0.1.0"
dependencies = ["click>=8.0"]
"""
        )

        validator = DependencyValidator(project_root=tmp_path)
        result = validator.validate_install(["requests"])

        assert result.is_valid is True

    def test_validate_conflicting_package(self, tmp_path):
        """Should return invalid for conflicting packages."""
        # Create a pyproject.toml with a pinned dependency
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            """
[project]
name = "test-project"
version = "0.1.0"
dependencies = ["click==8.0.0"]
"""
        )

        validator = DependencyValidator(project_root=tmp_path)
        # Try to install a conflicting click version
        result = validator.validate_install(["click>=99.0.0"])

        assert result.is_valid is False
        assert result.error_message is not None
        assert "click" in result.error_message.lower()

    def test_validate_multiple_packages(self, tmp_path):
        """Should validate multiple packages at once."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            """
[project]
name = "test-project"
version = "0.1.0"
dependencies = ["click>=8.0"]
"""
        )

        validator = DependencyValidator(project_root=tmp_path)
        result = validator.validate_install(["requests", "httpx"])

        assert result.is_valid is True

    def test_validate_empty_packages_list(self, tmp_path):
        """Should handle empty packages list."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            """
[project]
name = "test-project"
version = "0.1.0"
dependencies = ["click>=8.0"]
"""
        )

        validator = DependencyValidator(project_root=tmp_path)
        result = validator.validate_install([])

        # Empty list should resolve successfully (just project deps)
        assert result.is_valid is True

    def test_validate_nonexistent_package(self, tmp_path):
        """Should return invalid for packages that don't exist."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            """
[project]
name = "test-project"
version = "0.1.0"
dependencies = ["click>=8.0"]
"""
        )

        validator = DependencyValidator(project_root=tmp_path)
        result = validator.validate_install(
            ["this-package-definitely-does-not-exist-xyz123"]
        )

        assert result.is_valid is False
        assert result.error_message is not None

    def test_validate_with_version_specifier(self, tmp_path):
        """Should handle version specifiers correctly."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            """
[project]
name = "test-project"
version = "0.1.0"
dependencies = ["requests>=2.0"]
"""
        )

        validator = DependencyValidator(project_root=tmp_path)
        result = validator.validate_install(["urllib3>=1.26,<3"])

        assert result.is_valid is True

    def test_validate_empty_dependencies(self, tmp_path):
        """Should handle pyproject.toml with no dependencies."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            """
[project]
name = "test-project"
version = "0.1.0"
"""
        )

        validator = DependencyValidator(project_root=tmp_path)
        result = validator.validate_install(["requests"])

        # Should still work - uv pip compile can handle this
        assert result.is_valid is True
