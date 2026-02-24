"""Tests for workflow format version migration."""

from __future__ import annotations

import pytest

from scope.core.workflows.migrate import migrate_workflow


class TestMigrateWorkflow:
    """Tests for :func:`migrate_workflow`."""

    def _base_data(self, **overrides) -> dict:
        data = {
            "format": "scope-workflow",
            "format_version": "1.0",
            "metadata": {
                "name": "test",
                "description": "",
                "author": "",
                "created_at": "2025-01-01T00:00:00Z",
                "scope_version": "0.1.0",
            },
            "pipelines": [],
        }
        data.update(overrides)
        return data

    def test_current_version_passes_through(self):
        data = self._base_data()
        result = migrate_workflow(data)
        assert result is data  # same object, unchanged

    def test_future_version_raises(self):
        data = self._base_data(format_version="2.0")
        with pytest.raises(ValueError, match="newer than supported"):
            migrate_workflow(data)

    def test_semver_comparison_not_lexicographic(self):
        """'1.10' > '1.9' must hold (not lexicographic)."""
        data = self._base_data(format_version="1.10")
        with pytest.raises(ValueError, match="newer than supported"):
            migrate_workflow(data)

    def test_missing_format_version_defaults_to_1_0(self):
        data = self._base_data()
        del data["format_version"]
        result = migrate_workflow(data)
        assert result is data
