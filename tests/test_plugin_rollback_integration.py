"""Integration tests for plugin installation rollback using fixture plugins."""

import subprocess
from pathlib import Path

import pytest

from scope.core.plugins.manager import (
    PluginDependencyError,
    PluginInstallError,
    PluginManager,
)
from scope.core.plugins.plugins_config import (
    get_freeze_backup_file,
    get_plugins_file,
    get_resolved_backup_file,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "plugins"


class TestPluginRollbackIntegration:
    """Integration tests for venv rollback with real subprocess calls."""

    @pytest.fixture
    def clean_plugins_dir(self, tmp_path, monkeypatch):
        """Use a temporary plugins directory for isolation."""
        monkeypatch.setenv("DAYDREAM_SCOPE_PLUGINS_DIR", str(tmp_path))
        return tmp_path

    @pytest.fixture
    def capture_venv_state(self):
        """Capture current venv state for comparison."""

        def _capture():
            result = subprocess.run(
                ["uv", "pip", "freeze"], capture_output=True, text=True
            )
            return (
                set(result.stdout.strip().split("\n"))
                if result.stdout.strip()
                else set()
            )

        return _capture

    def test_compile_failure_discards_snapshot(self, clean_plugins_dir):
        """Compile failure should discard snapshot, not restore."""
        pm = PluginManager()
        plugin_path = FIXTURES_DIR / "scope-test-plugin-compile-fail"

        with pytest.raises(PluginDependencyError) as exc_info:
            pm._install_plugin_sync(str(plugin_path))

        # Error should mention torch version constraint
        assert (
            "torch" in str(exc_info.value).lower()
            or "resolution" in str(exc_info.value).lower()
        )

        # Verify no backup files remain
        assert not get_freeze_backup_file().exists()
        assert not get_resolved_backup_file().exists()

        # Verify plugins.txt is empty or doesn't contain the failed plugin
        plugins_file = get_plugins_file()
        if plugins_file.exists():
            content = plugins_file.read_text()
            assert "scope-test-plugin-compile-fail" not in content

    def test_sync_failure_restores_venv(self, clean_plugins_dir, capture_venv_state):
        """Sync failure should restore venv to pre-install state."""
        before_state = capture_venv_state()
        pm = PluginManager()
        plugin_path = FIXTURES_DIR / "scope-test-plugin-sync-fail"

        with pytest.raises(PluginInstallError) as exc_info:
            pm._install_plugin_sync(str(plugin_path))

        # Error should mention intentional failure
        assert (
            "INTENTIONAL FAILURE" in str(exc_info.value)
            or "failed" in str(exc_info.value).lower()
        )

        # Capture state after failed install + rollback
        after_state = capture_venv_state()

        # Venv should be restored to original state
        assert before_state == after_state, "Venv was not restored after rollback"

        # Verify no backup files remain
        assert not get_freeze_backup_file().exists()
        assert not get_resolved_backup_file().exists()

        # Verify plugins.txt unchanged
        plugins_file = get_plugins_file()
        if plugins_file.exists():
            assert "scope-test-plugin-sync-fail" not in plugins_file.read_text()

    def test_fixture_plugins_exist(self):
        """Verify the test fixture plugins are in place."""
        compile_fail_path = FIXTURES_DIR / "scope-test-plugin-compile-fail"
        sync_fail_path = FIXTURES_DIR / "scope-test-plugin-sync-fail"

        assert compile_fail_path.exists(), "compile-fail fixture plugin not found"
        assert sync_fail_path.exists(), "sync-fail fixture plugin not found"

        # Verify pyproject.toml files exist
        assert (compile_fail_path / "pyproject.toml").exists()
        assert (sync_fail_path / "pyproject.toml").exists()

        # Verify hatch_build.py exists for sync-fail
        assert (sync_fail_path / "hatch_build.py").exists()

    def test_editable_sync_failure_restores_venv(
        self, clean_plugins_dir, capture_venv_state
    ):
        """Editable install failure should also restore venv."""
        before_state = capture_venv_state()
        pm = PluginManager()
        plugin_path = FIXTURES_DIR / "scope-test-plugin-sync-fail"

        with pytest.raises(PluginInstallError):
            pm._install_editable_plugin(str(plugin_path))

        after_state = capture_venv_state()
        assert before_state == after_state, (
            "Venv was not restored after editable rollback"
        )
