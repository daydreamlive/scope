"""Unit tests for VenvSnapshot class."""

from unittest.mock import MagicMock, patch

from scope.core.plugins.venv_snapshot import VenvSnapshot


class TestVenvSnapshotCapture:
    """Tests for VenvSnapshot.capture() method."""

    def test_capture_creates_freeze_file(self, tmp_path):
        """Verify freeze.txt is created with uv pip freeze output."""
        freeze_file = tmp_path / "freeze.txt"
        resolved_backup = tmp_path / "resolved.txt.bak"
        resolved_file = tmp_path / "resolved.txt"

        with patch(
            "scope.core.plugins.venv_snapshot.get_freeze_backup_file",
            return_value=freeze_file,
        ):
            with patch(
                "scope.core.plugins.venv_snapshot.get_resolved_backup_file",
                return_value=resolved_backup,
            ):
                with patch(
                    "scope.core.plugins.venv_snapshot.get_resolved_file",
                    return_value=resolved_file,
                ):
                    with patch("subprocess.run") as mock_run:
                        mock_run.return_value = MagicMock(
                            returncode=0,
                            stdout="package1==1.0.0\npackage2==2.0.0\n",
                            stderr="",
                        )

                        snapshot = VenvSnapshot()
                        result = snapshot.capture()

        assert result is True
        assert freeze_file.exists()
        assert freeze_file.read_text() == "package1==1.0.0\npackage2==2.0.0\n"

    def test_capture_backs_up_resolved(self, tmp_path):
        """Verify resolved.txt is backed up to resolved.txt.bak."""
        freeze_file = tmp_path / "freeze.txt"
        resolved_backup = tmp_path / "resolved.txt.bak"
        resolved_file = tmp_path / "resolved.txt"

        # Create existing resolved.txt
        resolved_file.write_text("existing-package==1.0.0\n")

        with patch(
            "scope.core.plugins.venv_snapshot.get_freeze_backup_file",
            return_value=freeze_file,
        ):
            with patch(
                "scope.core.plugins.venv_snapshot.get_resolved_backup_file",
                return_value=resolved_backup,
            ):
                with patch(
                    "scope.core.plugins.venv_snapshot.get_resolved_file",
                    return_value=resolved_file,
                ):
                    with patch("subprocess.run") as mock_run:
                        mock_run.return_value = MagicMock(
                            returncode=0, stdout="frozen\n", stderr=""
                        )

                        snapshot = VenvSnapshot()
                        result = snapshot.capture()

        assert result is True
        assert resolved_backup.exists()
        assert resolved_backup.read_text() == "existing-package==1.0.0\n"

    def test_capture_skips_backup_if_no_resolved(self, tmp_path):
        """Verify no backup is created if resolved.txt doesn't exist."""
        freeze_file = tmp_path / "freeze.txt"
        resolved_backup = tmp_path / "resolved.txt.bak"
        resolved_file = tmp_path / "resolved.txt"  # Does not exist

        with patch(
            "scope.core.plugins.venv_snapshot.get_freeze_backup_file",
            return_value=freeze_file,
        ):
            with patch(
                "scope.core.plugins.venv_snapshot.get_resolved_backup_file",
                return_value=resolved_backup,
            ):
                with patch(
                    "scope.core.plugins.venv_snapshot.get_resolved_file",
                    return_value=resolved_file,
                ):
                    with patch("subprocess.run") as mock_run:
                        mock_run.return_value = MagicMock(
                            returncode=0, stdout="frozen\n", stderr=""
                        )

                        snapshot = VenvSnapshot()
                        result = snapshot.capture()

        assert result is True
        assert not resolved_backup.exists()

    def test_capture_returns_false_on_freeze_failure(self, tmp_path):
        """Verify capture returns False if uv pip freeze fails."""
        freeze_file = tmp_path / "freeze.txt"
        resolved_backup = tmp_path / "resolved.txt.bak"
        resolved_file = tmp_path / "resolved.txt"

        with patch(
            "scope.core.plugins.venv_snapshot.get_freeze_backup_file",
            return_value=freeze_file,
        ):
            with patch(
                "scope.core.plugins.venv_snapshot.get_resolved_backup_file",
                return_value=resolved_backup,
            ):
                with patch(
                    "scope.core.plugins.venv_snapshot.get_resolved_file",
                    return_value=resolved_file,
                ):
                    with patch("subprocess.run") as mock_run:
                        mock_run.return_value = MagicMock(
                            returncode=1, stdout="", stderr="Error: venv not found"
                        )

                        snapshot = VenvSnapshot()
                        result = snapshot.capture()

        assert result is False
        assert not freeze_file.exists()

    def test_capture_handles_exception(self, tmp_path):
        """Verify capture returns False on exception."""
        freeze_file = tmp_path / "freeze.txt"
        resolved_backup = tmp_path / "resolved.txt.bak"
        resolved_file = tmp_path / "resolved.txt"

        with patch(
            "scope.core.plugins.venv_snapshot.get_freeze_backup_file",
            return_value=freeze_file,
        ):
            with patch(
                "scope.core.plugins.venv_snapshot.get_resolved_backup_file",
                return_value=resolved_backup,
            ):
                with patch(
                    "scope.core.plugins.venv_snapshot.get_resolved_file",
                    return_value=resolved_file,
                ):
                    with patch(
                        "subprocess.run", side_effect=Exception("Subprocess error")
                    ):
                        snapshot = VenvSnapshot()
                        result = snapshot.capture()

        assert result is False


class TestVenvSnapshotRestore:
    """Tests for VenvSnapshot.restore() method."""

    def test_restore_syncs_from_freeze(self, tmp_path):
        """Verify uv pip sync is called with freeze file."""
        freeze_file = tmp_path / "freeze.txt"
        resolved_backup = tmp_path / "resolved.txt.bak"
        resolved_file = tmp_path / "resolved.txt"

        # Create freeze file
        freeze_file.write_text("package1==1.0.0\n")

        with patch(
            "scope.core.plugins.venv_snapshot.get_freeze_backup_file",
            return_value=freeze_file,
        ):
            with patch(
                "scope.core.plugins.venv_snapshot.get_resolved_backup_file",
                return_value=resolved_backup,
            ):
                with patch(
                    "scope.core.plugins.venv_snapshot.get_resolved_file",
                    return_value=resolved_file,
                ):
                    with patch("subprocess.run") as mock_run:
                        mock_run.return_value = MagicMock(
                            returncode=0, stdout="", stderr=""
                        )

                        snapshot = VenvSnapshot()
                        success, error = snapshot.restore()

        assert success is True
        assert error is None

        # Verify uv pip sync was called
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "uv" in args
        assert "pip" in args
        assert "sync" in args
        assert str(freeze_file) in args

    def test_restore_restores_resolved(self, tmp_path):
        """Verify resolved.txt is restored from backup."""
        freeze_file = tmp_path / "freeze.txt"
        resolved_backup = tmp_path / "resolved.txt.bak"
        resolved_file = tmp_path / "resolved.txt"

        # Create freeze file and backup
        freeze_file.write_text("package1==1.0.0\n")
        resolved_backup.write_text("original-content\n")

        with patch(
            "scope.core.plugins.venv_snapshot.get_freeze_backup_file",
            return_value=freeze_file,
        ):
            with patch(
                "scope.core.plugins.venv_snapshot.get_resolved_backup_file",
                return_value=resolved_backup,
            ):
                with patch(
                    "scope.core.plugins.venv_snapshot.get_resolved_file",
                    return_value=resolved_file,
                ):
                    with patch("subprocess.run") as mock_run:
                        mock_run.return_value = MagicMock(
                            returncode=0, stdout="", stderr=""
                        )

                        snapshot = VenvSnapshot()
                        success, error = snapshot.restore()

        assert success is True
        assert resolved_file.exists()
        assert resolved_file.read_text() == "original-content\n"

    def test_restore_fails_without_freeze_file(self, tmp_path):
        """Verify restore fails if no freeze file exists."""
        freeze_file = tmp_path / "freeze.txt"  # Does not exist
        resolved_backup = tmp_path / "resolved.txt.bak"
        resolved_file = tmp_path / "resolved.txt"

        with patch(
            "scope.core.plugins.venv_snapshot.get_freeze_backup_file",
            return_value=freeze_file,
        ):
            with patch(
                "scope.core.plugins.venv_snapshot.get_resolved_backup_file",
                return_value=resolved_backup,
            ):
                with patch(
                    "scope.core.plugins.venv_snapshot.get_resolved_file",
                    return_value=resolved_file,
                ):
                    snapshot = VenvSnapshot()
                    success, error = snapshot.restore()

        assert success is False
        assert "No freeze file found" in error

    def test_restore_returns_error_on_sync_failure(self, tmp_path):
        """Verify restore returns error if uv pip sync fails."""
        freeze_file = tmp_path / "freeze.txt"
        resolved_backup = tmp_path / "resolved.txt.bak"
        resolved_file = tmp_path / "resolved.txt"

        # Create freeze file
        freeze_file.write_text("package1==1.0.0\n")

        with patch(
            "scope.core.plugins.venv_snapshot.get_freeze_backup_file",
            return_value=freeze_file,
        ):
            with patch(
                "scope.core.plugins.venv_snapshot.get_resolved_backup_file",
                return_value=resolved_backup,
            ):
                with patch(
                    "scope.core.plugins.venv_snapshot.get_resolved_file",
                    return_value=resolved_file,
                ):
                    with patch("subprocess.run") as mock_run:
                        mock_run.return_value = MagicMock(
                            returncode=1, stdout="", stderr="Sync failed"
                        )

                        snapshot = VenvSnapshot()
                        success, error = snapshot.restore()

        assert success is False
        assert "Sync failed" in error


class TestVenvSnapshotDiscard:
    """Tests for VenvSnapshot.discard() method."""

    def test_discard_removes_backup_files(self, tmp_path):
        """Verify freeze.txt and resolved.txt.bak are removed."""
        freeze_file = tmp_path / "freeze.txt"
        resolved_backup = tmp_path / "resolved.txt.bak"
        resolved_file = tmp_path / "resolved.txt"

        # Create backup files
        freeze_file.write_text("frozen\n")
        resolved_backup.write_text("backup\n")

        with patch(
            "scope.core.plugins.venv_snapshot.get_freeze_backup_file",
            return_value=freeze_file,
        ):
            with patch(
                "scope.core.plugins.venv_snapshot.get_resolved_backup_file",
                return_value=resolved_backup,
            ):
                with patch(
                    "scope.core.plugins.venv_snapshot.get_resolved_file",
                    return_value=resolved_file,
                ):
                    snapshot = VenvSnapshot()
                    snapshot.discard()

        assert not freeze_file.exists()
        assert not resolved_backup.exists()

    def test_discard_handles_missing_files(self, tmp_path):
        """Verify discard doesn't fail if files don't exist."""
        freeze_file = tmp_path / "freeze.txt"  # Does not exist
        resolved_backup = tmp_path / "resolved.txt.bak"  # Does not exist
        resolved_file = tmp_path / "resolved.txt"

        with patch(
            "scope.core.plugins.venv_snapshot.get_freeze_backup_file",
            return_value=freeze_file,
        ):
            with patch(
                "scope.core.plugins.venv_snapshot.get_resolved_backup_file",
                return_value=resolved_backup,
            ):
                with patch(
                    "scope.core.plugins.venv_snapshot.get_resolved_file",
                    return_value=resolved_file,
                ):
                    snapshot = VenvSnapshot()
                    # Should not raise
                    snapshot.discard()


class TestVenvSnapshotHasSnapshot:
    """Tests for VenvSnapshot.has_snapshot() method."""

    def test_has_snapshot_true_when_freeze_exists(self, tmp_path):
        """Verify has_snapshot returns True if freeze.txt exists."""
        freeze_file = tmp_path / "freeze.txt"
        resolved_backup = tmp_path / "resolved.txt.bak"
        resolved_file = tmp_path / "resolved.txt"

        freeze_file.write_text("frozen\n")

        with patch(
            "scope.core.plugins.venv_snapshot.get_freeze_backup_file",
            return_value=freeze_file,
        ):
            with patch(
                "scope.core.plugins.venv_snapshot.get_resolved_backup_file",
                return_value=resolved_backup,
            ):
                with patch(
                    "scope.core.plugins.venv_snapshot.get_resolved_file",
                    return_value=resolved_file,
                ):
                    snapshot = VenvSnapshot()
                    assert snapshot.has_snapshot() is True

    def test_has_snapshot_false_when_no_freeze(self, tmp_path):
        """Verify has_snapshot returns False if freeze.txt doesn't exist."""
        freeze_file = tmp_path / "freeze.txt"  # Does not exist
        resolved_backup = tmp_path / "resolved.txt.bak"
        resolved_file = tmp_path / "resolved.txt"

        with patch(
            "scope.core.plugins.venv_snapshot.get_freeze_backup_file",
            return_value=freeze_file,
        ):
            with patch(
                "scope.core.plugins.venv_snapshot.get_resolved_backup_file",
                return_value=resolved_backup,
            ):
                with patch(
                    "scope.core.plugins.venv_snapshot.get_resolved_file",
                    return_value=resolved_file,
                ):
                    snapshot = VenvSnapshot()
                    assert snapshot.has_snapshot() is False
