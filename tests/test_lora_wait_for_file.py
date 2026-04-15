"""Tests for the LoRA download-wait helper in lora/utils.py.

Covers the race condition where a Civitai LoRA file is still being
downloaded when LongLivePipeline.__init__ calls load_lora_weights.
See: daydreamlive/scope#937
"""

import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from scope.core.pipelines.wan2_1.lora.utils import _wait_for_lora_file


class TestWaitForLoraFile:
    """Unit tests for _wait_for_lora_file."""

    def test_file_already_present_returns_immediately(self, tmp_path: Path):
        """If the file exists before the first check, return True right away."""
        lora_file = tmp_path / "model.safetensors"
        lora_file.touch()

        start = time.monotonic()
        result = _wait_for_lora_file(str(lora_file), timeout=10, poll_interval=0.1)
        elapsed = time.monotonic() - start

        assert result is True
        # Should not have slept at all
        assert elapsed < 0.5

    def test_file_appears_during_wait(self, tmp_path: Path):
        """File appears mid-poll; function returns True after ≤2 poll intervals."""
        lora_file = tmp_path / "late.safetensors"

        def _create_later():
            time.sleep(0.3)
            lora_file.touch()

        t = threading.Thread(target=_create_later, daemon=True)
        t.start()

        result = _wait_for_lora_file(str(lora_file), timeout=5, poll_interval=0.1)
        t.join()

        assert result is True

    def test_file_never_appears_returns_false(self, tmp_path: Path):
        """File never shows up; function returns False after timeout."""
        missing = str(tmp_path / "missing.safetensors")

        result = _wait_for_lora_file(missing, timeout=0.3, poll_interval=0.1)

        assert result is False

    def test_timeout_zero_disables_wait(self, tmp_path: Path):
        """timeout=0 means skip the poll entirely; missing file → False instantly."""
        missing = str(tmp_path / "no_wait.safetensors")

        start = time.monotonic()
        result = _wait_for_lora_file(missing, timeout=0, poll_interval=0.1)
        elapsed = time.monotonic() - start

        assert result is False
        assert elapsed < 0.1

    def test_env_var_overrides_default_timeout(self, tmp_path: Path, monkeypatch):
        """SCOPE_LORA_DOWNLOAD_WAIT_TIMEOUT env var controls the default timeout."""
        missing = str(tmp_path / "env_timeout.safetensors")
        monkeypatch.setenv("SCOPE_LORA_DOWNLOAD_WAIT_TIMEOUT", "0.2")

        start = time.monotonic()
        # Pass timeout=None so env var is picked up
        result = _wait_for_lora_file(missing, timeout=None, poll_interval=0.05)
        elapsed = time.monotonic() - start

        assert result is False
        # Should respect the 0.2 s limit (allow generous buffer for CI)
        assert elapsed < 1.5


class TestLoadLoraWeightsWaits:
    """Integration-style tests ensuring load_lora_weights uses the poll helper."""

    def test_raises_after_timeout_when_file_never_appears(self, tmp_path: Path):
        """load_lora_weights should raise FileNotFoundError when wait times out."""
        from scope.core.pipelines.wan2_1.lora.utils import load_lora_weights

        missing = str(tmp_path / "never_there.safetensors")
        # Short timeout so the test stays fast
        with patch.dict(os.environ, {"SCOPE_LORA_DOWNLOAD_WAIT_TIMEOUT": "0.2"}):
            with pytest.raises(FileNotFoundError, match="LoRA file not found"):
                load_lora_weights(missing)

    def test_succeeds_when_file_appears_during_wait(self, tmp_path: Path):
        """load_lora_weights should succeed if the file arrives within the timeout.

        We bypass load_lora_weights itself and test _wait_for_lora_file + the
        safetensors load in combination, keeping the test fast by using a short
        poll interval.
        """
        import torch
        from safetensors.torch import save_file

        from scope.core.pipelines.wan2_1.lora.utils import (
            _wait_for_lora_file,
            load_lora_weights,
        )

        lora_file = tmp_path / "delayed.safetensors"

        # Write a minimal safetensors file after a short delay
        def _write_later():
            time.sleep(0.3)
            tensors = {"lora_A.weight": torch.zeros(4, 4)}
            save_file(tensors, str(lora_file))

        t = threading.Thread(target=_write_later, daemon=True)
        t.start()

        # Directly exercise _wait_for_lora_file with a short poll interval, then
        # verify load_lora_weights can read the now-present file.
        appeared = _wait_for_lora_file(str(lora_file), timeout=5, poll_interval=0.1)
        t.join()

        assert appeared is True
        result = load_lora_weights(str(lora_file))
        assert "lora_A.weight" in result
