"""Shared pytest fixtures."""

from unittest.mock import patch

import pytest


@pytest.fixture
def patch_process_functions():
    """Patch functions that spawn processes or terminate pytest."""
    with patch("scope.server.app.os._exit") as mock_exit:
        with patch("scope.server.app.subprocess.Popen") as mock_popen:
            with patch("os.execv") as mock_execv:
                yield {
                    "os_exit": mock_exit,
                    "popen": mock_popen,
                    "execv": mock_execv,
                }
