"""Tests for prefpo.cli — argument parsing."""

import asyncio
import sys

import pytest
from prefpo.cli import main
def test_cli_missing_args(capsys):
    """Exits with error when required args are missing."""
    sys.argv = ["prefpo"]
    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(main())
    assert exc_info.value.code != 0
def test_cli_help_works(capsys):
    """--help prints usage and exits 0."""
    sys.argv = ["prefpo", "--help"]
    with pytest.raises(SystemExit) as exc_info:
        asyncio.run(main())
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "PrefPO" in captured.out
