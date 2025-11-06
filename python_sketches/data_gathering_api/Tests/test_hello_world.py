import pytest
from click.testing import CliRunner
from module.__main__ import cli


@pytest.fixture
def runner():
    """Fixture for Click CLI runner."""
    return CliRunner()


def test_cli_runs_successfully(runner):
    """Test that the CLI command runs without errors."""
    result = runner.invoke(cli)
    assert result.exit_code == 0


def test_cli_outputs_hello_world(runner):
    """Test that the CLI outputs the expected message."""
    result = runner.invoke(cli)
    assert "Hello World!" in result.output


def test_cli_output_format(runner):
    """Test that the CLI output format is correct."""
    result = runner.invoke(cli)
    assert result.output.strip() == "Hello World!"


def test_cli_help(runner):
    """Test that the CLI help command works."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Example script." in result.output
    assert "Usage:" in result.output
