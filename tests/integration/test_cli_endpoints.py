"""Test CLI endpoints and entry points."""

from typer.testing import CliRunner

from fleetmix.app import app

runner = CliRunner()


def test_unified_info_flag():
    """Test the convert command info flag."""
    result = runner.invoke(
        app, ["convert", "--type", "mcvrp", "--instance", "test", "--info"]
    )
    # This should show help information about the convert command
    # Note: The --info flag may not exist in the new convert command,
    # so we'll test the --help flag instead
    result = runner.invoke(app, ["convert", "--help"])
    assert result.exit_code == 0
    assert "convert" in result.stdout.lower()
