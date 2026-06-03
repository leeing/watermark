"""Smoke test — verifies the package can be imported and the CLI group exists."""

from click.testing import CliRunner

from watermark.main import cli


def test_cli_help() -> None:
    """The `watermark --help` command succeeds and lists subcommands."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0, result.output
    assert "detect" in result.output
    assert "mask" in result.output
    assert "remove" in result.output
