from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from recsys.cli import app


def test_cli_identity_plugins_and_workspace_root(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["plugins", "list"])
    assert result.exit_code == 0
    assert '"multivae"' in result.stdout

    prepared = runner.invoke(
        app,
        [
            "--workspace-root",
            str(tmp_path),
            "data",
            "prepare",
            "dataset.parameters.users=3",
            "dataset.parameters.items=9",
            "dataset.parameters.interactions_per_user=4",
        ],
    )
    assert prepared.exit_code == 0
    assert (tmp_path / "var" / "data" / "prepared").is_dir()
