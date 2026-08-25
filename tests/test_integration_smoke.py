from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_complete_synthetic_workflow(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "deep_recsys_lab.smoke",
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    result = json.loads((output / "smoke_result.json").read_text(encoding="utf-8"))
    assert result["status"] == "PASS"
    assert str(result["model_tag"]).startswith("deep_recsys:smoke-")
