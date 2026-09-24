"""The Streamlit entrypoint must run from any working directory, exactly as ``streamlit run`` does.

``streamlit run app/ui/streamlit_app.py`` executes the file with ``app/ui`` as ``sys.path[0]``,
so a plain ``from app... import`` fails unless the script puts the repo root on the path itself.
This test reproduces that environment in a subprocess: another working directory, no PYTHONPATH,
and the script executed by path through Streamlit's own AppTest runner.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "app" / "ui" / "streamlit_app.py"

RUNNER = """
import sys
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(sys.argv[1], default_timeout=60)
at.run()
if at.exception:
    print("EXCEPTION:", at.exception[0].value)
    sys.exit(1)
print("OK: rendered without exceptions")
"""


def test_streamlit_entrypoint_runs_without_repo_root_on_path(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["CONTEXTIQ_API_URL"] = "http://127.0.0.1:9"  # closed port: the UI must degrade gracefully
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", RUNNER, str(SCRIPT)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr[-2000:]}"
    assert "OK: rendered without exceptions" in result.stdout
