"""Job directory layout and subprocess helper shared by the robot stages.

A job is a directory: <job>/robotspec.json (the input), config.json (optional
overrides), data/, work/, out/, logs/ and .done/<stage> markers.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from common.env import getenv
from common.progress import emit

# Set FARMPAL_DRY_RUN=1 to walk every stage without a GPU: stages skip the
# heavy work (GR00T fine-tune, real compression) and write placeholder outputs.
DRY_RUN = getenv("DRY_RUN") == "1"


class Job:
    """A job directory. Subclasses set self.config (a dataclass) in __init__."""

    config: object

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in ("data", "work", "out", ".done"):
            (self.root / sub).mkdir(exist_ok=True)

    def overrides(self) -> dict:
        cfg_path = self.root / "config.json"
        return json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def is_done(self, stage: str) -> bool:
        return (self.root / ".done" / stage).exists()

    def mark_done(self, stage: str, info: dict | None = None) -> None:
        (self.root / ".done" / stage).write_text(json.dumps(info or {}))

    def clear_from(self, stages: list[str], start: str) -> None:
        for s in stages[stages.index(start):]:
            (self.root / ".done" / s).unlink(missing_ok=True)

    def save_config(self) -> None:
        """Record the effective config of this run. config.json stays the user's
        overrides only, so later default changes still apply on resume."""
        (self.root / "work" / "config.effective.json").write_text(json.dumps(asdict(self.config), indent=2))


def run(cmd: list[str], stage: str, cwd: str | Path | None = None, env: dict | None = None) -> None:
    """Run a command, forwarding its output as log lines (stderr is merged)."""
    emit(stage, msg="$ " + " ".join(cmd[:3]) + (" ..." if len(cmd) > 3 else ""))
    proc = subprocess.Popen(
        cmd, cwd=cwd, env={**os.environ, **(env or {})},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert proc.stdout
    for line in proc.stdout:
        sys.stdout.write("[" + stage + "] " + line)
        sys.stdout.flush()
    if proc.wait() != 0:
        raise RuntimeError(f"{cmd[0]} exited with {proc.returncode}")


def error_message(rc: int, tail: list[str]) -> str:
    """Best one-line explanation: the exception line of a traceback, else the last log line."""
    for line in reversed(tail):
        if line and not line.startswith(" ") and ("Error" in line or "Exception" in line):
            return line[:400]
    if rc < 0:
        return f"stage killed by signal {-rc}"
    return (tail[-1][:400] if tail else f"stage exited with {rc}")
