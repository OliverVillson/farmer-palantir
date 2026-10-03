"""Environment knobs. FARMPAL_<NAME> wins; LOBBOT_<NAME> is read as a fallback so
setups made for the LobBot repo, where this code started, keep working."""

from __future__ import annotations

import os


def getenv(name: str, default: str | None = None) -> str | None:
    v = os.environ.get("FARMPAL_" + name)
    if v is None:
        v = os.environ.get("LOBBOT_" + name)
    return default if v is None else v
