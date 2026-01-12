from __future__ import annotations

from pathlib import Path
from typing import Any


class Policy:
    """Stub policy container."""

    def __init__(self, name: str = "default"):
        self.name = name

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name}


def _parse_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value:
        return ""

    if value.lower() in {"true", "false"}:
        return value.lower() == "true"

    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def load_policy(path: str | Path) -> dict[str, Any]:
    policy_path = Path(path)
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    data: dict[str, Any] = {}
    for line in policy_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if ":" not in stripped:
            continue

        key, raw_value = stripped.split(":", 1)
        data[key.strip()] = _parse_scalar(raw_value)

    return data
