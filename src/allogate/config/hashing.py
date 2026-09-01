"""Deterministic hashing for public scientific configuration."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from typing import Any


def _plain_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported value in a scientific identity: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible scientific settings with stable ordering."""

    return json.dumps(
        _plain_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_digest(value: Any) -> str:
    """Return the SHA-256 identity of canonical scientific settings."""

    return sha256(canonical_json(value).encode("utf-8")).hexdigest()

