"""Small standard-library compatibility helpers for supported Python versions."""

from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:  # Python 3.10
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport of the string behavior needed by AlloGate enums."""

        def __str__(self) -> str:
            return str(self.value)


__all__ = ["StrEnum"]
