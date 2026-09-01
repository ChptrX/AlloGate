from dataclasses import dataclass

import pytest

from allogate.config.hashing import canonical_json, stable_digest


@dataclass(frozen=True)
class Example:
    count: int
    label: str


def test_digest_is_independent_of_mapping_order() -> None:
    left = {"b": [2, 3], "a": Example(1, "x")}
    right = {"a": {"label": "x", "count": 1}, "b": [2, 3]}
    assert canonical_json(left) == canonical_json(right)
    assert stable_digest(left) == stable_digest(right)


def test_digest_rejects_runtime_objects() -> None:
    with pytest.raises(TypeError, match="unsupported value"):
        stable_digest(object())

