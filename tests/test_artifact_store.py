import json
from pathlib import Path

import pytest

from allogate.artifacts import ContentAddressedStore


def test_content_addressed_store_deduplicates_and_uses_path_free_refs(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "store")
    first = store.put_json({"value": 3, "label": "example"}, logical_type="evaluation")
    second = store.put_json({"label": "example", "value": 3}, logical_type="evaluation")
    assert first.digest == second.digest
    ref_path = store.bind("paper/evaluation", first)
    observed, object_path = store.resolve("paper/evaluation")
    assert observed == first
    assert object_path.read_text(encoding="utf-8") == '{"label":"example","value":3}'
    payload = json.loads(ref_path.read_text(encoding="utf-8"))
    assert set(payload) == {"schema_version", "logical_name", "artifact"}
    assert payload["logical_name"] == "paper/evaluation"


def test_logical_names_reject_absolute_or_traversing_paths(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "store")
    artifact = store.put_bytes(b"value", logical_type="test", media_type="text/plain")
    for logical_name in ("../escape", "/absolute", "folder\\file"):
        with pytest.raises(ValueError):
            store.bind(logical_name, artifact)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "allogate.artifact_ref.v0", "schema version"),
        ("logical_name", "paper/other", "logical name"),
    ],
)
def test_resolve_rejects_tampered_reference_identity(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    store = ContentAddressedStore(tmp_path / "store")
    artifact = store.put_bytes(b"value", logical_type="test", media_type="text/plain")
    ref_path = store.bind("paper/value", artifact)
    payload = json.loads(ref_path.read_text(encoding="utf-8"))
    payload[field] = value
    ref_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        store.resolve("paper/value")


def test_resolve_rejects_tampered_byte_count(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "store")
    artifact = store.put_bytes(b"value", logical_type="test", media_type="text/plain")
    ref_path = store.bind("paper/value", artifact)
    payload = json.loads(ref_path.read_text(encoding="utf-8"))
    payload["artifact"]["byte_count"] += 1
    ref_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="byte count"):
        store.resolve("paper/value")
