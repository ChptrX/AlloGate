"""Small content-addressed store whose public references never embed source paths."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from allogate.config.hashing import canonical_json


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REF_SCHEMA = "allogate.artifact_ref.v1"
_REF_KEYS = {"schema_version", "logical_name", "artifact"}
_ARTIFACT_KEYS = {"digest", "logical_type", "media_type", "byte_count"}


def _logical_parts(logical_name: str) -> tuple[str, ...]:
    if "\\" in logical_name or logical_name.startswith("/"):
        raise ValueError("logical artifact names must be portable relative paths")
    parts = tuple(logical_name.split("/"))
    if not parts or any(_SEGMENT.fullmatch(part) is None for part in parts):
        raise ValueError("logical artifact names contain an invalid path segment")
    return parts


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    digest: str
    logical_type: str
    media_type: str
    byte_count: int

    def __post_init__(self) -> None:
        if _DIGEST.fullmatch(self.digest) is None:
            raise ValueError("artifact digest must be lowercase SHA-256")
        if not self.logical_type or not self.media_type or self.byte_count < 0:
            raise ValueError("artifact metadata is incomplete")

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "logical_type": self.logical_type,
            "media_type": self.media_type,
            "byte_count": self.byte_count,
        }


class ContentAddressedStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        (self.root / "objects" / "sha256").mkdir(parents=True, exist_ok=True)
        (self.root / "refs").mkdir(parents=True, exist_ok=True)

    def _object_path(self, digest: str) -> Path:
        return self.root / "objects" / "sha256" / digest[:2] / digest[2:]

    @staticmethod
    def _atomic_write(destination: Path, payload: bytes) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".allogate-", dir=destination.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def put_bytes(self, payload: bytes, *, logical_type: str, media_type: str) -> ArtifactRef:
        digest = sha256(payload).hexdigest()
        destination = self._object_path(digest)
        if destination.exists():
            if sha256(destination.read_bytes()).hexdigest() != digest:
                raise RuntimeError("content-addressed object failed integrity verification")
        else:
            self._atomic_write(destination, payload)
        return ArtifactRef(digest, logical_type, media_type, len(payload))

    def put_json(self, value: Any, *, logical_type: str) -> ArtifactRef:
        payload = canonical_json(value).encode("utf-8")
        return self.put_bytes(payload, logical_type=logical_type, media_type="application/json")

    def put_file(
        self,
        source: str | Path,
        *,
        logical_type: str,
        media_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        payload = Path(source).read_bytes()
        return self.put_bytes(payload, logical_type=logical_type, media_type=media_type)

    def bind(self, logical_name: str, artifact: ArtifactRef) -> Path:
        parts = _logical_parts(logical_name)
        object_path = self._object_path(artifact.digest)
        if not object_path.is_file() or object_path.stat().st_size != artifact.byte_count:
            raise RuntimeError("cannot bind a missing artifact or one with an invalid byte count")
        if sha256(object_path.read_bytes()).hexdigest() != artifact.digest:
            raise RuntimeError("cannot bind a missing or corrupt artifact")
        destination = self.root / "refs" / Path(*parts).with_suffix(".json")
        payload = canonical_json(
            {
                "schema_version": _REF_SCHEMA,
                "logical_name": "/".join(parts),
                "artifact": artifact.to_dict(),
            }
        ).encode("utf-8")
        self._atomic_write(destination, payload)
        return destination

    def resolve(self, logical_name: str) -> tuple[ArtifactRef, Path]:
        parts = _logical_parts(logical_name)
        normalized_name = "/".join(parts)
        ref_path = self.root / "refs" / Path(*parts).with_suffix(".json")
        payload = json.loads(ref_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != _REF_KEYS:
            raise ValueError("artifact reference has an invalid top-level structure")
        if payload["schema_version"] != _REF_SCHEMA:
            raise ValueError("artifact reference uses an unsupported schema version")
        if payload["logical_name"] != normalized_name:
            raise ValueError("artifact reference logical name does not match its binding")
        record = payload["artifact"]
        if not isinstance(record, dict) or set(record) != _ARTIFACT_KEYS:
            raise ValueError("artifact reference has an invalid artifact record")
        if (
            not isinstance(record["digest"], str)
            or not isinstance(record["logical_type"], str)
            or not isinstance(record["media_type"], str)
            or isinstance(record["byte_count"], bool)
            or not isinstance(record["byte_count"], int)
        ):
            raise ValueError("artifact reference metadata has invalid field types")
        artifact = ArtifactRef(
            digest=record["digest"],
            logical_type=record["logical_type"],
            media_type=record["media_type"],
            byte_count=record["byte_count"],
        )
        object_path = self._object_path(artifact.digest)
        if not object_path.is_file() or object_path.stat().st_size != artifact.byte_count:
            raise RuntimeError("resolved artifact is missing or has an invalid byte count")
        if sha256(object_path.read_bytes()).hexdigest() != artifact.digest:
            raise RuntimeError("resolved artifact is missing or corrupt")
        return artifact, object_path
