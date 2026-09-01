from io import BytesIO
from pathlib import Path
import runpy
import tarfile
import zipfile


NAMESPACE = runpy.run_path(str(Path(__file__).parents[1] / "tools" / "public_content_audit.py"))


def test_public_tree_passes_content_gate() -> None:
    assert NAMESPACE["audit"]() == []


def test_release_archive_audit_accepts_minimal_wheel_and_sdist(tmp_path: Path) -> None:
    wheel = tmp_path / "allogate-0.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as handle:
        handle.writestr("allogate/__init__.py", "__version__ = '0.1'\n")
        handle.writestr("allogate-0.1.dist-info/licenses/LICENSE", "Apache License 2.0\n")
        handle.writestr(
            "allogate-0.1.dist-info/licenses/THIRD_PARTY_NOTICES.md", "No bundled code.\n"
        )
        handle.writestr("allogate-0.1.dist-info/METADATA", "Name: allogate\nVersion: 0.1\n")

    sdist = tmp_path / "allogate-0.1.tar.gz"
    with tarfile.open(sdist, "w:gz") as handle:
        for name, content in {
            "LICENSE": b"Apache License 2.0\n",
            "THIRD_PARTY_NOTICES.md": b"No bundled code.\n",
            "README.md": b"# AlloGate\n",
            "pyproject.toml": b"[build-system]\n",
            "src/allogate/__init__.py": b"__version__ = '0.1'\n",
        }.items():
            info = tarfile.TarInfo(f"allogate-0.1/{name}")
            info.size = len(content)
            handle.addfile(info, BytesIO(content))

    assert NAMESPACE["audit_archive"](wheel) == []
    assert NAMESPACE["audit_archive"](sdist) == []


def test_release_archive_audit_rejects_unsafe_paths_and_private_content(tmp_path: Path) -> None:
    wheel = tmp_path / "allogate-0.1-py3-none-any.whl"
    study_identity = ("sp" + "cas" + "9").encode()
    with zipfile.ZipFile(wheel, "w") as handle:
        handle.writestr("../escape.txt", "unsafe\n")
        handle.writestr("allogate/private.txt", study_identity)
        handle.writestr("allogate-0.1.dist-info/licenses/LICENSE", "Apache License 2.0\n")
        handle.writestr(
            "allogate-0.1.dist-info/licenses/THIRD_PARTY_NOTICES.md", "No bundled code.\n"
        )
    findings = NAMESPACE["audit_archive"](wheel)
    assert any("unsafe archive path" in finding for finding in findings)
    assert any("study-specific identity" in finding for finding in findings)
