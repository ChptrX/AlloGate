"""Fail when public files or release archives contain private or unsafe content."""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".audit", ".git", ".pytest_cache", "__pycache__", ".venv", "dist", "build"}
MAX_PUBLIC_FILE_BYTES = 5 * 1024 * 1024
TEXT_SUFFIXES = {
    "",
    ".cff",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_BINARY_SUFFIXES = {
    ".ckpt",
    ".dcd",
    ".nc",
    ".npz",
    ".pt",
    ".pth",
    ".tar",
    ".trr",
    ".whl",
    ".xtc",
    ".zip",
}


def _patterns() -> dict[str, re.Pattern[str]]:
    study_name = "sp" + "cas" + "9"
    protein_family = "cas" + "9"
    windows_profile = r"[A-Za-z]:[\\/]" + "Users" + r"[\\/][^\\/\s]+"
    unix_profile = r"/(?:(?:" + "home|Users" + r"))/[^/\s]+"
    secret_assignment = r"(?i)(?:api[_-]?key|access[_-]?token|password)\s*[:=]\s*['\"][^'\"]+"
    return {
        "study-specific identity": re.compile(rf"(?i)\b(?:{study_name}|{protein_family})\b"),
        "Windows user profile": re.compile(windows_profile),
        "Unix user profile": re.compile(unix_profile),
        "credential-like assignment": re.compile(secret_assignment),
        "private key": re.compile("BEGIN " + "(?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    }


def iter_public_files() -> list[Path]:
    files: list[Path] = []
    for directory, child_directories, names in os.walk(ROOT):
        child_directories[:] = [
            name
            for name in child_directories
            if name not in IGNORED_PARTS and not name.endswith(".egg-info")
        ]
        base = Path(directory)
        files.extend(base / name for name in names)
    return sorted(files)


def _audit_bytes(label: str, payload: bytes, findings: list[str]) -> None:
    suffixes = {suffix.lower() for suffix in PurePosixPath(label).suffixes}
    if suffixes.intersection(FORBIDDEN_BINARY_SUFFIXES):
        findings.append(f"{label}: forbidden binary/archive type")
        return
    if len(payload) > MAX_PUBLIC_FILE_BYTES:
        findings.append(f"{label}: file exceeds the 5 MiB public limit")
    if PurePosixPath(label).suffix.lower() not in TEXT_SUFFIXES:
        return
    try:
        content = payload.decode("utf-8")
    except UnicodeDecodeError:
        findings.append(f"{label}: non-UTF-8 content in a text-like file")
        return
    for description, pattern in _patterns().items():
        match = pattern.search(content)
        if match is not None:
            line = content.count("\n", 0, match.start()) + 1
            findings.append(f"{label}:{line}: {description}")


def audit() -> list[str]:
    findings: list[str] = []
    for path in iter_public_files():
        _audit_bytes(path.relative_to(ROOT).as_posix(), path.read_bytes(), findings)
    return findings


def _safe_archive_name(name: str) -> bool:
    if "\\" in name or re.match(r"^[A-Za-z]:", name):
        return False
    path = PurePosixPath(name)
    return not path.is_absolute() and bool(path.parts) and ".." not in path.parts


def _audit_archive_layout(path: Path, names: list[str], findings: list[str]) -> None:
    files = [PurePosixPath(name) for name in names if name and not name.endswith("/")]
    if path.suffix.lower() == ".whl":
        dist_info = {member.parts[0] for member in files if member.parts[0].endswith(".dist-info")}
        allowed_roots = {"allogate", *dist_info}
        if len(dist_info) != 1:
            findings.append(f"{path.name}: wheel must contain exactly one .dist-info directory")
        unexpected = sorted({member.parts[0] for member in files} - allowed_roots)
        if unexpected:
            findings.append(f"{path.name}: unexpected wheel top-level entries: {', '.join(unexpected)}")
        dist_root = next(iter(dist_info), "")
        basenames = {member.name for member in files if member.parts[0] == dist_root}
        for required in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
            if required not in basenames:
                findings.append(f"{path.name}: wheel is missing {required}")
        return

    roots = {member.parts[0] for member in files}
    if len(roots) != 1:
        findings.append(f"{path.name}: sdist must contain exactly one root directory")
        return
    root = next(iter(roots))
    top_entries = {member.parts[1] for member in files if len(member.parts) > 1}
    required = {"LICENSE", "THIRD_PARTY_NOTICES.md", "README.md", "pyproject.toml", "src"}
    missing = sorted(required - top_entries)
    if missing:
        findings.append(f"{path.name}: sdist is missing {', '.join(missing)}")


def audit_archive(archive: str | Path) -> list[str]:
    path = Path(archive)
    findings: list[str] = []
    names: list[str] = []
    if not path.is_file():
        return [f"{path}: release archive does not exist"]
    try:
        if path.suffix.lower() == ".whl":
            with zipfile.ZipFile(path) as handle:
                for member in handle.infolist():
                    names.append(member.filename)
                    if not _safe_archive_name(member.filename):
                        findings.append(f"{path.name}:{member.filename}: unsafe archive path")
                        continue
                    mode = member.external_attr >> 16
                    if stat.S_ISLNK(mode):
                        findings.append(f"{path.name}:{member.filename}: symbolic links are not allowed")
                        continue
                    if member.is_dir():
                        continue
                    if member.file_size > MAX_PUBLIC_FILE_BYTES:
                        findings.append(f"{path.name}:{member.filename}: file exceeds the 5 MiB public limit")
                        continue
                    _audit_bytes(f"{path.name}:{member.filename}", handle.read(member), findings)
        elif path.name.lower().endswith((".tar.gz", ".tgz")):
            with tarfile.open(path, mode="r:gz") as handle:
                for member in handle.getmembers():
                    names.append(member.name)
                    if not _safe_archive_name(member.name):
                        findings.append(f"{path.name}:{member.name}: unsafe archive path")
                        continue
                    if member.issym() or member.islnk():
                        findings.append(f"{path.name}:{member.name}: links are not allowed")
                        continue
                    if member.isdir():
                        continue
                    if not member.isfile():
                        findings.append(f"{path.name}:{member.name}: unsupported archive member type")
                        continue
                    if member.size > MAX_PUBLIC_FILE_BYTES:
                        findings.append(f"{path.name}:{member.name}: file exceeds the 5 MiB public limit")
                        continue
                    extracted = handle.extractfile(member)
                    if extracted is None:
                        findings.append(f"{path.name}:{member.name}: could not read archive member")
                        continue
                    _audit_bytes(f"{path.name}:{member.name}", extracted.read(), findings)
        else:
            return [f"{path}: expected a .whl, .tar.gz, or .tgz release archive"]
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as error:
        return [f"{path}: invalid release archive ({error})"]
    _audit_archive_layout(path, names, findings)
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="append", default=[], help="wheel or sdist to audit")
    arguments = parser.parse_args(argv)
    findings = audit()
    for archive in arguments.archive:
        findings.extend(audit_archive(archive))
    if findings:
        print("Public content audit failed:")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print(
        f"Public content audit passed ({len(iter_public_files())} tree files and "
        f"{len(arguments.archive)} release archives checked)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
