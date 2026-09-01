"""Fail when public files contain study identity, private paths, secrets, or bundles."""

from __future__ import annotations

from pathlib import Path
import os
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".audit", ".git", ".pytest_cache", "__pycache__", ".venv", "dist", "build"}
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
    unix_profile = r"/(?:" + "home|Users" + r")/[^/\s]+"
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


def audit() -> list[str]:
    findings: list[str] = []
    patterns = _patterns()
    for path in iter_public_files():
        relative = path.relative_to(ROOT).as_posix()
        suffixes = {suffix.lower() for suffix in path.suffixes}
        if suffixes.intersection(FORBIDDEN_BINARY_SUFFIXES):
            findings.append(f"{relative}: forbidden binary/archive type")
            continue
        if path.stat().st_size > 5 * 1024 * 1024:
            findings.append(f"{relative}: file exceeds the 5 MiB public-tree limit")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"{relative}: non-UTF-8 content in a text-like file")
            continue
        for label, pattern in patterns.items():
            match = pattern.search(content)
            if match is not None:
                line = content.count("\n", 0, match.start()) + 1
                findings.append(f"{relative}:{line}: {label}")
    return findings


def main() -> int:
    findings = audit()
    if findings:
        print("Public content audit failed:")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print(f"Public content audit passed ({len(iter_public_files())} files checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
