"""Fail CI when tracked files look like private/generated media or secrets."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

FORBIDDEN_SUFFIXES = {
    ".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus",
    ".safetensors", ".pt", ".pth", ".m3u", ".m3u8",
}
FORBIDDEN_NAMES = {
    "pipeline-manifest.json", "verification-report.json", "karaoke-verification.json",
    "album-qa-ledger.json", "player-visual-audit.json", "transcript-consensus.json",
    "transcription-index.json", "remediation-summary.json",
}
FORBIDDEN_NAME_PATTERNS = (
    re.compile(r".* - (?:Karaoke|Instrumental)\.html$", re.IGNORECASE),
    re.compile(r".*\.(?:whisper|karaoke)\.json$", re.IGNORECASE),
    re.compile(r"cover\.(?:jpe?g|png|webp)$", re.IGNORECASE),
)
TEXT_PATTERNS = {
    "private Windows path": re.compile(r"[A-Za-z]:[\\/](?:Users|Music|Downloads|Documents|Desktop)[\\/][^\s<]+", re.IGNORECASE),
    "private POSIX home path": re.compile(r"/(?:home|Users)/[^/\s<]+/[^\s<]+"),
    "GitHub token": re.compile(r"\b(?:gho|ghp|github_pat)_[A-Za-z0-9_]+"),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
}
MAX_TRACKED_BYTES = 2 * 1024 * 1024


def main() -> int:
    names = subprocess.check_output(["git", "ls-files", "-z"]).decode("utf-8").split("\0")
    failures = []
    for name in filter(None, names):
        path = Path(name)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden tracked artifact: {name}")
        if path.name.lower() in FORBIDDEN_NAMES or any(pattern.fullmatch(path.name) for pattern in FORBIDDEN_NAME_PATTERNS):
            failures.append(f"forbidden generated artifact: {name}")
        if path.stat().st_size > MAX_TRACKED_BYTES:
            failures.append(f"tracked file exceeds 2 MiB: {name}")
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in TEXT_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{label} in {name}")
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
