"""Create and verify a bounded manifest for an SSLI browser distribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path

SCHEMA = "karaokeizer-ssli-distribution/v1"
MANIFEST_NAME = "karaokeizer-ssli-manifest.json"
ALLOWED_SUFFIXES = {
    ".css", ".gif", ".html", ".jpeg", ".jpg", ".js", ".json", ".mp3",
    ".ogg", ".png", ".svg", ".ttf", ".wasm", ".wav", ".webp", ".woff", ".woff2",
}
MAX_FILES = 10_000
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_TOTAL_BYTES = 500 * 1024 * 1024


def sha256(path: Path) -> str:
    """Hash one regular distribution file without loading it whole."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_link_or_reparse(path: Path) -> bool:
    """Detect POSIX links and Windows junction/reparse entries."""

    metadata = path.lstat()
    attributes = getattr(metadata, "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def distribution_files(root: Path) -> list[Path]:
    """Return allowed regular files, rejecting hidden, linked, or oversized input."""

    root = root.resolve(strict=True)
    files: list[Path] = []
    total = 0
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in list(directories):
            path = current_path / name
            if name.startswith(".") or is_link_or_reparse(path):
                raise ValueError(f"SSLI distribution contains hidden or linked directory: {path.relative_to(root)}")
        for name in names:
            path = current_path / name
            relative = path.relative_to(root)
            if name == MANIFEST_NAME:
                continue
            if name.startswith(".") or is_link_or_reparse(path) or not path.is_file():
                raise ValueError(f"SSLI distribution contains hidden, linked, or special file: {relative}")
            if path.suffix.lower() not in ALLOWED_SUFFIXES:
                raise ValueError(f"SSLI distribution contains unexpected file type: {relative}")
            size = path.stat().st_size
            if size > MAX_FILE_BYTES:
                raise ValueError(f"SSLI distribution file exceeds 50 MiB: {relative}")
            total += size
            files.append(path)
            if len(files) > MAX_FILES or total > MAX_TOTAL_BYTES:
                raise ValueError("SSLI distribution exceeds file-count or total-size limit")
    if not (root / "index.html").is_file() or (root / "index.html") not in files:
        raise ValueError("SSLI distribution must contain a regular index.html")
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def build_manifest(root: Path) -> dict:
    """Build the deterministic allowlist and hashes for a reviewed distribution."""

    root = root.resolve(strict=True)
    return {
        "schema": SCHEMA,
        "files": [
            {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in distribution_files(root)
        ],
    }


def validate_manifest(root: Path) -> dict:
    """Require the on-disk distribution to exactly match its allowlist manifest."""

    root = root.resolve(strict=True)
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("SSLI distribution manifest has an unexpected schema")
    actual = build_manifest(root)
    if manifest != actual:
        raise ValueError("SSLI distribution does not exactly match its reviewed manifest")
    return manifest


def main() -> int:
    """Write a reviewed-distribution manifest after all safety checks pass."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    manifest = build_manifest(args.root)
    destination = args.root / MANIFEST_NAME
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
