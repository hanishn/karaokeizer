import json
import os
import sys
from pathlib import Path

import pytest

from karaokeizer.ssli_manifest import MANIFEST_NAME, build_manifest, validate_manifest

COMPATIBLE_INDEX = "<script>formant.scheduleNote = formant.cancelSchedule = function () {}</script>"


def write_manifest(root: Path) -> None:
    (root / MANIFEST_NAME).write_text(json.dumps(build_manifest(root)), encoding="utf-8")


def test_manifest_binds_exact_reviewed_distribution(tmp_path: Path):
    (tmp_path / "index.html").write_text(COMPATIBLE_INDEX, encoding="utf-8")
    (tmp_path / "engine.js").write_text("export const ready=true", encoding="utf-8")
    write_manifest(tmp_path)

    assert len(validate_manifest(tmp_path)["files"]) == 2
    (tmp_path / "engine.js").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="does not exactly match"):
        validate_manifest(tmp_path)


def test_manifest_rejects_hidden_secret_file(tmp_path: Path):
    (tmp_path / "index.html").write_text(COMPATIBLE_INDEX, encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=secret", encoding="utf-8")

    with pytest.raises(ValueError, match="hidden"):
        build_manifest(tmp_path)


def test_manifest_rejects_symlink_or_junction(tmp_path: Path):
    (tmp_path / "index.html").write_text(COMPATIBLE_INDEX, encoding="utf-8")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("private", encoding="utf-8")
    link = tmp_path / "private.js"
    try:
        os.symlink(outside, link)
    except OSError:
        pytest.skip("creating symlinks is not permitted on this host")

    with pytest.raises(ValueError, match="linked"):
        build_manifest(tmp_path)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file attributes are platform-specific")
def test_manifest_rejects_windows_hidden_attribute(tmp_path: Path):
    import ctypes

    index = tmp_path / "index.html"
    index.write_text(COMPATIBLE_INDEX, encoding="utf-8")
    secret = tmp_path / "credentials.json"
    secret.write_text('{"token":"private"}', encoding="utf-8")
    hidden = 0x2
    normal = 0x80
    assert ctypes.windll.kernel32.SetFileAttributesW(str(secret), hidden)
    try:
        with pytest.raises(ValueError, match="hidden|unsafe"):
            build_manifest(tmp_path)
    finally:
        ctypes.windll.kernel32.SetFileAttributesW(str(secret), normal)


def test_manifest_rejects_immediate_only_formant_engine(tmp_path: Path):
    (tmp_path / "index.html").write_text("<script>formant.noteOn = function () {}</script>", encoding="utf-8")

    with pytest.raises(ValueError, match="timestamped formant scheduling API"):
        build_manifest(tmp_path)
