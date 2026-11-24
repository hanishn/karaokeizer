from pathlib import Path

import pytest

from karaokeizer.instrumental_album_pipeline import source_tracks


def config_for(source_dir: Path, expression: str) -> dict:
    """Return the minimal source-discovery configuration used by these tests."""

    return {
        "source_dir": str(source_dir),
        "source_glob": "*.wav",
        "track_regex": expression,
    }


def test_source_tracks_extracts_number_and_title(tmp_path: Path):
    (tmp_path / "Album - 01 Opening.wav").touch()
    tracks = source_tracks(config_for(tmp_path, r" - (?P<number>\d{2}) (?P<title>.+)\.wav$"))

    assert [(track["number"], track["title"]) for track in tracks] == [(1, "Opening")]


def test_source_tracks_rejects_unreasonably_large_expression(tmp_path: Path):
    with pytest.raises(ValueError, match="1,000"):
        source_tracks(config_for(tmp_path, "a" * 1_001))
