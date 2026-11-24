import json

import pytest

from karaokeizer import instrumental_album_pipeline as pipeline


def test_separation_checkpoint_rejects_changed_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    work = tmp_path / "work"
    source.mkdir()
    track = source / "Album - 01 Song.wav"
    track.write_bytes(b"first")
    config = {
        "source_dir": str(source), "source_glob": "*.wav",
        "track_regex": r" - (?P<number>\d{2}) (?P<title>.+)\.wav$",
        "work_dir": str(work), "demucs_model": "fixture",
    }
    destination = work / "fixture" / track.stem
    destination.mkdir(parents=True)
    vocal = destination / "vocals.wav"
    accompaniment = destination / "no_vocals.wav"
    vocal.write_bytes(b"vocal")
    accompaniment.write_bytes(b"music")
    monkeypatch.setattr(pipeline.importlib.metadata, "version", lambda _: "1.0")
    recipe = pipeline.separation_recipe(config, track)
    (destination / "karaokeizer-checkpoint.json").write_text(json.dumps({
        "recipe": recipe,
        "outputs": {"vocals.wav": pipeline.sha256(vocal), "no_vocals.wav": pipeline.sha256(accompaniment)},
    }), encoding="utf-8")
    track.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="stale or unverifiable"):
        pipeline.separate(config, "python")


def test_package_recipe_changes_when_stem_changes(tmp_path):
    stem, cover = tmp_path / "no_vocals.wav", tmp_path / "cover.jpg"
    stem.write_bytes(b"first stem")
    cover.write_bytes(b"cover")
    config = {
        "artist": "Human Artist", "album": "Album", "album_suffix": " (Instrumental)",
        "year": "2026", "genre": "Synthwave",
    }
    row = {"number": 1, "title": "Song"}
    first = pipeline.package_recipe(config, row, stem, cover, "ffmpeg fixture")
    stem.write_bytes(b"regenerated stem")
    second = pipeline.package_recipe(config, row, stem, cover, "ffmpeg fixture")

    assert first != second
    assert first["accompaniment_sha256"] != second["accompaniment_sha256"]
