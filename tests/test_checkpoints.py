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
