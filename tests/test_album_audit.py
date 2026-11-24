import numpy as np
import soundfile as sf

from karaokeizer.audit_karaoke_album import lyric_metrics, pitch_metrics, sonic_metrics, vocal_alignment


def write_audio(path, audio, sample_rate=8000):
    sf.write(path, audio, sample_rate, subtype="FLOAT")


def test_degenerate_silent_vocal_stem_cannot_pass_as_vocal_track(tmp_path):
    time = np.arange(8000) / 8000
    source_audio = (0.2 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    source, vocal, accompaniment = (tmp_path / name for name in ("source.wav", "vocal.wav", "music.wav"))
    write_audio(source, source_audio)
    write_audio(vocal, np.zeros_like(source_audio))
    write_audio(accompaniment, source_audio)
    metrics = sonic_metrics(source, vocal, accompaniment, {"status": "pass", "track_type": "vocal"})
    assert "degenerate_or_silent_vocal_stem" in metrics["failures"]
    assert not metrics["pass"]


def test_empty_lyrics_require_explicit_instrumental_classification(tmp_path):
    vocal = tmp_path / "vocal.wav"
    write_audio(vocal, np.ones(8000, dtype=np.float32) * 0.05)
    audit = {"failures": [], "warnings": []}
    assert not lyric_metrics({"segments": []}, audit)["pass"]
    assert not vocal_alignment(vocal, [], 1.0)["pass"]
    assert not pitch_metrics({"words": [], "note_events": []})["pass"]
    assert lyric_metrics({"segments": []}, audit, allow_empty=True)["pass"]
