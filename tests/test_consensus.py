import json
import sys
from pathlib import Path

import pytest

from karaokeizer.compare_karaoke_transcripts import main


def write_transcript(path: Path, words: list[str], identity: dict, remediation: bool = False) -> None:
    """Write a minimal timed transcript accepted by the consensus command."""

    timed = [
        {"text": word, "start": index * 0.5, "end": index * 0.5 + 0.4}
        for index, word in enumerate(words)
    ]
    payload = {
        "decode_identity": identity,
        "segments": [{"start": 0, "end": max(1, len(words)), "words": timed}],
    }
    if remediation:
        payload["qa_remediation"] = [{"regions": [{"start": 0, "end": 0.5, "action": "retranscribe"}]}]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def invoke(monkeypatch: pytest.MonkeyPatch, primary: Path, candidate: Path, output: Path) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "karaokeizer-compare-transcripts",
            "--primary", str(primary),
            "--candidate", str(candidate),
            "--output", str(output),
            "--max-disagreement", "0.1",
        ],
    )
    return main()


def test_regions_cannot_hide_album_wide_disagreement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    primary, candidate = tmp_path / "primary", tmp_path / "candidate"
    write_transcript(primary / "01 - Song.whisper.json", ["same", "one", "two", "three"], {"model": "a"})
    write_transcript(candidate / "01 - Song.whisper.json", ["same", "wrong", "wrong", "wrong"], {"model": "b"}, True)

    assert invoke(monkeypatch, primary, candidate, tmp_path / "report.json") == 1
    row = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["tracks"][0]
    assert row["failure"] == "global_disagreement_exceeds_threshold"


def test_identical_run_identity_is_not_independent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    primary, candidate = tmp_path / "primary", tmp_path / "candidate"
    identity = {"backend": "openai", "model": "large-v3"}
    write_transcript(primary / "01 - Song.whisper.json", ["matching", "words"], identity)
    write_transcript(candidate / "01 - Song.whisper.json", ["matching", "words"], identity)

    assert invoke(monkeypatch, primary, candidate, tmp_path / "report.json") == 1
    row = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["tracks"][0]
    assert row["failure"] == "candidates_not_independent"


def test_device_only_difference_is_not_independent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    primary, candidate = tmp_path / "primary", tmp_path / "candidate"
    write_transcript(
        primary / "01 - Song.whisper.json", ["matching", "words"],
        {"backend": "openai", "model": "large-v3", "model_revision": "abc", "device": "cpu"},
    )
    write_transcript(
        candidate / "01 - Song.whisper.json", ["matching", "words"],
        {"backend": "openai", "model": "large-v3", "model_revision": "abc", "device": "cuda"},
    )

    assert invoke(monkeypatch, primary, candidate, tmp_path / "report.json") == 1
