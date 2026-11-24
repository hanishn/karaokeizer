import json
import sys
from pathlib import Path

import pytest

from karaokeizer.promote import main, sha256


def write_json(path: Path, payload: dict) -> None:
    """Write a compact JSON fixture, creating its parent directory first."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def promotion_fixture(tmp_path: Path) -> dict[str, Path]:
    """Create the smallest fully bound candidate and review evidence set."""

    candidate = tmp_path / "candidate"
    transcripts = tmp_path / "transcripts"
    canonical = tmp_path / "canonical"
    player = candidate / "Album - Karaoke.html"
    data = candidate / "karaoke" / "data" / "01.karaoke.json"
    transcript = transcripts / "01 - Song.whisper.json"
    for path, content in ((player, "<html>safe</html>"), (data, "{}"), (transcript, "{}")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    verification = candidate / "karaoke-verification.json"
    write_json(
        verification,
        {
            "pass": True,
            "player": player.name,
            "artifacts": [
                {"path": player.name, "sha256": sha256(player)},
                {"path": "karaoke/data/01.karaoke.json", "sha256": sha256(data)},
            ],
        },
    )
    album = tmp_path / "album.json"
    visual = tmp_path / "visual.json"
    consensus = tmp_path / "consensus.json"
    review = tmp_path / "review.json"
    write_json(
        album,
        {
            "pass": True,
            "tracks": [
                {
                    "track": 1,
                    "evidence": {
                        "karaoke_data_sha256": sha256(data),
                        "transcript_sha256": sha256(transcript),
                    },
                }
            ],
        },
    )
    write_json(visual, {"pass": True, "player_sha256": sha256(player)})
    write_json(
        consensus,
        {
            "pass": True,
            "tracks": [{"file": transcript.name, "candidate_sha256": sha256(transcript)}],
        },
    )
    write_json(
        review,
        {
            "pass": True,
            "tracks": {"1": {"status": "pass", "transcript_sha256": sha256(transcript)}},
        },
    )
    return {
        "candidate": candidate,
        "canonical": canonical,
        "transcripts": transcripts,
        "album": album,
        "visual": visual,
        "consensus": consensus,
        "review": review,
        "receipt": tmp_path / "receipt.json",
        "player": player,
    }


def invoke(monkeypatch: pytest.MonkeyPatch, fixture: dict[str, Path]) -> int:
    """Invoke the CLI entry point with paths from a promotion fixture."""

    arguments = ["karaokeizer-promote"]
    for option, key in (
        ("--candidate-dir", "candidate"),
        ("--canonical-dir", "canonical"),
        ("--transcripts", "transcripts"),
        ("--album-ledger", "album"),
        ("--visual-ledger", "visual"),
        ("--consensus-ledger", "consensus"),
        ("--lyric-review", "review"),
        ("--receipt", "receipt"),
    ):
        arguments.extend((option, str(fixture[key])))
    monkeypatch.setattr(sys, "argv", arguments)
    return main()


def test_promotion_copies_only_hash_bound_passing_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = promotion_fixture(tmp_path)

    assert invoke(monkeypatch, fixture) == 0
    promoted_player = fixture["canonical"] / fixture["player"].name
    assert promoted_player.read_text(encoding="utf-8") == "<html>safe</html>"
    assert json.loads(fixture["receipt"].read_text(encoding="utf-8"))["pass"] is True


def test_promotion_rejects_stale_candidate_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = promotion_fixture(tmp_path)
    fixture["player"].write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        invoke(monkeypatch, fixture)
    assert not fixture["canonical"].exists()


def test_promotion_rejects_failed_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = promotion_fixture(tmp_path)
    write_json(fixture["review"], {"pass": False, "tracks": {}})

    with pytest.raises(ValueError, match="lyric review report is not PASS"):
        invoke(monkeypatch, fixture)
    assert not fixture["canonical"].exists()
