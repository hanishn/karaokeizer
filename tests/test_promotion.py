import json
import sys
from pathlib import Path

import pytest

from karaokeizer import promote
from karaokeizer.promote import main, promote_tree, sha256


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
    audio = candidate / "media" / "01 - Song.flac"
    analysis = tmp_path / "analysis.json"
    for path, content in ((player, "<html>safe</html>"), (data, "{}"), (transcript, "{}")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"synthetic-audio-fixture")
    write_json(analysis, {"schema": "karaokeizer-analysis-overrides/v1", "pass": True})

    verification = candidate / "karaoke-verification.json"
    write_json(
        verification,
        {
            "schema": "karaoke-overlay/v1",
            "pass": True,
            "tracks_expected": 1,
            "tracks_built": 1,
            "album_manifest_sha256": "manifest-hash",
            "player": player.name,
            "artifacts": [
                {"path": player.name, "sha256": sha256(player)},
                {"path": "karaoke/data/01.karaoke.json", "sha256": sha256(data)},
                {"path": "media/01 - Song.flac", "sha256": sha256(audio)},
            ],
            "analysis_review_sha256": sha256(analysis),
        },
    )
    album = tmp_path / "album.json"
    visual = tmp_path / "visual.json"
    consensus = tmp_path / "consensus.json"
    review = tmp_path / "review.json"
    write_json(
        album,
        {
            "schema": "karaoke-album-qa/v1",
            "pass": True,
            "track_count": 1,
            "album_manifest_sha256": "manifest-hash",
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
    write_json(
        visual,
        {
            "schema": "karaoke-player-visual-audit/v1", "pass": True,
            "track_count": 1, "passing_tracks": 1, "player_sha256": sha256(player),
        },
    )
    write_json(
        consensus,
        {
            "schema": "karaoke-transcript-consensus/v1",
            "pass": True,
            "tracks": [{"file": transcript.name, "candidate_sha256": sha256(transcript)}],
        },
    )
    write_json(
        review,
        {
            "schema": "karaokeizer-lyric-review/v1",
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
        "analysis": analysis,
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
        ("--analysis-review", "analysis"),
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
    write_json(
        fixture["review"],
        {"schema": "karaokeizer-lyric-review/v1", "pass": False, "tracks": {}},
    )

    with pytest.raises(ValueError, match="lyric review report is not PASS"):
        invoke(monkeypatch, fixture)
    assert not fixture["canonical"].exists()


def test_promotion_rejects_mutated_candidate_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = promotion_fixture(tmp_path)
    next((fixture["candidate"] / "media").glob("*.flac")).write_bytes(b"changed")

    with pytest.raises(ValueError, match="candidate artifact hash mismatch"):
        invoke(monkeypatch, fixture)
    assert not fixture["canonical"].exists()


def test_tree_staging_failure_leaves_existing_canonical_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    (canonical / "old.html").write_text("old", encoding="utf-8")
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    real_copy = promote.shutil.copy2
    calls = 0

    def fail_second_copy(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic copy failure")
        return real_copy(source, destination)

    monkeypatch.setattr(promote.shutil, "copy2", fail_second_copy)
    with pytest.raises(OSError, match="synthetic"):
        promote_tree(canonical, [(first, "first"), (second, "second")])

    assert (canonical / "old.html").read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".canonical.stage-*"))


def test_promotion_rejects_incomplete_album_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = promotion_fixture(tmp_path)
    album = json.loads(fixture["album"].read_text(encoding="utf-8"))
    album["tracks"] = []
    write_json(fixture["album"], album)

    with pytest.raises(ValueError, match="album ledger track inventory is incomplete"):
        invoke(monkeypatch, fixture)
    assert not fixture["canonical"].exists()


def test_promotion_rejects_unexpected_ledger_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = promotion_fixture(tmp_path)
    visual = json.loads(fixture["visual"].read_text(encoding="utf-8"))
    visual["schema"] = "lookalike/v1"
    write_json(fixture["visual"], visual)

    with pytest.raises(ValueError, match="unexpected schema"):
        invoke(monkeypatch, fixture)
