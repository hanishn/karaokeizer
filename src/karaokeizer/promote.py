"""Atomically promote a rendered candidate only after all bound evidence passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path


def sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file without loading it whole."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_passing(path: Path, label: str) -> dict:
    """Load a JSON evidence ledger and require an explicit boolean PASS."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("pass") is not True:
        raise ValueError(f"{label} report is not PASS: {path}")
    return payload


def candidate_artifacts(candidate: Path, verification: dict) -> list[tuple[Path, str]]:
    """Resolve and re-hash candidate artifacts while enforcing root containment."""
    artifacts = []
    for item in verification.get("artifacts", []):
        relative = Path(item["path"])
        source = (candidate / relative).resolve(strict=True)
        if relative.is_absolute() or not source.is_relative_to(candidate):
            raise ValueError(f"candidate artifact escapes root: {relative}")
        actual = sha256(source)
        if actual != item["sha256"]:
            raise ValueError(f"candidate artifact hash mismatch: {relative}")
        artifacts.append((source, relative.as_posix()))
    if not artifacts:
        raise ValueError("candidate verification contains no artifacts")
    return artifacts


def main() -> int:
    """Validate all evidence bindings and atomically promote the candidate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--canonical-dir", required=True, type=Path)
    parser.add_argument("--transcripts", required=True, type=Path)
    parser.add_argument("--album-ledger", required=True, type=Path)
    parser.add_argument("--visual-ledger", required=True, type=Path)
    parser.add_argument("--consensus-ledger", required=True, type=Path)
    parser.add_argument("--lyric-review", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    candidate = args.candidate_dir.resolve(strict=True)
    canonical = args.canonical_dir.resolve()
    verification_path = candidate / "karaoke-verification.json"
    verification = load_passing(verification_path, "rendered candidate")
    album = load_passing(args.album_ledger, "album")
    visual = load_passing(args.visual_ledger, "visual")
    consensus = load_passing(args.consensus_ledger, "consensus")
    lyric_review = load_passing(args.lyric_review, "lyric review")
    artifacts = candidate_artifacts(candidate, verification)

    player_relative = verification.get("player")
    player_item = next((item for item in verification["artifacts"] if item["path"] == player_relative), None)
    if not player_item or visual.get("player_sha256") != player_item["sha256"]:
        raise ValueError("visual report is not bound to the rendered candidate player")

    consensus_by_file = {row["file"]: row for row in consensus["tracks"]}
    reviews = lyric_review.get("tracks", {})
    for row in album["tracks"]:
        number = int(row["track"])
        data_relative = f"karaoke/data/{number:02d}.karaoke.json"
        data_path = candidate / data_relative
        if sha256(data_path) != row["evidence"]["karaoke_data_sha256"]:
            raise ValueError(f"album ledger karaoke-data mismatch for track {number}")
        transcript_candidates = sorted(args.transcripts.glob(f"{number:02d} - *.whisper.json"))
        if len(transcript_candidates) != 1:
            raise ValueError(f"expected exactly one transcript for track {number}")
        transcript_path = transcript_candidates[0]
        transcript_hash = sha256(transcript_path)
        if transcript_hash != row["evidence"]["transcript_sha256"]:
            raise ValueError(f"album ledger transcript mismatch for track {number}")
        consensus_row = consensus_by_file.get(transcript_path.name)
        if not consensus_row or consensus_row.get("candidate_sha256") != transcript_hash:
            raise ValueError(f"consensus ledger mismatch for track {number}")
        review = reviews.get(str(number), {})
        if review.get("status") != "pass" or review.get("transcript_sha256") != transcript_hash:
            raise ValueError(f"lyric review missing or stale for track {number}")

    promoted = []
    canonical.mkdir(parents=True, exist_ok=True)
    for source, relative_text in artifacts:
        relative = Path(relative_text)
        destination = canonical / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        promoted.append({"path": relative.as_posix(), "sha256": sha256(destination)})

    receipt = {
        "schema": "karaokeizer-promotion/v1",
        "pass": True,
        "promoted_utc": datetime.now(UTC).isoformat(),
        "candidate_verification_sha256": sha256(verification_path),
        "album_ledger_sha256": sha256(args.album_ledger),
        "visual_ledger_sha256": sha256(args.visual_ledger),
        "consensus_ledger_sha256": sha256(args.consensus_ledger),
        "lyric_review_sha256": sha256(args.lyric_review),
        "artifacts": promoted,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
