"""Create a new transcript candidate with specified tracks explicitly lyric-free."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tracks", required=True, help="Comma-separated track numbers")
    parser.add_argument("--evidence", required=True, help="Audit/evidence statement supporting classification")
    args = parser.parse_args()
    tracks = {int(value) for value in args.tracks.split(",") if value.strip()}
    if args.output_dir.exists() and any(args.output_dir.glob("*.whisper.json")):
        parser.error("output directory already contains transcript candidates; use a fresh versioned directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    changed = []
    for source in args.transcripts.glob("*.whisper.json"):
        payload = json.loads(source.read_text(encoding="utf-8"))
        track = int(payload.get("track", int(source.name[:2])))
        if track in tracks:
            payload["segments"] = []
            payload["text"] = ""
            payload["word_count"] = 0
            payload["mean_word_probability"] = 0
            payload["low_confidence_words"] = []
            payload["repetition_ratio"] = 0
            payload["lyric_classification"] = {
                "status": "instrumental_or_no_reliable_vocal",
                "created_utc": datetime.now(UTC).isoformat(),
                "evidence": args.evidence,
            }
            changed.append(track)
        (args.output_dir / source.name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    summary = {"schema": "karaoke-lyric-classification/v1", "tracks": sorted(changed), "evidence": args.evidence}
    (args.output_dir / "classification-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir), "classified_tracks": sorted(changed)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
