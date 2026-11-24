"""Compare two album transcript passes and flag lexical disagreement for review."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def normalized_words(data: dict, start: float | None = None, end: float | None = None) -> list[str]:
    return [
        normalized
        for segment in data.get("segments", [])
        for word in segment.get("words", [])
        if (start is None or float(word["end"]) > start) and (end is None or float(word["start"]) < end)
        if (normalized := re.sub(r"[^a-z0-9']", "", str(word.get("text", "")).lower()))
    ]


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for row, left_word in enumerate(left, 1):
        current = [row]
        for column, right_word in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (left_word != right_word)))
        previous = current
    return previous[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-disagreement", type=float, default=0.08)
    args = parser.parse_args()
    rows = []
    for primary_path in sorted(args.primary.glob("*.whisper.json")):
        candidate_path = args.candidate / primary_path.name
        if not candidate_path.exists():
            rows.append({"file": primary_path.name, "pass": False, "failure": "candidate_missing"})
            continue
        primary_data = json.loads(primary_path.read_text(encoding="utf-8"))
        candidate_data = json.loads(candidate_path.read_text(encoding="utf-8"))
        primary_words, candidate_words = normalized_words(primary_data), normalized_words(candidate_data)
        primary_instrumental = (primary_data.get("lyric_classification") or {}).get("status") == "instrumental_or_no_reliable_vocal"
        candidate_instrumental = (candidate_data.get("lyric_classification") or {}).get("status") == "instrumental_or_no_reliable_vocal"
        distance = edit_distance(primary_words, candidate_words)
        disagreement = distance / max(1, max(len(primary_words), len(candidate_words)))
        remediations = candidate_data.get("qa_remediation", [])
        regions = remediations[-1].get("regions", []) if remediations else []
        region_rows = []
        for region in regions:
            if region.get("action") == "drop_hallucination":
                region_rows.append({**region, "disagreement_ratio": 0.0, "pass": True, "note": "low-energy hallucination removed"})
                continue
            start, end = float(region["start"]), float(region["end"])
            left, right = normalized_words(primary_data, start, end), normalized_words(candidate_data, start, end)
            region_distance = edit_distance(left, right)
            region_disagreement = region_distance / max(1, max(len(left), len(right)))
            region_rows.append({
                **region, "primary_words": len(left), "candidate_words": len(right),
                "edit_distance": region_distance, "disagreement_ratio": round(region_disagreement, 5),
                "pass": region_disagreement <= args.max_disagreement,
            })
        content_valid = bool(primary_words and candidate_words) or (primary_instrumental and candidate_instrumental)
        regional_pass = (all(region["pass"] for region in region_rows) if region_rows else disagreement <= args.max_disagreement) and content_valid
        rows.append({
            "file": primary_path.name,
            "primary_sha256": sha256(primary_path),
            "candidate_sha256": sha256(candidate_path),
            "primary_words": len(primary_words),
            "candidate_words": len(candidate_words),
            "edit_distance": distance,
            "disagreement_ratio": round(disagreement, 5),
            "regions": region_rows,
            "requires_lyric_review": not (primary_instrumental and candidate_instrumental),
            "failure": None if content_valid else "empty_transcripts_without_instrumental_classification",
            "pass": regional_pass,
        })
    report = {
        "schema": "karaoke-transcript-consensus/v1",
        "primary": str(args.primary), "candidate": str(args.candidate),
        "tracks": rows, "passing_tracks": sum(row["pass"] for row in rows), "track_count": len(rows),
        "pass": all(row["pass"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "passing_tracks": report["passing_tracks"], "track_count": report["track_count"], "pass": report["pass"]}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
