"""Convert transcript-audit failures into a minimal re-transcription ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def merged_regions(failures: list[dict], padding: float, gap: float) -> list[dict]:
    spans = []
    for failure in failures:
        start = float(failure.get("start", failure.get("segment_start", 0)))
        end = float(failure.get("end", start + 0.5))
        spans.append([max(0.0, start - padding), end + padding])
    spans.sort()
    merged: list[list[float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [
        {"start": round(start, 3), "end": round(end, 3), "reason": "failed_candidate_gate", "action": "retranscribe"}
        for start, end in merged
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--padding", type=float, default=1.5)
    parser.add_argument("--merge-gap", type=float, default=1.0)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    tracks = []
    for result in audit["results"]:
        regions = merged_regions(result.get("failures", []), args.padding, args.merge_gap)
        tracks.append({
            "track": int(result["track"]), "title": result["title"],
            "remediation_regions": regions,
        })
    payload = {"schema": "karaoke-escalation-ledger/v1", "source_audit": str(args.audit), "tracks": tracks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "tracks": sum(bool(t["remediation_regions"]) for t in tracks), "regions": sum(len(t["remediation_regions"]) for t in tracks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
