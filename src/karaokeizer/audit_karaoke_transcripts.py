"""Audit Whisper karaoke transcripts for timing defects that make lyrics unreadable."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def audit_transcript(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    failures: list[dict] = []
    warnings: list[dict] = []
    previous_word: dict | None = None
    for segment_index, segment in enumerate(data.get("segments", [])):
        words = segment.get("words", [])
        segment_start = float(segment["start"])
        segment_end = float(segment["end"])
        duration = max(0.0, segment_end - segment_start)
        if duration and len(words) / duration > 5.5:
            warnings.append({"type": "dense_segment", "segment": segment_index, "words_per_second": round(len(words) / duration, 3)})
        starts = Counter(round(float(word["start"]), 3) for word in words)
        for start, count in starts.items():
            if count >= 2:
                failures.append({"type": "duplicate_word_start", "segment": segment_index, "start": start, "count": count})
        for word_index, word in enumerate(words):
            start = float(word["start"])
            end = float(word["end"])
            word_ref = {"segment": segment_index, "word": word_index, "text": word.get("text", ""), "start": start, "end": end}
            if end - start <= 0.02:
                failures.append({"type": "collapsed_word", **word_ref})
            elif end - start < 0.06:
                warnings.append({"type": "very_short_word", **word_ref})
            if start < segment_start - 0.12 or end > segment_end + 0.12:
                failures.append({"type": "word_outside_segment", **word_ref, "segment_start": segment_start, "segment_end": segment_end})
            if previous_word:
                if start < float(previous_word["start"]) - 0.02:
                    failures.append({"type": "non_monotonic_start", **word_ref, "previous": previous_word.get("text", "")})
                overlap = float(previous_word["end"]) - start
                if overlap > 0.12:
                    failures.append({"type": "word_overlap", **word_ref, "overlap": round(overlap, 3), "previous": previous_word.get("text", "")})
            previous_word = word
    return {
        "path": str(path),
        "track": data.get("track"),
        "title": data.get("title"),
        "segments": len(data.get("segments", [])),
        "failures": failures,
        "warnings": warnings,
        "pass": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Transcript JSON file or directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true", help="Return nonzero when timing failures are found")
    args = parser.parse_args()
    paths = sorted(args.path.glob("*.whisper.json")) if args.path.is_dir() else [args.path]
    results = [audit_transcript(path) for path in paths]
    report = {
        "schema": "karaoke-transcript-audit/v1",
        "files": len(results),
        "passing": sum(result["pass"] for result in results),
        "failures": sum(len(result["failures"]) for result in results),
        "results": results,
        "pass": all(result["pass"] for result in results),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.output:
        print(json.dumps({"output": str(args.output), "files": report["files"], "failures": report["failures"], "pass": report["pass"]}))
    else:
        print(rendered)
    return 1 if args.strict and not report["pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
