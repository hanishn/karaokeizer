"""Re-transcribe one vocal-stem interval and deterministically merge it into a Whisper transcript."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from .remediate_karaoke_regions import (
    normalize_stats,
    replacement_segments,
    retained_segments_outside_region,
    validate_merged_timeline,
    validate_replacement,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Fresh candidate path; input is never overwritten")
    parser.add_argument("--vocal", required=True, type=Path)
    parser.add_argument("--start", required=True, type=float)
    parser.add_argument("--end", required=True, type=float)
    parser.add_argument("--model", default="turbo")
    parser.add_argument("--beam-size", type=int, default=8)
    args = parser.parse_args()
    import whisper
    if args.output.exists():
        parser.error("output candidate already exists; choose a fresh path")
    if args.start < 0 or args.end <= args.start:
        parser.error("segment must satisfy 0 <= start < end")

    transcript = json.loads(args.transcript.read_text(encoding="utf-8"))
    model = whisper.load_model(args.model, device="cpu")
    audio = whisper.load_audio(str(args.vocal))
    sample_rate = whisper.audio.SAMPLE_RATE
    clip = audio[int(args.start * sample_rate) : int(args.end * sample_rate)]
    result = model.transcribe(
        clip,
        language="en",
        word_timestamps=True,
        condition_on_previous_text=False,
        fp16=False,
        beam_size=args.beam_size,
        temperature=0,
    )
    replacement = replacement_segments(result, args.start)
    failures = validate_replacement(replacement, args.start, args.end)
    retained = retained_segments_outside_region(transcript["segments"], args.start, args.end)
    merged = sorted(retained + replacement, key=lambda segment: float(segment["start"]))
    failures.extend(validate_merged_timeline(merged))
    if failures:
        print(json.dumps({"pass": False, "failures": sorted(set(failures))}, indent=2))
        return 1
    transcript["segments"] = merged
    normalize_stats(transcript)
    transcript.setdefault("repairs", []).append(
        {
            "repaired_utc": datetime.now(UTC).isoformat(),
            "start": args.start,
            "end": args.end,
            "model": args.model,
            "beam_size": args.beam_size,
            "condition_on_previous_text": False,
            "replacement_segments": len(replacement),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(transcript, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pass": True, "candidate": str(args.output), "replacement_segments": len(replacement), "words": transcript["word_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
