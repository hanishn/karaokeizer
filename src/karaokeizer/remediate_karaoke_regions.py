"""Batch re-transcribe QA-failed album regions with one model load and quarantined output."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path


def normalize_stats(transcript: dict) -> None:
    words = [word for segment in transcript["segments"] for word in segment.get("words", [])]
    transcript["text"] = " ".join(segment["text"] for segment in transcript["segments"])
    transcript["word_count"] = len(words)
    transcript["mean_word_probability"] = round(sum(float(word.get("probability") or 0) for word in words) / max(1, len(words)), 6)
    transcript["low_confidence_words"] = [word for word in words if float(word.get("probability") or 0) < 0.55]
    normalized = [re.sub(r"[^a-z0-9']", "", str(word.get("text", "")).lower()) for word in words]
    normalized = [word for word in normalized if word]
    transcript["repetition_ratio"] = round(max((normalized.count(word) for word in set(normalized)), default=0) / max(1, len(normalized)), 6)


def replacement_segments(result: dict, offset: float) -> list[dict]:
    replacement = []
    for segment in result["segments"]:
        words = [
            {
                "text": word["word"].strip(),
                "start": round(float(word["start"]) + offset, 3),
                "end": round(float(word["end"]) + offset, 3),
                "probability": round(float(word.get("probability") or 0), 6),
            }
            for word in segment.get("words", [])
            if word["word"].strip() and re.search(r"[A-Za-z0-9]", word["word"])
        ]
        if words:
            replacement.append({
                "start": round(float(segment["start"]) + offset, 3),
                "end": round(float(segment["end"]) + offset, 3),
                "text": segment["text"].strip(),
                "avg_logprob": round(float(segment.get("avg_logprob") or 0), 6),
                "no_speech_probability": round(float(segment.get("no_speech_prob") or 0), 6),
                "words": words,
            })
    return replacement


def validate_replacement(segments: list[dict], region_start: float, region_end: float) -> list[str]:
    """Reject a region atomically when Whisper emits malformed timing.

    This intentionally duplicates the essential structural gate locally: a bad
    candidate must never be merged merely because transcription completed.
    """
    failures: list[str] = []
    previous_start = -1.0
    for segment_index, segment in enumerate(segments):
        segment_start, segment_end = float(segment["start"]), float(segment["end"])
        if segment_end <= segment_start:
            failures.append(f"segment_{segment_index}:non_positive_duration")
        if segment_start < region_start - 0.05 or segment_end > region_end + 0.05:
            failures.append(f"segment_{segment_index}:outside_region")
        seen_starts: set[float] = set()
        for word_index, word in enumerate(segment.get("words", [])):
            start, end = float(word["start"]), float(word["end"])
            if end - start < 0.02:
                failures.append(f"segment_{segment_index}/word_{word_index}:collapsed")
            if start in seen_starts:
                failures.append(f"segment_{segment_index}/word_{word_index}:duplicate_start")
            if start < previous_start:
                failures.append(f"segment_{segment_index}/word_{word_index}:non_monotonic")
            if start < segment_start - 0.05 or end > segment_end + 0.05:
                failures.append(f"segment_{segment_index}/word_{word_index}:outside_segment")
            seen_starts.add(start)
            previous_start = start
    if not segments:
        failures.append("empty_replacement")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--album-manifest", required=True, type=Path)
    parser.add_argument("--transcripts", required=True, type=Path)
    parser.add_argument("--qa-ledger", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--skip-tracks", default="", help="Comma-separated instrumental/no-vocal track numbers")
    parser.add_argument("--resume", action="store_true", help="Explicitly resume a previously created candidate directory")
    args = parser.parse_args()
    import whisper
    album = json.loads(args.album_manifest.read_text(encoding="utf-8"))
    ledger = json.loads(args.qa_ledger.read_text(encoding="utf-8"))
    tracks = {int(track["number"]): track for track in album["tracks"]}
    skip = {int(value) for value in args.skip_tracks.split(",") if value.strip()}
    existing = list(args.output_dir.glob("*.whisper.json")) if args.output_dir.exists() else []
    if existing and not args.resume:
        parser.error(
            f"output directory already contains {len(existing)} transcript(s); "
            "use a fresh versioned directory or pass --resume explicitly"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in args.transcripts.glob("*.whisper.json"):
        destination = args.output_dir / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
    model = whisper.load_model(args.model, device="cpu")
    completed = []
    for row in ledger["tracks"]:
        track_number = int(row["track"])
        regions = row.get("remediation_regions", [])
        if track_number in skip or not regions:
            continue
        track = tracks[track_number]
        destination = args.output_dir / f"{track_number:02d} - {track['title']}.whisper.json"
        transcript = json.loads(destination.read_text(encoding="utf-8"))
        audio = whisper.load_audio(str(track["vocal_stem"]))
        sample_rate = whisper.audio.SAMPLE_RATE
        region_records = []
        for region_index, region in enumerate(regions, 1):
            start, end = float(region["start"]), float(region["end"])
            if region.get("action") == "drop_hallucination":
                transcript["segments"] = [segment for segment in transcript["segments"] if float(segment["end"]) <= start or float(segment["start"]) >= end]
                region_records.append({"start": start, "end": end, "action": "drop_hallucination", "replacement_segments": 0})
                print(f"track {track_number:02d} region {region_index}/{len(regions)}: drop low-energy hallucination {start:.3f}-{end:.3f}", flush=True)
                continue
            print(f"track {track_number:02d} region {region_index}/{len(regions)}: {start:.3f}-{end:.3f}", flush=True)
            clip = audio[int(start * sample_rate): int(end * sample_rate)]
            decode_options = {"best_of": 1} if args.beam_size <= 1 else {"beam_size": args.beam_size}
            result = model.transcribe(
                clip, language="en", word_timestamps=True, condition_on_previous_text=False,
                fp16=False, temperature=0, **decode_options,
            )
            replacement = replacement_segments(result, start)
            validation_failures = validate_replacement(replacement, start, end)
            if validation_failures:
                region_records.append({
                    "start": start, "end": end, "action": "retranscribe",
                    "status": "rejected", "replacement_segments": len(replacement),
                    "validation_failures": validation_failures,
                })
                print(
                    f"track {track_number:02d} region {region_index}/{len(regions)}: "
                    f"REJECTED ({len(validation_failures)} structural failures)", flush=True,
                )
                continue
            retained = [segment for segment in transcript["segments"] if float(segment["end"]) <= start or float(segment["start"]) >= end]
            transcript["segments"] = sorted(retained + replacement, key=lambda segment: float(segment["start"]))
            region_records.append({"start": start, "end": end, "action": "retranscribe", "status": "accepted", "replacement_segments": len(replacement)})
        normalize_stats(transcript)
        transcript["whisper"] = {
            **transcript.get("whisper", {}), "backend": "openai", "package_version": importlib.metadata.version("openai-whisper"),
            "model": args.model, "device": "cpu", "beam_size": args.beam_size,
        }
        transcript.setdefault("qa_remediation", []).append({
            "created_utc": datetime.now(UTC).isoformat(), "source_ledger": str(args.qa_ledger),
            "model": args.model, "condition_on_previous_text": False, "regions": region_records,
        })
        destination.write_text(json.dumps(transcript, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        completed.append({"track": track_number, "title": track["title"], "regions": len(regions)})
    summary = {
        "schema": "karaoke-region-remediation/v1", "created_utc": datetime.now(UTC).isoformat(),
        "model": args.model, "source_ledger": str(args.qa_ledger), "skip_tracks": sorted(skip), "completed": completed,
    }
    (args.output_dir / "remediation-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
