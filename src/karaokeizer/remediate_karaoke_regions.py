"""Batch re-transcribe QA-failed album regions with one model load and quarantined output."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path


def sha256(path: Path) -> str:
    """Hash a remediation input so stale ledgers cannot authorize changes."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_stats(transcript: dict) -> None:
    """Recompute transcript summaries after an accepted regional merge."""
    words = [word for segment in transcript["segments"] for word in segment.get("words", [])]
    transcript["text"] = " ".join(segment["text"] for segment in transcript["segments"])
    transcript["word_count"] = len(words)
    transcript["mean_word_probability"] = round(sum(float(word.get("probability") or 0) for word in words) / max(1, len(words)), 6)
    transcript["low_confidence_words"] = [word for word in words if float(word.get("probability") or 0) < 0.55]
    normalized = [re.sub(r"[^a-z0-9']", "", str(word.get("text", "")).lower()) for word in words]
    normalized = [word for word in normalized if word]
    transcript["repetition_ratio"] = round(max((normalized.count(word) for word in set(normalized)), default=0) / max(1, len(normalized)), 6)


def replacement_segments(result: dict, offset: float) -> list[dict]:
    """Normalize and shift region-local Whisper output to album time."""
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


def retained_segments_outside_region(segments: list[dict], start: float, end: float) -> list[dict]:
    """Preserve words outside a padded repair interval, even in intersecting segments."""

    retained: list[dict] = []
    for segment in segments:
        if float(segment["end"]) <= start or float(segment["start"]) >= end:
            retained.append(segment)
            continue
        words = [
            dict(word) for word in segment.get("words", [])
            if float(word["end"]) <= start or float(word["start"]) >= end
        ]
        if words:
            retained.append({
                **segment,
                "start": min(float(word["start"]) for word in words),
                "end": max(float(word["end"]) for word in words),
                "text": " ".join(str(word.get("text", "")) for word in words).strip(),
                "words": words,
            })
    return retained


def validate_merged_timeline(segments: list[dict]) -> list[str]:
    """Validate chronological word timing across replacement boundaries."""

    failures: list[str] = []
    previous_start = -1.0
    previous_end = -1.0
    for segment in sorted(segments, key=lambda item: float(item["start"])):
        for word in segment.get("words", []):
            start, end = float(word["start"]), float(word["end"])
            if start <= previous_start:
                failures.append("non_monotonic_or_duplicate_word_start")
            if previous_end - start > 0.12:
                failures.append("cross_boundary_word_overlap")
            previous_start, previous_end = start, end
    return sorted(set(failures))


def main() -> int:
    """Repair only hash-authorized regions into a quarantined candidate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--album-manifest", required=True, type=Path)
    parser.add_argument("--stem-root", required=True, type=Path)
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
    if ledger.get("schema") != "karaoke-album-qa/v1":
        parser.error("QA ledger schema is not karaoke-album-qa/v1")
    if ledger.get("album_manifest_sha256") != sha256(args.album_manifest):
        parser.error("QA ledger is stale for the supplied album manifest")
    ledger_hash = sha256(args.qa_ledger)
    tracks = {int(track["number"]): track for track in album["tracks"]}
    stem_root = args.stem_root.resolve(strict=True)
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
        source_transcript = args.transcripts / f"{track_number:02d} - {track['title']}.whisper.json"
        destination = args.output_dir / f"{track_number:02d} - {track['title']}.whisper.json"
        transcript = json.loads(destination.read_text(encoding="utf-8"))
        vocal_path = (stem_root / track["vocal_stem"]).resolve(strict=True)
        if not vocal_path.is_relative_to(stem_root):
            raise ValueError(f"vocal stem escapes --stem-root: {track['vocal_stem']}")
        evidence = row.get("evidence", {})
        if (
            not source_transcript.exists()
            or sha256(source_transcript) != evidence.get("transcript_sha256")
            or sha256(vocal_path) != evidence.get("vocal_sha256")
            or evidence.get("vocal_sha256") != track.get("vocal_sha256")
        ):
            raise ValueError(f"stale QA remediation evidence for track {track_number}")
        audio = whisper.load_audio(str(vocal_path))
        sample_rate = whisper.audio.SAMPLE_RATE
        region_records = []
        for region_index, region in enumerate(regions, 1):
            start, end = float(region["start"]), float(region["end"])
            if region.get("action") == "drop_hallucination":
                transcript["segments"] = retained_segments_outside_region(transcript["segments"], start, end)
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
            retained = retained_segments_outside_region(transcript["segments"], start, end)
            merged = sorted(retained + replacement, key=lambda segment: float(segment["start"]))
            merge_failures = validate_merged_timeline(merged)
            if merge_failures:
                region_records.append({
                    "start": start, "end": end, "action": "retranscribe", "status": "rejected",
                    "replacement_segments": len(replacement), "validation_failures": merge_failures,
                })
                continue
            transcript["segments"] = merged
            region_records.append({"start": start, "end": end, "action": "retranscribe", "status": "accepted", "replacement_segments": len(replacement)})
        normalize_stats(transcript)
        transcript["whisper"] = {
            **transcript.get("whisper", {}), "backend": "openai", "package_version": importlib.metadata.version("openai-whisper"),
            "model": args.model, "device": "cpu", "beam_size": args.beam_size,
        }
        transcript.setdefault("qa_remediation", []).append({
            "created_utc": datetime.now(UTC).isoformat(), "source_ledger": str(args.qa_ledger),
            "source_ledger_sha256": ledger_hash,
            "source_transcript_sha256": evidence["transcript_sha256"],
            "vocal_sha256": evidence["vocal_sha256"],
            "model": args.model, "condition_on_previous_text": False, "regions": region_records,
        })
        destination.write_text(json.dumps(transcript, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        completed.append({"track": track_number, "title": track["title"], "regions": len(regions)})
    summary = {
        "schema": "karaoke-region-remediation/v1", "created_utc": datetime.now(UTC).isoformat(),
        "model": args.model, "source_ledger": str(args.qa_ledger), "source_ledger_sha256": ledger_hash,
        "album_manifest_sha256": sha256(args.album_manifest), "skip_tracks": sorted(skip), "completed": completed,
    }
    (args.output_dir / "remediation-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
