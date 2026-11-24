"""Produce a repeatable track-by-track karaoke QA ledger from audio, lyrics, pitch, and timing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from .audit_karaoke_transcripts import audit_transcript
from .paths import safe_component

SCHEMA = "karaoke-album-qa/v1"


def sha256(path: Path) -> str:
    """Hash an evidence file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def db_ratio(numerator: float, denominator: float) -> float:
    """Convert an amplitude ratio to decibels with finite silence handling."""
    return round(20 * math.log10(max(numerator, 1e-12) / max(denominator, 1e-12)), 3)


def sonic_metrics(source: Path, vocal: Path, accompaniment: Path, separation_review: dict | None = None) -> dict:
    """Measure stem integrity and require independent vocal-leakage review."""
    files = [sf.SoundFile(path) for path in (source, vocal, accompaniment)]
    try:
        sample_rates = [handle.samplerate for handle in files]
        channels = [handle.channels for handle in files]
        frames = [len(handle) for handle in files]
        block_size = 65536
        sum_source = sum_residual = sum_vocal = sum_accompaniment = dot = 0.0
        sum_mix = source_mix_dot = 0.0
        peak_accompaniment = 0.0
        clipped = samples = 0
        while True:
            blocks = [handle.read(block_size, dtype="float32", always_2d=True) for handle in files]
            count = min(len(block) for block in blocks)
            if not count:
                break
            source_block, vocal_block, accompaniment_block = [block[:count] for block in blocks]
            common_channels = min(source_block.shape[1], vocal_block.shape[1], accompaniment_block.shape[1])
            source_block = source_block[:, :common_channels]
            vocal_block = vocal_block[:, :common_channels]
            accompaniment_block = accompaniment_block[:, :common_channels]
            residual = source_block - vocal_block - accompaniment_block
            mixture = vocal_block + accompaniment_block
            sum_source += float(np.sum(source_block * source_block))
            sum_residual += float(np.sum(residual * residual))
            sum_vocal += float(np.sum(vocal_block * vocal_block))
            sum_accompaniment += float(np.sum(accompaniment_block * accompaniment_block))
            dot += float(np.sum(vocal_block * accompaniment_block))
            sum_mix += float(np.sum(mixture * mixture))
            source_mix_dot += float(np.sum(source_block * mixture))
            peak_accompaniment = max(peak_accompaniment, float(np.max(np.abs(accompaniment_block))))
            clipped += int(np.sum(np.abs(accompaniment_block) >= 0.999))
            samples += accompaniment_block.size
        source_rms = math.sqrt(sum_source / max(1, samples))
        vocal_rms = math.sqrt(sum_vocal / max(1, samples))
        accompaniment_rms = math.sqrt(sum_accompaniment / max(1, samples))
        optimal_gain = source_mix_dot / max(1e-12, sum_mix)
        normalized_residual_energy = max(0.0, sum_source - 2 * optimal_gain * source_mix_dot + optimal_gain * optimal_gain * sum_mix)
        residual_rms = math.sqrt(normalized_residual_energy / max(1, samples))
        correlation = dot / max(1e-12, math.sqrt(sum_vocal * sum_accompaniment))
        failures = []
        if len(set(sample_rates)) != 1 or len(set(channels)) != 1:
            failures.append("stem_audio_layout_mismatch")
        if max(frames) - min(frames) > max(sample_rates) * 0.1:
            failures.append("stem_duration_mismatch")
        if peak_accompaniment >= 1.0 or clipped / max(1, samples) > 1e-5:
            failures.append("accompaniment_clipping")
        reconstruction_db = db_ratio(residual_rms, source_rms)
        if reconstruction_db > -20:
            failures.append("poor_stem_reconstruction")
        vocal_relative_db = db_ratio(vocal_rms, source_rms)
        if vocal_relative_db < -35 and (separation_review or {}).get("track_type") != "instrumental":
            failures.append("degenerate_or_silent_vocal_stem")
        if not separation_review or separation_review.get("status") != "pass":
            failures.append("vocal_leakage_not_independently_verified")
        expected_hashes = {
            "source_sha256": sha256(source),
            "vocal_sha256": sha256(vocal),
            "accompaniment_sha256": sha256(accompaniment),
        }
        if not separation_review or any(separation_review.get(key) != value for key, value in expected_hashes.items()):
            failures.append("separation_review_hash_mismatch")
        return {
            "sample_rates": sample_rates,
            "channels": channels,
            "frames": frames,
            "duration_seconds": round(min(frames) / sample_rates[0], 3),
            "accompaniment_peak": round(peak_accompaniment, 6),
            "accompaniment_clipped_fraction": round(clipped / max(1, samples), 9),
            "stem_reconstruction_error_db": reconstruction_db,
            "stem_reconstruction_gain": round(optimal_gain, 6),
            "vocal_rms_db_relative_source": vocal_relative_db,
            "accompaniment_rms_db_relative_source": db_ratio(accompaniment_rms, source_rms),
            "vocal_accompaniment_correlation": round(correlation, 5),
            "separation_review": separation_review,
            "reviewed_audio_hashes": expected_hashes,
            "failures": failures,
            "pass": not failures,
        }
    finally:
        for handle in files:
            handle.close()


def vocal_alignment(vocal: Path, segments: list[dict], duration: float, *, allow_empty: bool = False) -> dict:
    """Measure lyric coverage of vocal activity without treating empty drafts as valid."""
    audio, sample_rate = sf.read(vocal, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1)
    frame_seconds = 0.1
    frame_size = max(1, int(sample_rate * frame_seconds))
    usable = len(mono) // frame_size * frame_size
    if not usable:
        return {"pass": False, "failures": ["empty_vocal_stem"]}
    frames = mono[:usable].reshape(-1, frame_size)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    threshold = max(10 ** (-45 / 20), float(np.percentile(rms, 65)) * 0.35)
    active = rms >= threshold
    times = (np.arange(len(rms)) + 0.5) * frame_seconds
    covered = np.zeros(len(rms), dtype=bool)
    for segment in segments:
        for word in segment.get("words", []):
            covered |= (times >= float(word["start"]) - 0.08) & (times <= float(word["end"]) + 0.12)
    active_count = int(np.sum(active))
    active_covered = int(np.sum(active & covered))
    voiced_coverage = active_covered / max(1, active_count)
    lyric_silence = int(np.sum(covered & ~active)) / max(1, int(np.sum(covered)))
    failures = []
    if not segments and not allow_empty:
        failures.append("vocal_track_has_no_lyric_segments")
    if segments and voiced_coverage < 0.45:
        failures.append("vocal_activity_poorly_covered_by_lyrics")
    if segments and lyric_silence > 0.65:
        failures.append("lyrics_frequently_cover_silence")
    return {
        "frame_seconds": frame_seconds,
        "activity_threshold_dbfs": round(20 * math.log10(threshold), 3),
        "active_vocal_coverage": round(voiced_coverage, 4),
        "lyric_silence_fraction": round(lyric_silence, 4),
        "failures": failures,
        "pass": not failures,
    }


def lyric_metrics(transcript: dict, transcript_audit: dict, *, allow_empty: bool = False) -> dict:
    """Summarize lexical/timing evidence and reject unexplained empty drafts."""
    words = [word for segment in transcript.get("segments", []) for word in segment.get("words", [])]
    unexpected = [word for word in words if re.search(r"[^\x00-\x7f]", str(word.get("text", "")))]
    low_confidence = [word for word in words if float(word.get("probability") or 0) < 0.55]
    failures = list(transcript_audit["failures"])
    if unexpected:
        failures.append({"type": "unexpected_script", "count": len(unexpected)})
    if not words and not allow_empty:
        failures.append({"type": "vocal_track_has_no_words"})
    return {
        "word_count": len(words),
        "mean_word_probability": transcript.get("mean_word_probability"),
        "low_confidence_word_count": len(low_confidence),
        "unexpected_script_word_count": len(unexpected),
        "timing_failures": transcript_audit["failures"],
        "warnings": transcript_audit["warnings"],
        "failures": failures,
        "pass": not failures,
    }


def pitch_metrics(karaoke_data: dict | None, *, allow_empty: bool = False) -> dict:
    """Require note coverage for lyric-bearing tracks and explicit empty-track classification."""
    if not karaoke_data:
        return {"pass": False, "failures": ["karaoke_analysis_missing"]}
    words = karaoke_data.get("words", [])
    events = karaoke_data.get("note_events", [])
    pitched_words = {event.get("word_index") for event in events}
    ratio = len(pitched_words) / max(1, len(words))
    failures = []
    if not words and not allow_empty:
        failures.append("vocal_track_has_no_pitch_words")
    if words and ratio < 0.80:
        failures.append("insufficient_word_pitch_coverage")
    if karaoke_data.get("detected_key", {}).get("status") != "verified_override":
        failures.append("key_not_independently_verified")
    if karaoke_data.get("rhythm", {}).get("meter", {}).get("status") != "verified_override":
        failures.append("meter_not_independently_verified")
    return {
        "words": len(words),
        "note_events": len(events),
        "pitched_word_fraction": round(ratio, 4),
        "timing_rejected_segments": karaoke_data.get("timing_rejected_segments", 0),
        "failures": failures,
        "pass": not failures,
    }


def remediation_regions(transcript: dict, transcript_audit: dict, padding: float = 1.5) -> list[dict]:
    """Merge transcript failures into padded regions for targeted reprocessing."""
    bad_segments = sorted({int(item["segment"]) for item in transcript_audit["failures"] if "segment" in item})
    raw = []
    for index in bad_segments:
        segment = transcript["segments"][index]
        raw.append([max(0.0, float(segment["start"]) - padding), float(segment["end"]) + padding])
    merged: list[list[float]] = []
    for start, end in raw:
        if merged and start <= merged[-1][1] + 0.5:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [{"start": round(start, 3), "end": round(end, 3), "reason": "transcript_timing_failure"} for start, end in merged]


def classify_remediation_regions(vocal: Path, regions: list[dict]) -> list[dict]:
    """Classify whether failed regions contain enough vocal energy to retry."""
    if not regions:
        return []
    audio, sample_rate = sf.read(vocal, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1)
    frame_size = max(1, int(sample_rate * 0.1))
    usable = len(mono) // frame_size * frame_size
    frame_rms = np.sqrt(np.mean(mono[:usable].reshape(-1, frame_size) ** 2, axis=1) + 1e-12) if usable else np.asarray([0.0])
    activity_threshold = max(10 ** (-45 / 20), float(np.percentile(frame_rms, 65)) * 0.35)
    classified = []
    for region in regions:
        start_sample = max(0, int(float(region["start"]) * sample_rate))
        end_sample = min(len(mono), int(float(region["end"]) * sample_rate))
        clip = mono[start_sample:end_sample]
        rms = float(np.sqrt(np.mean(clip * clip) + 1e-12)) if clip.size else 0.0
        classified.append({
            **region,
            "vocal_rms_dbfs": round(20 * math.log10(max(rms, 1e-12)), 3),
            "activity_threshold_dbfs": round(20 * math.log10(activity_threshold), 3),
            "action": "retranscribe" if rms >= activity_threshold else "drop_hallucination",
        })
    return classified


def main() -> int:
    """Run album-scale sonic, transcript, pitch, and evidence-binding gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--album-manifest", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--stem-root", required=True, type=Path)
    parser.add_argument("--transcripts", required=True, type=Path)
    parser.add_argument("--karaoke-data", type=Path)
    parser.add_argument(
        "--separation-review", type=Path,
        help="JSON evidence with tracks.<number>.status=pass from independent listening or leakage analysis",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    album = json.loads(args.album_manifest.read_text(encoding="utf-8"))
    source_root = args.source_root.resolve(strict=True)
    stem_root = args.stem_root.resolve(strict=True)
    review_payload = json.loads(args.separation_review.read_text(encoding="utf-8")) if args.separation_review else {}
    separation_reviews = review_payload.get("tracks", {})
    rows = []
    for track in album["tracks"]:
        safe_title = safe_component(track["title"], field=f"track {track['number']} title")
        transcript_path = args.transcripts / f"{track['number']:02d} - {safe_title}.whisper.json"
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        classification = (transcript.get("lyric_classification") or {}).get("status")
        explicitly_instrumental = classification == "instrumental_or_no_reliable_vocal"
        transcript_audit = audit_transcript(transcript_path)
        karaoke_path = args.karaoke_data / f"{track['number']:02d}.karaoke.json" if args.karaoke_data else None
        karaoke_data = json.loads(karaoke_path.read_text(encoding="utf-8")) if karaoke_path and karaoke_path.exists() else None
        source_path = (source_root / track["source_file"]).resolve(strict=True)
        vocal_path = (stem_root / track["vocal_stem"]).resolve(strict=True)
        accompaniment_path = (stem_root / track["accompaniment_stem"]).resolve(strict=True)
        if not source_path.is_relative_to(source_root) or not vocal_path.is_relative_to(stem_root) or not accompaniment_path.is_relative_to(stem_root):
            raise ValueError(f"track path escapes a trusted root: {track['title']}")
        if (
            transcript.get("vocal_sha256") != track["vocal_sha256"]
            or int(transcript.get("track", -1)) != int(track["number"])
        ):
            raise ValueError(f"transcript provenance mismatch for track {track['number']}")
        review = separation_reviews.get(str(track["number"]))
        if review and (
            review.get("track_type") not in {"vocal", "instrumental"}
            or not review.get("notes")
            or not review_payload.get("reviewer")
            or not review_payload.get("method")
        ):
            review = {**review, "status": "invalid", "validation_error": "incomplete separation review evidence"}
        sonic = sonic_metrics(source_path, vocal_path, accompaniment_path, review)
        lyrics = lyric_metrics(transcript, transcript_audit, allow_empty=explicitly_instrumental)
        alignment = vocal_alignment(
            vocal_path, transcript.get("segments", []), sonic["duration_seconds"],
            allow_empty=explicitly_instrumental,
        )
        pitch = pitch_metrics(karaoke_data, allow_empty=explicitly_instrumental)
        gates = {"sonic": sonic["pass"], "lyrics": lyrics["pass"], "alignment": alignment["pass"], "pitch": pitch["pass"]}
        regions = classify_remediation_regions(vocal_path, remediation_regions(transcript, transcript_audit))
        rows.append({
            "track": track["number"], "title": track["title"], "gates": gates, "pass": all(gates.values()),
            "sonic": sonic, "lyrics": lyrics, "alignment": alignment, "pitch": pitch,
            "evidence": {
                "source_sha256": sha256(source_path),
                "vocal_sha256": sha256(vocal_path),
                "accompaniment_sha256": sha256(accompaniment_path),
                "transcript_sha256": sha256(transcript_path),
                "karaoke_data_sha256": sha256(karaoke_path) if karaoke_path and karaoke_path.exists() else None,
            },
            "remediation_regions": regions,
        })
    report = {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "album_manifest": str(args.album_manifest),
        "album_manifest_sha256": sha256(args.album_manifest),
        "separation_review_sha256": sha256(args.separation_review) if args.separation_review else None,
        "thresholds": {
            "stem_reconstruction_error_db_max": -20,
            "minimum_word_pitch_coverage": 0.80,
            "minimum_active_vocal_coverage": 0.45,
            "maximum_lyric_silence_fraction": 0.65,
        },
        "tracks": rows,
        "summary": {gate: sum(row["gates"][gate] for row in rows) for gate in ("sonic", "lyrics", "alignment", "pitch")},
        "passing_tracks": sum(row["pass"] for row in rows),
        "track_count": len(rows),
        "pass": all(row["pass"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "track_count": report["track_count"], "passing_tracks": report["passing_tracks"], "summary": report["summary"], "pass": report["pass"]}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
