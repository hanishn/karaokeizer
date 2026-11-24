"""Create resumable, hash-verified Whisper lyric drafts from vocal stems."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from .paths import safe_component

SCHEMA = "karaoke-whisper-transcript/v1"


def sha256(path: Path) -> str:
    """Hash vocal stems for candidate identity and safe checkpoint reuse."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    """Transcribe manifest-bound stems into an immutable candidate directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--album-manifest", required=True, type=Path)
    parser.add_argument("--stem-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--model-revision", help="Required immutable Hugging Face revision for faster-whisper")
    parser.add_argument("--backend", choices=["openai", "faster-whisper"], default="openai")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--initial-prompt", default="", help="Optional literal lyric context; never use instructions")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N tracks")
    parser.add_argument("--start-track", type=int, default=1)
    parser.add_argument("--tracks", default="", help="Optional comma-separated track numbers")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    album = json.loads(args.album_manifest.read_text(encoding="utf-8"))
    config = album["config"]
    stem_root = args.stem_root.resolve(strict=True)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    tracks = [track for track in album["tracks"] if track["number"] >= args.start_track]
    selected = {int(value) for value in args.tracks.split(",") if value.strip()}
    if selected:
        tracks = [track for track in tracks if track["number"] in selected]
    tracks = tracks[: args.limit or None]
    package_name = "faster-whisper" if args.backend == "faster-whisper" else "openai-whisper"
    package_version = importlib.metadata.version(package_name)
    if args.backend == "faster-whisper":
        if not args.model_revision:
            parser.error("--model-revision is required with faster-whisper")
        model_revision = args.model_revision
        model_source = args.model
    else:
        import whisper

        model_source = whisper._MODELS.get(args.model)  # noqa: SLF001 - upstream checksum-bearing registry
        if not model_source:
            parser.error("OpenAI Whisper model must be a checksum-pinned built-in model name")
        model_revision = model_source.rstrip("/").split("/")[-2]
    decode_identity = {
        "backend": args.backend,
        "package_version": package_version,
        "model": args.model,
        "model_revision": model_revision,
        "model_source": model_source,
        "device": args.device,
        "compute_type": args.compute_type,
        "beam_size": args.beam_size,
        "language": "en",
        "word_timestamps": True,
        "condition_on_previous_text": True,
        "initial_prompt_sha256": hashlib.sha256(args.initial_prompt.encode("utf-8")).hexdigest(),
    }
    if args.backend == "faster-whisper":
        from faster_whisper import WhisperModel

        model = WhisperModel(
            args.model, device=args.device, compute_type=args.compute_type, revision=args.model_revision,
        )
    else:
        model = whisper.load_model(args.model, device=args.device)
    index = []

    for position, track in enumerate(tracks, 1):
        vocal = (stem_root / track["vocal_stem"]).resolve(strict=True)
        if not vocal.is_relative_to(stem_root):
            raise ValueError(f"vocal stem escapes --stem-root: {track['vocal_stem']}")
        vocal_hash = sha256(vocal)
        safe_title = safe_component(track["title"], field=f"track {track['number']} title")
        destination = output / f"{track['number']:02d} - {safe_title}.whisper.json"
        if destination.exists() and not args.force:
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing.get("vocal_sha256") == vocal_hash and existing.get("decode_identity") == decode_identity:
                print(f"[{position}/{len(tracks)}] checkpoint exists: {track['title']}")
                index.append(existing)
                continue
            raise RuntimeError(
                f"candidate identity mismatch for {track['title']!r}; "
                "write to a fresh output directory or use --force explicitly"
            )
        print(f"[{position}/{len(tracks)}] transcribing: {track['title']}", flush=True)
        prompt = args.initial_prompt or None
        if args.backend == "faster-whisper":
            segments_iter, info = model.transcribe(
                str(vocal), language="en", beam_size=args.beam_size, word_timestamps=True,
                vad_filter=False, condition_on_previous_text=True, initial_prompt=prompt,
            )
            raw_segments = list(segments_iter)
            detected_language = info.language
            language_probability = round(info.language_probability, 6)
            audio_duration = round(info.duration, 3)
        else:
            result = model.transcribe(
                str(vocal), language="en", beam_size=args.beam_size, word_timestamps=True,
                condition_on_previous_text=True, initial_prompt=prompt, fp16=False,
            )
            raw_segments = result["segments"]
            detected_language = result.get("language", "en")
            language_probability = None
            audio_duration = round(max((segment["end"] for segment in raw_segments), default=0), 3)
        segments = []
        words = []
        for segment in raw_segments:
            segment_words_raw = segment.words or [] if args.backend == "faster-whisper" else segment.get("words", [])
            segment_words = []
            for word in segment_words_raw:
                if args.backend == "faster-whisper":
                    word_text, word_start, word_end, word_probability = word.word, word.start, word.end, word.probability
                else:
                    word_text, word_start, word_end = word["word"], word["start"], word["end"]
                    word_probability = word.get("probability", 0.0)
                item = {
                    "text": word_text.strip(),
                    "start": round(word_start, 3),
                    "end": round(word_end, 3),
                    "probability": round(word_probability, 6),
                }
                segment_words.append(item)
                words.append(item)
            segments.append(
                {
                    "start": round(segment.start if args.backend == "faster-whisper" else segment["start"], 3),
                    "end": round(segment.end if args.backend == "faster-whisper" else segment["end"], 3),
                    "text": (segment.text if args.backend == "faster-whisper" else segment["text"]).strip(),
                    "avg_logprob": round(segment.avg_logprob if args.backend == "faster-whisper" else segment.get("avg_logprob", 0.0), 6),
                    "no_speech_probability": round(segment.no_speech_prob if args.backend == "faster-whisper" else segment.get("no_speech_prob", 0.0), 6),
                    "words": segment_words,
                }
            )
        transcript = {
            "schema": SCHEMA,
            "created_utc": datetime.now(UTC).isoformat(),
            "track": track["number"],
            "title": track["title"],
            "artist": config["artist"],
            "album": config["album"],
            "vocal_stem": track["vocal_stem"],
            "vocal_sha256": vocal_hash,
            "decode_identity": decode_identity,
            "whisper": {
                "backend": args.backend,
                "package_version": package_version,
                "model": args.model,
                "device": args.device,
                "compute_type": args.compute_type,
                "beam_size": args.beam_size,
                "language": detected_language,
                "language_probability": language_probability,
                "duration": audio_duration,
            },
            "text": " ".join(segment["text"] for segment in segments),
            "segments": segments,
            "word_count": len(words),
            "mean_word_probability": (
                round(sum(word["probability"] for word in words) / len(words), 6) if words else 0
            ),
            "low_confidence_words": [word for word in words if word["probability"] < 0.55],
        }
        normalized_words = [re.sub(r"[^a-z0-9']", "", word["text"].lower()) for word in words]
        normalized_words = [word for word in normalized_words if word]
        most_common_ratio = (
            max(normalized_words.count(word) for word in set(normalized_words)) / len(normalized_words)
            if normalized_words else 0
        )
        transcript["repetition_ratio"] = round(most_common_ratio, 6)
        if not words:
            transcript["quality_status"] = "no_reliable_lyrics"
        elif most_common_ratio >= 0.4 and transcript["mean_word_probability"] < 0.65:
            transcript["quality_status"] = "rejected_probable_hallucination"
        elif transcript["mean_word_probability"] < 0.55:
            transcript["quality_status"] = "needs_manual_review"
        else:
            transcript["quality_status"] = "draft_ready_for_review"
        destination.write_text(json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8")
        index.append(transcript)

    summary = {
        "schema": SCHEMA,
        "album_manifest": str(args.album_manifest),
        "model": args.model,
        "backend": args.backend,
        "track_count": len(index),
        "tracks": [
            {
                "track": item["track"],
                "title": item["title"],
                "word_count": item["word_count"],
                "mean_word_probability": item["mean_word_probability"],
                "low_confidence_word_count": len(item["low_confidence_words"]),
                "repetition_ratio": item["repetition_ratio"],
                "quality_status": item["quality_status"],
            }
            for item in index
        ],
    }
    (output / "transcription-index.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
