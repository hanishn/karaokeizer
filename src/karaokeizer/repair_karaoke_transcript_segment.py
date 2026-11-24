"""Re-transcribe one vocal-stem interval and deterministically merge it into a Whisper transcript."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--vocal", required=True, type=Path)
    parser.add_argument("--start", required=True, type=float)
    parser.add_argument("--end", required=True, type=float)
    parser.add_argument("--model", default="turbo")
    parser.add_argument("--beam-size", type=int, default=8)
    args = parser.parse_args()
    import whisper
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
    replacement = []
    for segment in result["segments"]:
        words = [
            {
                "text": word["word"].strip(),
                "start": round(float(word["start"]) + args.start, 3),
                "end": round(float(word["end"]) + args.start, 3),
                "probability": round(float(word.get("probability", 0)), 6),
            }
            for word in segment.get("words", [])
        ]
        if words:
            replacement.append(
                {
                    "start": round(float(segment["start"]) + args.start, 3),
                    "end": round(float(segment["end"]) + args.start, 3),
                    "text": segment["text"].strip(),
                    "avg_logprob": round(float(segment.get("avg_logprob", 0)), 6),
                    "no_speech_probability": round(float(segment.get("no_speech_prob", 0)), 6),
                    "words": words,
                }
            )

    retained = [s for s in transcript["segments"] if float(s["end"]) <= args.start or float(s["start"]) >= args.end]
    transcript["segments"] = sorted(retained + replacement, key=lambda segment: float(segment["start"]))
    words = [word for segment in transcript["segments"] for word in segment["words"]]
    transcript["text"] = " ".join(segment["text"] for segment in transcript["segments"])
    transcript["word_count"] = len(words)
    transcript["mean_word_probability"] = round(sum(float(w["probability"]) for w in words) / len(words), 6)
    transcript["low_confidence_words"] = [w for w in words if float(w["probability"]) < 0.55]
    normalized = [re.sub(r"[^a-z0-9']", "", w["text"].lower()) for w in words]
    normalized = [word for word in normalized if word]
    transcript["repetition_ratio"] = round(max(normalized.count(w) for w in set(normalized)) / len(normalized), 6)
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
    backup = args.transcript.with_suffix(".pre-segment-repair.json")
    if not backup.exists():
        shutil.copy2(args.transcript, backup)
    args.transcript.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"transcript": str(args.transcript), "backup": str(backup), "replacement_segments": len(replacement), "words": len(words)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
