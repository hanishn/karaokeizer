"""Build and verify a lyric/note-overlay player from Whisper transcripts and vocal stems."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import shutil
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import librosa
import numpy as np
import pyphen

from .html_safety import json_for_script
from .paths import safe_component
from .ssli_manifest import MANIFEST_NAME, validate_manifest

SCHEMA = "karaoke-overlay/v1"
KEY_METHOD = "cqt_chroma_ks-v1"
RHYTHM_METHOD = "librosa-beat-accent-phase-v1"
PITCH_METHOD = "pyin-contour-46ms-v2"
NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
ENGLISH_HYPHENATOR = pyphen.Pyphen(lang="en_US")


def lyric_syllables(text: str) -> list[str]:
    """Dictionary-backed display syllables while preserving punctuation."""
    match = re.search(r"[A-Za-z]+", text)
    if not match:
        return [text] if text else []
    pieces = ENGLISH_HYPHENATOR.inserted(match.group(0)).split("-")
    if not pieces:
        return [text]
    pieces[0] = text[: match.start()] + pieces[0]
    pieces[-1] += text[match.end() :]
    return pieces


def has_unexpected_script(text: str) -> bool:
    """Flag scripts outside the configured lyric alphabet for human review."""
    blocked = ("CYRILLIC", "HANGUL", "CJK", "HIRAGANA", "KATAKANA", "ARABIC", "HEBREW", "DEVANAGARI")
    return any(any(script in unicodedata.name(char, "") for script in blocked) for char in text)


def sha256(path: Path) -> str:
    """Hash one input or rendered artifact for downstream evidence binding."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def note_for_midi(midi: float | None) -> str | None:
    """Format a MIDI pitch as a note name with octave, or return ``None``."""
    return librosa.midi_to_note(midi, octave=True, cents=False) if midi is not None else None


def classify_chroma(chroma: np.ndarray) -> list[dict]:
    """Rank all major and minor keys against an averaged chroma vector."""
    vector = np.mean(chroma, axis=1)
    scores = []
    for tonic in range(12):
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            score = float(np.corrcoef(vector, np.roll(profile, tonic))[0, 1])
            scores.append({"tonic": NOTE_NAMES[tonic], "mode": mode, "label": f"{NOTE_NAMES[tonic]} {mode}", "score": score})
    return sorted(scores, key=lambda item: item["score"], reverse=True)


def detect_key(audio: Path) -> dict:
    """Estimate key from audio and expose confidence and runner-up data."""
    y, sr = librosa.load(audio, sr=22050, mono=True)
    harmonic = librosa.effects.harmonic(y)
    ranked = classify_chroma(librosa.feature.chroma_cqt(y=harmonic, sr=sr))
    best, second = ranked[0], ranked[1]
    margin = float(np.clip((best["score"] - second["score"]) / max(abs(best["score"]), 1e-9), 0, 1))
    sections = []
    for name, start, end in (("intro", 0.0, 0.2), ("middle", 0.4, 0.6), ("outro", 0.8, 1.0)):
        clip = harmonic[int(len(harmonic) * start) : int(len(harmonic) * end)]
        winner = classify_chroma(librosa.feature.chroma_cqt(y=clip, sr=sr))[0]
        sections.append({"section": name, "label": winner["label"], "score": round(winner["score"], 6)})
    agreement = sum(section["label"] == best["label"] for section in sections)
    if margin < 0.12:
        display, status = "Key ambiguous", "ambiguous"
    elif margin < 0.20 or agreement < 2:
        display, status = f"{best['label']} (estimated)", "estimated"
    else:
        display, status = best["label"], "confident"
    return {
        "tonic": best["tonic"], "mode": best["mode"], "label": best["label"],
        "display": display, "status": status, "confidence": round(margin, 6),
        "score": round(best["score"], 6),
        "runner_up": {"label": second["label"], "score": round(second["score"], 6)},
        "section_checks": sections, "section_agreement": agreement,
        "method": KEY_METHOD,
    }


def analyze_rhythm(audio: Path, meter_override: dict | None = None) -> dict:
    """Estimate beats and apply an explicit, reviewed meter/downbeat override."""
    meter_override = meter_override or {}
    numerator = int(meter_override.get("numerator", 4))
    denominator = int(meter_override.get("denominator", 4))
    if numerator < 2 or numerator > 12 or denominator not in {2, 4, 8, 16}:
        raise ValueError("meter override must use numerator 2..12 and denominator 2, 4, 8, or 16")
    y, sr = librosa.load(audio, sr=22050, mono=True)
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset, sr=sr, units="frames")
    beat_frames = np.asarray(beat_frames, dtype=int)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    bpm = float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else 0.0
    if not len(beat_times):
        return {"tempo_bpm": 0.0, "meter": {"numerator": numerator, "denominator": denominator, "status": "no_beat_grid"},
                "beats": [], "measures": [], "method": RHYTHM_METHOD}
    strengths = onset[np.clip(beat_frames, 0, len(onset) - 1)]
    phase_scores = [float(np.sum(strengths[p::numerator])) for p in range(numerator)]
    downbeat_phase = int(meter_override.get("downbeat_phase", np.argmax(phase_scores)))
    if not 0 <= downbeat_phase < numerator:
        raise ValueError("downbeat_phase must be within the selected meter")
    beats = [{"time": round(float(t), 3), "beat": ((i - downbeat_phase) % numerator) + 1}
             for i, t in enumerate(beat_times)]
    measures = []
    number = 1
    downbeat_indices = [i for i in range(downbeat_phase, len(beats), numerator)]
    for pos, index in enumerate(downbeat_indices):
        end_index = downbeat_indices[pos + 1] if pos + 1 < len(downbeat_indices) else min(index + numerator, len(beats))
        end_time = beats[end_index]["time"] if end_index < len(beats) else round(float(librosa.get_duration(y=y, sr=sr)), 3)
        measures.append({"number": number, "start": beats[index]["time"], "end": end_time})
        number += 1
    return {"tempo_bpm": round(bpm, 2),
            "meter": {
                "numerator": numerator, "denominator": denominator,
                "status": "verified_override" if meter_override else "estimated_unverified",
            },
            "beats": beats, "measures": measures, "downbeat_phase": downbeat_phase,
            "method": RHYTHM_METHOD}


def analyze_pitch(vocal: Path, words: list[dict]) -> tuple[list[dict], list[dict]]:
    """Extract a vocal pitch contour and bind note events to timed words."""
    y, sr = librosa.load(vocal, sr=22050, mono=True)
    hop = 512
    f0, voiced, probability = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
        hop_length=hop,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    midi = librosa.hz_to_midi(f0)
    contour = []
    stride = 2
    for index in range(0, len(times), stride):
        if not voiced[index] or not np.isfinite(midi[index]):
            continue
        contour.append(
            {
                "time": round(float(times[index]), 3),
                "midi": round(float(midi[index]), 3),
                "probability": round(float(probability[index]), 4),
            }
        )
    annotated = []
    for word_index, word in enumerate(words):
        mask = (times >= word["start"]) & (times <= word["end"]) & voiced & np.isfinite(midi)
        confident = mask & (probability >= 0.5)
        values = midi[confident]
        median = round(float(np.median(values)), 3) if values.size else None
        item = dict(word)
        item.update(
            {
                "word_index": word_index,
                "midi": median,
                "note": note_for_midi(median),
                "pitch_confidence": round(float(np.mean(probability[confident])), 4)
                if values.size
                else 0,
            }
        )
        annotated.append(item)
    return annotated, contour


def annotate_words_from_contour(contour: list[dict], words: list[dict]) -> list[dict]:
    """Re-bind cached pitch frames to revised word timings without rerunning pYIN."""
    annotated = []
    for word_index, word in enumerate(words):
        points = [
            point for point in contour
            if float(word["start"]) <= float(point["time"]) <= float(word["end"])
            and float(point.get("probability") or 0) >= 0.5
        ]
        values = np.asarray([float(point["midi"]) for point in points], dtype=float)
        median = round(float(np.median(values)), 3) if values.size else None
        item = dict(word)
        item.update({
            "word_index": word_index,
            "midi": median,
            "note": note_for_midi(median),
            "pitch_confidence": round(float(np.mean([float(point["probability"]) for point in points])), 4) if points else 0,
        })
        annotated.append(item)
    return annotated


def contour_to_note_events(contour: list[dict], words: list[dict]) -> list[dict]:
    """Turn frame-level f0 into note events split at every lyric-word boundary."""
    if not contour:
        return []
    events: list[dict] = []
    for lyric_word in words:
        word_start, word_end = float(lyric_word["start"]), float(lyric_word["end"])
        points = [point for point in contour if word_start - 0.04 <= float(point["time"]) <= word_end + 0.06]
        current: list[dict] = []
        pending_change: list[dict] = []

        def flush_word_event(
            word_start: float = word_start,
            word_end: float = word_end,
            lyric_word: dict = lyric_word,
        ) -> None:
            nonlocal current
            if not current:
                return
            start = max(word_start, float(current[0]["time"]))
            step = float(np.median(np.diff([p["time"] for p in current]))) if len(current) > 1 else 0.093
            end = min(word_end, float(current[-1]["time"]) + max(0.06, step))
            if end - start >= 0.06:
                midi = int(round(float(np.median([p["midi"] for p in current]))))
                events.append({
                    "start": round(start, 3), "end": round(end, 3), "midi": midi,
                    "note": note_for_midi(midi),
                    "pitch_confidence": round(float(np.mean([p["probability"] for p in current])), 4),
                    "lyric": lyric_word.get("text", ""), "word_index": lyric_word.get("word_index"),
                })
            current = []

        for point in points:
            if current:
                previous_time = float(current[-1]["time"])
                if float(point["time"]) - previous_time > 0.18:
                    current.extend(pending_change)
                    pending_change = []
                    flush_word_event()
                else:
                    center = float(np.median([p["midi"] for p in current[-7:]]))
                    if abs(float(point["midi"]) - center) >= 0.75:
                        pending_change.append(point)
                        stable_for = float(pending_change[-1]["time"]) - float(pending_change[0]["time"])
                        if len(pending_change) >= 3 and stable_for >= 0.12:
                            flush_word_event()
                            current = pending_change
                            pending_change = []
                        continue
                    current.extend(pending_change)
                    pending_change = []
            current.append(point)
        current.extend(pending_change)
        flush_word_event()
    return events


def analysis_cache_key(track: dict, transcript_hash: str, analysis_review_hash: str | None = None) -> dict:
    """Describe every input and algorithm version that can affect analysis."""

    return {
        "vocal_sha256": track["vocal_sha256"],
        "accompaniment_sha256": track["accompaniment_sha256"],
        "transcript_sha256": transcript_hash,
        "pitch_method": PITCH_METHOD,
        "key_method": KEY_METHOD,
        "rhythm_method": RHYTHM_METHOD,
        "analysis_review_sha256": analysis_review_hash,
    }


def apply_key_override(detected: dict, override: dict | None) -> dict:
    """Replace an estimated key only with explicit reviewed key metadata."""

    if not override:
        return detected
    tonic, mode = str(override["tonic"]), str(override["mode"])
    if tonic not in NOTE_NAMES or mode not in {"major", "minor"}:
        raise ValueError("key override requires a supported tonic and major/minor mode")
    label = f"{tonic} {mode}"
    return {
        **detected, "tonic": tonic, "mode": mode, "label": label, "display": label,
        "status": "verified_override", "review": override,
    }


def validate_analysis_overrides(payload: dict, album_tracks: list[dict]) -> dict:
    """Require a named, complete key/meter review for every album track."""

    if (
        payload.get("schema") != "karaokeizer-analysis-overrides/v1"
        or payload.get("pass") is not True
        or not payload.get("reviewer")
        or not payload.get("method")
        or not payload.get("reviewed_utc")
    ):
        raise ValueError("analysis review requires schema, pass=true, reviewer, method, and reviewed_utc")
    rows = payload.get("tracks", {})
    expected = {str(int(track["number"])) for track in album_tracks}
    if set(rows) != expected:
        raise ValueError("analysis review track inventory must exactly match the album")
    for number, row in rows.items():
        key, meter = row.get("key", {}), row.get("meter", {})
        if (
            row.get("status") != "pass"
            or not key.get("evidence")
            or key.get("tonic") not in NOTE_NAMES
            or key.get("mode") not in {"major", "minor"}
            or not meter.get("evidence")
            or not isinstance(meter.get("numerator"), int)
            or meter.get("denominator") not in {2, 4, 8, 16}
            or not isinstance(meter.get("downbeat_phase"), int)
        ):
            raise ValueError(f"analysis review is incomplete for track {number}")
    return rows


def segment_timing_errors(segment: dict) -> list[str]:
    """Return singer-visible timing defects; defective segments must never reach the player."""
    errors: list[str] = []
    words = segment.get("words", [])
    starts: set[float] = set()
    previous_end: float | None = None
    segment_start = float(segment["start"])
    segment_end = float(segment["end"])
    for word in words:
        start = float(word["start"])
        end = float(word["end"])
        rounded_start = round(start, 3)
        if end - start <= 0.02:
            errors.append("collapsed_word")
        if start < segment_start - 0.12 or end > segment_end + 0.12:
            errors.append("word_outside_segment")
        if rounded_start in starts:
            errors.append("duplicate_word_start")
        starts.add(rounded_start)
        if previous_end is not None and previous_end - start > 0.12:
            errors.append("word_overlap")
        previous_end = end
    return sorted(set(errors))


def filter_lyric_segments(segments: list[dict], transcript: dict) -> tuple[list[dict], str]:
    """Reject Whisper tail hallucinations and tracks without credible vocal content."""
    classification = transcript.get("lyric_classification") or {}
    if classification.get("status") == "instrumental_or_no_reliable_vocal":
        return [], "instrumental_or_no_reliable_vocal"
    repetition = float(transcript.get("repetition_ratio") or 0)
    word_count = int(transcript.get("word_count") or sum(len(s["words"]) for s in segments))
    status = transcript.get("quality_status") or "needs_manual_review"
    if status == "rejected_probable_hallucination" or word_count == 0 or repetition >= 0.30:
        return [], "rejected_probable_hallucination"
    boilerplate = {"thank you", "thanks", "thank you very much"}
    kept = []
    for segment in segments:
        if segment_timing_errors(segment):
            continue
        normalized = re.sub(r"[^a-z0-9' ]+", "", segment["text"].lower()).strip()
        probabilities = [float(w.get("probability") or 0) for w in segment["words"]]
        mean_probability = sum(probabilities) / max(1, len(probabilities))
        if not normalized or normalized in boilerplate or mean_probability < 0.25:
            continue
        kept.append(segment)
    pitched = [s for s in kept if any(float(w.get("pitch_confidence") or 0) >= 0.5 for w in s["words"])]
    if pitched:
        last_credible_end = max(s["end"] for s in pitched)
        kept = [s for s in kept if s["start"] <= last_credible_end + 4.0]
    return kept, status


def build_html(output: Path, album: dict, tracks: list[dict]) -> Path:
    """Write the generated player while keeping metadata out of HTML context."""
    payload = json_for_script(tracks)
    raw_album_name = safe_component(album["config"]["album"], field="album")
    album_name = html.escape(raw_album_name)
    artist = html.escape(str(album["config"]["artist"]))
    html_document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{artist} — {album_name} Karaoke</title>
<style>:root{{color-scheme:dark;font-family:system-ui,sans-serif}}body{{margin:0;background:#070916;color:#f5f2ff}}
main{{width:min(1100px,94vw);margin:1.5rem auto}}header{{display:flex;gap:1rem;align-items:center}}header img{{width:120px;border-radius:10px}}
audio{{width:100%;margin:1rem 0}}#now-playing{{display:block;margin-top:.35rem;color:#c4b5fd;font-size:1.05rem}}.performance-status{{display:grid;grid-template-columns:minmax(180px,.7fr) 1.3fr;gap:.75rem;margin:.5rem 0 0}}.status-card{{background:#111a31;border:1px solid #475569;border-radius:12px;padding:.55rem .8rem;text-align:center;box-shadow:0 8px 22px #0007}}.status-card small{{display:block;color:#94a3b8;font-size:.72rem;font-weight:800;letter-spacing:.12em}}#key{{display:block;font-size:clamp(1.1rem,2.2vw,1.65rem);color:#f8fafc}}#note-line{{min-height:2.4rem;text-align:center;color:#fbbf24;font-size:clamp(1.3rem,2.5vw,2rem);font-weight:800;padding:.1rem}}
#lyrics{{position:absolute!important;width:1px!important;height:1px!important;padding:0!important;margin:-1px!important;overflow:hidden!important;clip:rect(0,0,0,0)!important;white-space:nowrap!important;border:0!important}}#analysis-state{{margin:.45rem 0 .55rem;padding:.42rem .7rem;border-left:4px solid #22c55e;background:#0f172a;color:#dbeafe;font-size:.88rem;font-weight:700}}#analysis-state.partial{{border-color:#fbbf24;color:#fde68a}}#analysis-state.rest{{border-color:#64748b;color:#cbd5e1}}
.word{{opacity:1;margin:.12em;color:#64748b}}.word.pitched{{color:var(--note-color)}}.word.active{{opacity:1;font-weight:850;text-decoration:underline;text-decoration-thickness:.12em;text-underline-offset:.16em;text-shadow:0 0 4px #000,0 0 10px var(--note-color)}}canvas{{width:100%;background:#10162b;border-radius:10px}}#notes{{height:clamp(190px,28vh,260px)}}#staff{{height:300px;margin-bottom:.75rem}}.view-controls{{display:flex;gap:.5rem;align-items:center;margin:.75rem 0;flex-wrap:wrap}}.view-controls button{{border:1px solid #475569;border-radius:999px;padding:.4rem .75rem;min-height:44px}}.view-controls button.active{{background:#334155;color:#fff}}button:focus-visible,summary:focus-visible{{outline:3px solid #fbbf24;outline-offset:3px}}.degree-legend{{display:flex;gap:.55rem;flex-wrap:wrap;font-size:.8rem;color:#cbd5e1}}.degree-legend span::before{{content:'';display:inline-block;width:.65rem;height:.65rem;border-radius:50%;margin-right:.25rem;background:var(--c)}}
details{{margin-top:1rem}}summary{{cursor:pointer;font-weight:700;color:#c4b5fd;min-height:44px;display:flex;align-items:center}}ol{{columns:2;padding-left:1.5rem}}button{{color:inherit;background:transparent;border:0;padding:.45rem;text-align:left;cursor:pointer}}#list button{{min-height:44px}}li.active button{{color:#67e8f9;border-left:3px solid #fbbf24}}.guide{{margin-left:auto;border:1px solid #22d3ee!important;border-radius:999px!important;padding:.45rem .8rem!important}}.guide.active{{background:#164e63}}#guide-status{{font-size:.8rem;color:#94a3b8}}.guide-control{{display:flex;align-items:center;gap:.35rem;font-size:.78rem;color:#cbd5e1;white-space:nowrap}}.guide-control input{{width:92px;accent-color:#22d3ee}}.guide-value{{min-width:3.5rem;color:#67e8f9;font-variant-numeric:tabular-nums}}.guide-envelope{{width:100%;margin:.1rem 0 0;background:#0f172a;border-radius:10px;padding:0 .65rem}}.guide-envelope summary{{min-height:34px;font-size:.82rem}}.envelope-grid{{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:.65rem;padding:.35rem 0 .65rem}}.envelope-grid .guide-control input{{width:100%}}@media(max-width:700px){{ol{{columns:1}}#lyrics{{min-height:5rem}}.performance-status{{grid-template-columns:1fr 1fr}}#staff,#notes{{height:230px}}.envelope-grid{{grid-template-columns:1fr 1fr}}}}</style></head>
<body><main><header><img src="cover.jpg" alt="Album cover"><div><h1>{album_name} — Karaoke</h1><p>{artist} · instrumental playback · Whisper lyrics · detected vocal notes</p></div></header>
<span id="now-playing">Track: —</span><audio id="audio" controls preload="metadata"></audio><div class="performance-status"><div class="status-card"><small>SONG KEY</small><strong id="key">—</strong></div><div class="status-card"><small>CURRENT NOTE</small><div id="note-line" aria-live="polite">—</div></div></div><div id="lyrics">Select a track</div>
<div class="view-controls" role="group" aria-label="Playback and timing controls"><div class="degree-legend"><span style="--c:#22c55e">root</span><span style="--c:#22d3ee">2</span><span style="--c:#3b82f6">3</span><span style="--c:#a855f7">4</span><span style="--c:#ec4899">5</span><span style="--c:#f97316">6</span><span style="--c:#facc15">7</span></div><label class="guide-control">Lyrics timing <input id="lyric-bias" type="range" min="-500" max="500" step="10" value="0"><output id="lyric-bias-value" class="guide-value">on time</output></label><button id="guide-toggle" class="guide" type="button" aria-pressed="false">Humming guide: off</button><label class="guide-control">Volume <input id="guide-volume" type="range" min="0" max="500" step="10" value="180"><output id="guide-volume-value" class="guide-value">180%</output></label><label class="guide-control">Hum timing <input id="guide-bias" type="range" min="-500" max="500" step="10" value="0"><output id="guide-bias-value" class="guide-value">on time</output></label><span id="guide-status">SSLI warm alto hum</span><details class="guide-envelope"><summary>Hum envelope (ADSR)</summary><div class="envelope-grid"><label class="guide-control">Attack <input id="adsr-a" type="range" min="1" max="500" value="25"><output id="adsr-a-value" class="guide-value">25 ms</output></label><label class="guide-control">Decay <input id="adsr-d" type="range" min="1" max="500" value="120"><output id="adsr-d-value" class="guide-value">120 ms</output></label><label class="guide-control">Sustain <input id="adsr-s" type="range" min="0" max="100" value="82"><output id="adsr-s-value" class="guide-value">82%</output></label><label class="guide-control">Release <input id="adsr-r" type="range" min="1" max="1000" value="180"><output id="adsr-r-value" class="guide-value">180 ms</output></label></div></details></div>
<div id="analysis-state" role="status" aria-live="polite">Preparing vocal guide</div><canvas id="staff" width="1000" height="400" style="height:400px" aria-label="Two-measure grand staff with synchronized vocal notes, whole-word lyrics, and proportional rhythm"></canvas><details id="track-picker" open><summary>Choose track</summary><ol id="list"></ol></details><iframe id="ssli-engine" src="karaoke/ssli/index.html?exhibitHost=1" title="SSLI guide voice engine" hidden></iframe>
<script>const tracks={payload};let current=0,animationFrame=null,guideEnabled=false,guideReady=false,guideInstrument=null,guideNodes=[],guideScheduled=new Set(),guideVolume=1.8,guideBias=0,lyricBias=0,guideAdsr={{a:25,d:120,s:82,r:180}};window.karaokeGuideTelemetry={{scheduled:0,cancelled:0}};const audio=document.querySelector('#audio'),lyrics=document.querySelector('#lyrics'),noteLine=document.querySelector('#note-line'),nowPlaying=document.querySelector('#now-playing'),analysisState=document.querySelector('#analysis-state'),staff=document.querySelector('#staff'),sctx=staff.getContext('2d'),list=document.querySelector('#list'),keyline=document.querySelector('#key'),guideToggle=document.querySelector('#guide-toggle'),guideStatus=document.querySelector('#guide-status'),guideVolumeInput=document.querySelector('#guide-volume'),guideVolumeValue=document.querySelector('#guide-volume-value'),guideBiasInput=document.querySelector('#guide-bias'),guideBiasValue=document.querySelector('#guide-bias-value'),lyricBiasInput=document.querySelector('#lyric-bias'),lyricBiasValue=document.querySelector('#lyric-bias-value'),degreeColors=['#22c55e','#22d3ee','#3b82f6','#a855f7','#ec4899','#f97316','#facc15'],letterNames=['C','D','E','F','G','A','B'];
function escapeHtml(value){{return String(value).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]))}}
function resetGuide(){{guideNodes.forEach(node=>{{try{{node.stop()}}catch{{}}}});window.karaokeGuideTelemetry.cancelled+=guideNodes.length;guideNodes=[];guideScheduled.clear();const SL=ssliHost();SL?.audio?.stopAllSustained?.();SL?.audio?.clearExpression?.()}}
function load(i,play=true){{resetGuide();current=i;const t=tracks[i],ambiguous=t.detected_key.status==='ambiguous',display=ambiguous?`Likely ${{t.detected_key.label}}`:t.detected_key.label||t.detected_key.display;audio.src=t.audio;nowPlaying.textContent=`Track ${{i+1}} of ${{tracks.length}}: ${{t.title}}`;keyline.textContent=display;keyline.title=`${{ambiguous?'Low confidence; ':''}}confidence ${{t.detected_key.confidence.toFixed(2)}}${{t.detected_key.runner_up?.label?`; alternate ${{t.detected_key.runner_up.label}}`:''}}`;[...list.children].forEach((li,n)=>{{li.classList.toggle('active',n===i);li.querySelector('button').setAttribute('aria-current',n===i?'true':'false')}});render();if(play)audio.play().catch(()=>{{}})}}
function activeSegment(t,time){{return t.segments.find(s=>time>=s.start&&time<=s.end)||null}}
function effectiveEnd(w){{return Math.min(w.end,w.start+2.5)}}
function degreeIndex(note,key){{const noteLetter=(note||'C')[0].toUpperCase(),root=(key?.tonic||'C')[0].toUpperCase();return (letterNames.indexOf(noteLetter)-letterNames.indexOf(root)+7)%7}}
function degreeColor(note,key){{return degreeColors[degreeIndex(note,key)]}}
function isBlackPitch(midi){{return [1,3,6,8,10].includes(((Math.round(midi)%12)+12)%12)}}
function degreeTextColor(note,key){{const color=degreeColor(note,key);return ['#3b82f6','#a855f7','#ec4899'].includes(color)?'#fff':'#0f172a'}}
function staffY(note){{const m=(note||'').match(/^([A-G])(?:[#♯b♭])?(-?\\d+)$/);if(!m)return 150;const diatonic=Number(m[2])*7+letterNames.indexOf(m[1]);return 170-(diatonic-30)*9}}
function keySignatureFor(key){{const table={{'C major':0,'G major':1,'D major':2,'A major':3,'E major':4,'B major':5,'F# major':6,'C# major':7,'F major':-1,'Bb major':-2,'Eb major':-3,'Ab major':-4,'Db major':-5,'Gb major':-6,'Cb major':-7,'A minor':0,'E minor':1,'B minor':2,'F# minor':3,'C# minor':4,'G# minor':5,'D# minor':6,'A# minor':7,'D minor':-1,'G minor':-2,'C minor':-3,'F minor':-4,'Bb minor':-5,'Eb minor':-6,'Ab minor':-7}},count=table[`${{key.tonic}} ${{key.mode}}`]??0;return {{count:Math.abs(count),symbol:count>=0?'♯':'♭',kind:count>=0?'sharp':'flat'}}}}
function lyricColorNote(segment,word,key){{if(word?.note)return word.note;const center=((word?.start||0)+(word?.end||0))/2,candidates=(segment?.words||[]).filter(w=>w.note).map(w=>({{note:w.note,distance:Math.abs(((w.start+w.end)/2)-center)}})).sort((a,b)=>a.distance-b.distance);return candidates[0]?.distance<=1.5?candidates[0].note:`${{key?.tonic||'C'}}4`}}
function drawKeySignature(t){{const sig=keySignatureFor(t.detected_key),treble=sig.kind==='sharp'?[98,125,89,116,143,107,134]:[134,107,143,116,152,125,161],bass=sig.kind==='sharp'?[215,242,206,233,269,224,251]:[251,224,269,233,278,242,287];sctx.save();sctx.fillStyle='#e2e8f0';sctx.font='bold 24px serif';for(let i=0;i<sig.count;i++){{const x=58+i*15;sctx.fillText(sig.symbol,x,treble[i]+7);sctx.fillText(sig.symbol,x,bass[i]+7)}}sctx.restore();return 72+sig.count*15}}
function drawTimeSignature(t,x){{const meter=t.rhythm?.meter||{{numerator:4,denominator:4}};sctx.save();sctx.fillStyle='#f8fafc';sctx.font='bold 25px serif';sctx.textAlign='center';[134,242].forEach(mid=>{{sctx.fillText(String(meter.numerator),x,mid-4);sctx.fillText(String(meter.denominator),x,mid+20)}});sctx.restore();return x+26}}
function drawRhythmicNote(w,x,endX,y,color,bpm,active){{sctx.save();sctx.lineCap='round';sctx.strokeStyle=color;sctx.fillStyle=color;sctx.globalAlpha=active?1:(w.estimated?.62:.78);sctx.shadowColor=active?'#fff':'#000';sctx.shadowBlur=active?10:5;sctx.lineWidth=active?9:7;if(w.estimated)sctx.setLineDash([8,5]);sctx.beginPath();sctx.moveTo(x+5,y);sctx.lineTo(Math.max(x+7,endX),y);sctx.stroke();sctx.setLineDash([]);sctx.shadowBlur=0;sctx.globalAlpha=1;sctx.beginPath();sctx.arc(x,y,active?8:7,0,Math.PI*2);w.estimated?sctx.stroke():sctx.fill();sctx.strokeStyle='#f8fafc';sctx.lineWidth=active?2:1;sctx.beginPath();sctx.moveTo(x,y-10);sctx.lineTo(x,y+10);sctx.stroke();sctx.restore()}}
function notationWindow(t,time){{const beats=Math.max(1,t.rhythm?.meter?.numerator||4),beatSeconds=60/Math.max(1,t.rhythm?.tempo_bpm||120),measureSeconds=beats*beatSeconds,span=measureSeconds*2,start=Math.max(0,time-span*.5);return {{start,end:start+span,span,measureSeconds}}}}
function syllabify(value){{if(value&&Array.isArray(value.syllables)&&value.syllables.length)return value.syllables;const raw=String(value?.text??value??'');return raw?[raw]:[]}}
function syllableSpans(word,wordNotes){{const syllables=syllabify(word),start=word.start,end=effectiveEnd(word);if(wordNotes.length<syllables.length)return syllables.map((text,index)=>({{text,start:start+(end-start)*index/syllables.length,end:start+(end-start)*(index+1)/syllables.length,note:wordNotes[index]||wordNotes[0]||word}}));return syllables.map((text,index)=>{{const from=Math.floor(index*wordNotes.length/syllables.length),to=Math.max(from,Math.floor((index+1)*wordNotes.length/syllables.length)-1),assigned=wordNotes.slice(from,to+1),syllableStart=index===0?Math.min(start,assigned[0].start):assigned[0].start,syllableEnd=index===syllables.length-1?Math.max(end,assigned[assigned.length-1].end):assigned[assigned.length-1].end;return {{text,start:syllableStart,end:syllableEnd,note:assigned[0]||word}}}})}}
function drawSpreadText(text,startX,endX,y){{const value=String(text),chars=[...value],span=Math.max(0,endX-startX),natural=sctx.measureText(value).width;sctx.textAlign='left';if(chars.length<2||span<=natural+6){{sctx.fillText(value,startX,y);return}}const lastWidth=sctx.measureText(chars.at(-1)).width,usable=Math.max(0,span-lastWidth),step=usable/(chars.length-1);chars.forEach((char,index)=>sctx.fillText(char,startX+step*index,y))}}
function noteAtFraction(notes,fraction){{if(!notes.length)return null;const start=notes[0].start,end=notes.at(-1).end,target=start+(end-start)*fraction;return notes.find(note=>target>=note.start&&target<note.end)||notes.reduce((best,note)=>Math.abs((note.start+note.end)/2-target)<Math.abs((best.start+best.end)/2-target)?note:best,notes[0])}}
function drawColoredWord(text,x,y,notes,key,active,fallback){{let cursor=x;const chars=[...String(text)];sctx.textAlign='left';chars.forEach((char,index)=>{{const note=noteAtFraction(notes,(index+.5)/Math.max(1,chars.length)),color=degreeColor(note?.note||fallback,key);sctx.fillStyle=color;sctx.shadowColor=active?color:'#000';sctx.shadowBlur=active?11:4;sctx.fillText(char,cursor,y);cursor+=sctx.measureText(char).width}});sctx.shadowBlur=0;return cursor}}
function drawTimedSyllable(text,startX,endX,y){{const value=String(text),span=Math.max(10,endX-startX-4);let size=19;sctx.font=`800 ${{size}}px system-ui`;const natural=sctx.measureText(value).width;if(natural>span)size=Math.max(13,Math.floor(size*span/natural));sctx.font=`800 ${{size}}px system-ui`;const width=sctx.measureText(value).width;sctx.textAlign='center';sctx.fillText(value,(startX+endX)/2,y);return width}}
function stableWordLaneMap(t){{stableWordLaneMap.cache??=new WeakMap();if(stableWordLaneMap.cache.has(t))return stableWordLaneMap.cache.get(t);const map=new Map(),right=[-Infinity,-Infinity,-Infinity,-Infinity],secondsPerPixel=notationWindow(t,0).span/850,phraseByWord=new Map(),lastLaneByPhrase=new Map();t.segments.forEach((segment,index)=>segment.words.forEach(word=>phraseByWord.set(word.word_index,index)));sctx.save();sctx.font='800 19px system-ui';t.words.forEach(word=>{{const notes=t.note_events.filter(note=>note.word_index===word.word_index),start=(notes[0]?.start??word.start),widthSeconds=(sctx.measureText(String(word.text||'')).width+10)*secondsPerPixel,phrase=phraseByWord.get(word.word_index),preferred=lastLaneByPhrase.get(phrase),available=right.map((value,lane)=>start>=value?lane:null).filter(lane=>lane!==null);let lane;if(preferred!==undefined&&available.includes(preferred))lane=preferred;else if(available.length)lane=preferred===undefined?available[0]:available.sort((a,b)=>Math.abs(a-preferred)-Math.abs(b-preferred))[0];else lane=right.indexOf(Math.min(...right));map.set(word,lane);lastLaneByPhrase.set(phrase,lane);right[lane]=start+widthSeconds}});sctx.restore();stableWordLaneMap.cache.set(t,map);return map}}
function wordIndexMap(t){{wordIndexMap.cache??=new WeakMap();if(!wordIndexMap.cache.has(t))wordIndexMap.cache.set(t,new Map(t.words.map(word=>[word.word_index,word])));return wordIndexMap.cache.get(t)}}
function linkedNoteEvents(t){{linkedNoteEvents.cache??=new WeakMap();if(!linkedNoteEvents.cache.has(t)){{const words=wordIndexMap(t);linkedNoteEvents.cache.set(t,t.note_events.filter(note=>words.has(note.word_index)))}}return linkedNoteEvents.cache.get(t)}}
function visualNoteEvents(t){{visualNoteEvents.cache??=new WeakMap();if(visualNoteEvents.cache.has(t))return visualNoteEvents.cache.get(t);const linked=linkedNoteEvents(t),byWord=new Set(linked.map(note=>note.word_index)),events=[...linked];t.words.forEach(word=>{{if(byWord.has(word.word_index))return;const center=(word.start+effectiveEnd(word))/2,nearest=linked.reduce((best,note)=>{{const distance=Math.abs((note.start+note.end)/2-center);return !best||distance<best.distance?{{note,distance}}:best}},null),fallback=word.note?{{note:word.note,midi:word.midi}}:nearest?.distance<=1.5?nearest.note:null;if(fallback?.note&&Number.isFinite(Number(fallback.midi)))events.push({{start:word.start,end:effectiveEnd(word),note:fallback.note,midi:Number(fallback.midi),word_index:word.word_index,estimated:true}})}});events.sort((a,b)=>a.start-b.start);visualNoteEvents.cache.set(t,events);return events}}
function syllableAttackTimes(word,notes){{const count=Math.max(1,word.syllables?.length||1),start=notes[0]?.start??word.start,end=Math.max(effectiveEnd(word),notes.at(-1)?.end??word.end);if(count===1)return [start];if(notes.length>=count)return Array.from({{length:count}},(_,index)=>notes[Math.floor(index*notes.length/count)].start);return Array.from({{length:count}},(_,index)=>start+(end-start)*index/count)}}
function drawSyllableAttacks(t,xFor,windowStart,windowEnd){{t.words.forEach(word=>{{const notes=visualNoteEvents(t).filter(note=>note.word_index===word.word_index);if(!notes.length)return;syllableAttackTimes(word,notes).filter(time=>time>=windowStart&&time<=windowEnd).forEach(time=>{{const note=notes.find(event=>time>=event.start&&time<=event.end)||notes.reduce((best,event)=>Math.abs(event.start-time)<Math.abs(best.start-time)?event:best,notes[0]),x=xFor(time),y=staffY(note.note);sctx.save();sctx.strokeStyle='#f8fafc';sctx.globalAlpha=.9;sctx.lineWidth=2;sctx.beginPath();sctx.moveTo(x,y-11);sctx.lineTo(x,y+11);sctx.stroke();sctx.restore()}})}})}}
function guideWindowState(t,windowStart,windowEnd){{const words=t.words.filter(w=>w.end>=windowStart&&w.start<=windowEnd),visual=visualNoteEvents(t).filter(n=>n.end>=windowStart&&n.start<=windowEnd),visualWords=new Set(visual.map(n=>n.word_index)),estimatedWords=new Set(visual.filter(n=>n.estimated).map(n=>n.word_index)),unresolved=words.filter(w=>!visualWords.has(w.word_index)).length,orphan=t.note_events.filter(n=>n.end>=windowStart&&n.start<=windowEnd&&!wordIndexMap(t).has(n.word_index)).length;if(!words.length&&!visual.length)return {{kind:'rest',label:'Instrumental or rest · no mapped vocal event in this window'}};if(estimatedWords.size||unresolved||orphan)return {{kind:'partial',label:`Automated partial guide · estimated pitch for ${{estimatedWords.size}} lyric${{estimatedWords.size===1?'':'s'}}${{unresolved?` · ${{unresolved}} lyric${{unresolved===1?'':'s'}} hidden`:''}}${{orphan?` · ${{orphan}} unlinked pitch event${{orphan===1?'':'s'}} hidden`:''}} · lyrics unverified`}};return {{kind:'mapped',label:'Automated vocal guide · mapping complete · lyrics unverified'}}}}
function drawTimelineLyrics(t,time,xFor,windowStart,windowEnd){{const shiftedTime=time-lyricBias,visible=t.words.filter(w=>w.end+lyricBias>=windowStart&&w.start+lyricBias<=windowEnd),active=activeNoteAt(t,shiftedTime),laneMap=stableWordLaneMap(t),clipLeft=xFor(windowStart);sctx.save();sctx.globalAlpha=1;sctx.fillStyle='rgba(3,7,18,.92)';sctx.fillRect(0,290,staff.width,102);sctx.textBaseline='middle';sctx.fillStyle='#94a3b8';sctx.font='700 11px system-ui';sctx.textAlign='left';sctx.fillText('WORDS',6,338);sctx.save();sctx.beginPath();sctx.rect(clipLeft,290,staff.width-clipLeft,102);sctx.clip();visible.forEach(w=>{{const wordNotes=visualNoteEvents(t).filter(n=>n.word_index===w.word_index);if(!wordNotes.length)return;const firstNote=wordNotes[0],start=firstNote.start+lyricBias,end=Math.max(effectiveEnd(w),Math.max(...wordNotes.map(n=>n.end)))+lyricBias,text=String(w.text||'').trim(),x=xFor(start),isActive=active?.word_index===w.word_index;if(!text)return;sctx.font='800 19px system-ui';const textWidth=sctx.measureText(text).width;if(x<clipLeft+8||x+textWidth>staff.width-8)return;const lane=laneMap.get(w)??0,wordY=304+lane*23;sctx.globalAlpha=isActive?1:(start>time?0.58:0.84);drawColoredWord(text,x,wordY,wordNotes,t.detected_key,isActive,w.note);sctx.lineWidth=isActive?4:2;wordNotes.forEach(note=>{{sctx.strokeStyle=degreeColor(note.note,t.detected_key);sctx.setLineDash(note.estimated?[6,4]:[]);sctx.beginPath();sctx.moveTo(xFor(note.start+lyricBias),wordY+10);sctx.lineTo(Math.max(xFor(note.start+lyricBias)+2,xFor(note.end+lyricBias)),wordY+10);sctx.stroke();sctx.setLineDash([])}})}});sctx.restore();sctx.restore()}}
function drawStaff(t,time,notes,windowStart,windowEnd,state){{sctx.clearRect(0,0,staff.width,staff.height);sctx.fillStyle='#10162b';sctx.fillRect(0,0,staff.width,staff.height);sctx.strokeStyle='#94a3b8';sctx.lineWidth=1.5;[98,116,134,152,170,206,224,242,260,278].forEach(y=>{{sctx.beginPath();sctx.moveTo(54,y);sctx.lineTo(staff.width,y);sctx.stroke()}});sctx.fillStyle='#e2e8f0';sctx.font='44px serif';sctx.fillText('𝄞',10,154);sctx.fillText('𝄢',10,266);const plotStart=drawTimeSignature(t,drawKeySignature(t)+10),span=staff.width-plotStart,xFor=seconds=>plotStart+(seconds-windowStart)/(windowEnd-windowStart)*span;(t.rhythm?.beats||[]).filter(b=>b.time>=windowStart&&b.time<=windowEnd).forEach(b=>{{const x=xFor(b.time);sctx.strokeStyle=b.beat===1?'#cbd5e1':'#475569';sctx.lineWidth=b.beat===1?2:1;sctx.beginPath();sctx.moveTo(x,82);sctx.lineTo(x,278);sctx.stroke()}});(t.rhythm?.measures||[]).filter(m=>m.start>=windowStart&&m.start<=windowEnd).forEach(m=>{{const x=xFor(m.start);sctx.fillStyle='#cbd5e1';sctx.font='bold 12px system-ui';sctx.fillText(`m. ${{m.number}}`,x+4,78)}});sctx.save();sctx.beginPath();sctx.rect(plotStart,60,staff.width-plotStart,225);sctx.clip();notes.forEach(w=>drawRhythmicNote(w,xFor(w.start),xFor(w.end),staffY(w.note),degreeColor(w.note,t.detected_key),t.rhythm?.tempo_bpm||120,isEventActive(w,time)));drawSyllableAttacks(t,xFor,windowStart,windowEnd);sctx.restore();drawTimelineLyrics(t,time,xFor,windowStart,windowEnd);sctx.strokeStyle='#f43f5e';sctx.lineWidth=2;sctx.beginPath();sctx.moveTo(xFor(time),60);sctx.lineTo(xFor(time),staff.height);sctx.stroke()}}
function phraseHtml(segment,activeWord,key,kind){{if(!segment)return '';return `<div class="lyric-phrase ${{kind}}"${{kind==='next'?' style="opacity:.5"':''}}>${{segment.words.map(w=>{{const active=w===activeWord,colorNote=lyricColorNote(segment,w,key),color=degreeColor(colorNote,key);return `<span class="word pitched ${{active?'active':''}}" style="--note-color:${{color}}">${{escapeHtml(w.text)}}</span>`}}).join(' ')}}</div>`}}
function isEventActive(event,time){{return time>=event.start&&time<event.end}}
function activeNoteAt(t,time){{const exact=visualNoteEvents(t).filter(n=>isEventActive(n,time));return exact.sort((a,b)=>b.start-a.start)[0]||null}}
function lyricTimeline(t,time){{let index=-1;for(let i=0;i<t.segments.length;i++){{if(t.segments[i].start<=time)index=i;else break}}const candidate=index>=0?t.segments[index]:null,currentPhrase=candidate&&time<=candidate.end+.12?candidate:null,nextPhrase=t.segments[index+1]||null,progress=currentPhrase?Math.max(0,Math.min(1,(time-currentPhrase.start)/Math.max(.01,currentPhrase.end-currentPhrase.start))):0;return {{index,currentPhrase,nextPhrase,progress}}}}
function render(){{const t=tracks[current],time=audio.currentTime,activeSungNote=activeNoteAt(t,time),timeline=lyricTimeline(t,time),s=timeline.currentPhrase,next=timeline.nextPhrase,activeWord=s?(s.words.find(w=>w.word_index===activeSungNote?.word_index)||s.words.find(w=>!t.note_events.some(n=>n.word_index===w.word_index)&&time>=w.start&&time<effectiveEnd(w))):null,activeNote=activeSungNote?.note||(activeWord?lyricColorNote(s,activeWord,t.detected_key):null),wait=next?Math.max(0,Math.ceil(next.start-time)):null;lyrics.style.setProperty('--phrase-progress',timeline.progress);noteLine.textContent=activeNote?`${{activeSungNote?'':'≈ '}}${{activeNote}} · degree ${{degreeIndex(activeNote,t.detected_key)+1}}`:'—';lyrics.innerHTML=s?phraseHtml(s,activeWord,t.detected_key,'current')+phraseHtml(next,null,t.detected_key,'next'):t.status==='rejected_probable_hallucination'?'No reliable lyrics detected for this track.':next?`<div class="lyric-phrase current"><span class="word">Instrumental passage · next lyric in ${{wait}}s</span></div>${{phraseHtml(next,null,t.detected_key,'next')}}`:'<span class="word">Instrumental passage · no more lyrics in this track</span>';
const viewport=notationWindow(t,time),windowStart=viewport.start,windowEnd=viewport.end,state=guideWindowState(t,windowStart,windowEnd),notes=visualNoteEvents(t).filter(w=>w.end>=windowStart&&w.start<=windowEnd);analysisState.className=state.kind;analysisState.textContent=state.label;drawStaff(t,time,notes,windowStart,windowEnd,state)}}
function setView(){{staff.hidden=false;render()}}
function ssliHost(){{try{{const w=document.querySelector('#ssli-engine').contentWindow;return w?.SynthLab?.audio&&w?.SynthLab?.presets?w.SynthLab:null}}catch(e){{return null}}}}
function applyGuideVolume(){{const SL=ssliHost(),id=guideInstrument?.id??guideInstrument,slot=SL?.audio?.getInstruments?.()?.find(x=>x.id===id)||SL?.audio?.getInstruments?.()?.[id];if(slot?.masterOutput?.gain)slot.masterOutput.gain.value=guideVolume}}
function applyGuideAdsr(){{const SL=ssliHost(),id=guideInstrument?.id??guideInstrument,slot=SL?.audio?.getInstruments?.()?.find(x=>x.id===id)||SL?.audio?.getInstruments?.()?.[id];if(!slot?.settings||!SL?.audio?.timeToSlider)return;slot.settings.adsr={{a:SL.audio.timeToSlider(guideAdsr.a,500,500),d:SL.audio.timeToSlider(guideAdsr.d,500,500),s:guideAdsr.s,r:SL.audio.timeToSlider(guideAdsr.r,1000,1000)}}}}
function timingLabel(ms){{return ms===0?'on time':`${{Math.abs(ms)}} ms ${{ms<0?'early':'late'}}`}}
async function initGuide(){{const SL=ssliHost();if(!SL){{guideStatus.textContent='SSLI engine is still loading';return false}}try{{const ctx=SL.audio.getCtx?.();if(ctx?.resume)await ctx.resume();SL.audio.initEffectChain?.();if(SL.formant?.init)await SL.formant.init(ctx);const inst=SL.audio.getCurrentInstrument?.()??0,alto=SL.presets.getFormantPresets?.().find(p=>p.name==='Alto Choir');if(!alto||!SL.presets.applyFormantPreset)throw new Error('Alto Choir preset missing');SL.presets.applyFormantPreset(alto,inst);SL.audio.setFormantSettings?.(inst,{{vowel:'O',vowelTarget:'U',morphX:18,breathiness:8,glottalPulseWidth:58,vowelSequenceEnabled:false}});guideInstrument=inst;const slot=SL.audio.getInstruments?.()?.find(x=>x.id===inst)||SL.audio.getInstruments?.()?.[inst];if(slot?.type!=='formant'||!slot.masterOutput)throw new Error('formant output did not initialize');applyGuideVolume();applyGuideAdsr();guideReady=true;guideStatus.textContent='SSLI warm alto hum · ready';return true}}catch(e){{guideStatus.textContent=`SSLI unavailable: ${{e.message||e}}`;return false}}}}
function scheduleHumNode(ctx,output,midi,startAt,duration){{const fundamental=440*Math.pow(2,(midi-69)/12),mix=ctx.createGain(),filter=ctx.createBiquadFilter(),envelope=ctx.createGain(),osc1=ctx.createOscillator(),osc2=ctx.createOscillator(),attack=guideAdsr.a/1000,decay=guideAdsr.d/1000,release=guideAdsr.r/1000,sustain=Math.max(.001,guideAdsr.s/100);osc1.type='sine';osc2.type='triangle';osc1.frequency.setValueAtTime(fundamental,startAt);osc2.frequency.setValueAtTime(fundamental*2,startAt);mix.gain.setValueAtTime(.11,startAt);filter.type='lowpass';filter.frequency.setValueAtTime(1400,startAt);filter.Q.setValueAtTime(.7,startAt);envelope.gain.setValueAtTime(.0001,startAt);envelope.gain.linearRampToValueAtTime(.22,startAt+attack);envelope.gain.linearRampToValueAtTime(.22*sustain,startAt+attack+decay);const releaseAt=Math.max(startAt+attack+decay,startAt+duration);envelope.gain.setValueAtTime(.22*sustain,releaseAt);envelope.gain.exponentialRampToValueAtTime(.0001,releaseAt+release);osc1.connect(mix);osc2.connect(mix);mix.connect(filter);filter.connect(envelope);envelope.connect(output);osc1.start(startAt);osc2.start(startAt);osc1.stop(releaseAt+release+.02);osc2.stop(releaseAt+release+.02);guideNodes.push(osc1,osc2);window.karaokeGuideTelemetry.scheduled++}}
function scheduleGuide(){{if(!guideEnabled||!guideReady||audio.paused)return;const SL=ssliHost(),id=guideInstrument?.id??guideInstrument,ctx=SL?.audio?.getCtx?.(),slot=SL?.audio?.getInstruments?.()?.find(x=>x.id===id)||SL?.audio?.getInstruments?.()?.[id];if(!ctx||!slot?.masterOutput)return;const now=audio.currentTime,horizon=now+2,rate=Math.max(.25,audio.playbackRate||1);tracks[current].note_events.filter(w=>{{const biased=w.start+guideBias;return biased<=horizon&&w.end+guideBias>=now-.04}}).forEach(w=>{{const key=`${{current}}:${{w.start}}:${{w.end}}:${{w.midi}}`;if(guideScheduled.has(key))return;guideScheduled.add(key);const biased=w.start+guideBias,startAt=ctx.currentTime+Math.max(0,biased-now)/rate,remaining=Math.max(.06,(w.end+guideBias-Math.max(now,biased))/rate);scheduleHumNode(ctx,slot.masterOutput,Math.round(w.midi),startAt,remaining)}})}}
async function toggleGuide(){{guideEnabled=!guideEnabled;if(guideEnabled&&!guideReady)guideEnabled=await initGuide();if(!guideEnabled)resetGuide();guideToggle.classList.toggle('active',guideEnabled);guideToggle.setAttribute('aria-pressed',String(guideEnabled));guideToggle.textContent=`Humming guide: ${{guideEnabled?'on':'off'}}`}}
function animateNotation(){{render();scheduleGuide();if(!audio.paused&&!audio.ended)animationFrame=requestAnimationFrame(animateNotation)}}
function startNotation(){{if(animationFrame)cancelAnimationFrame(animationFrame);animationFrame=requestAnimationFrame(animateNotation)}}
function stopNotation(){{if(animationFrame)cancelAnimationFrame(animationFrame);animationFrame=null;render()}}
lyricBiasInput.oninput=()=>{{lyricBias=Number(lyricBiasInput.value)/1000;lyricBiasValue.textContent=timingLabel(Number(lyricBiasInput.value));render()}};
tracks.forEach((t,i)=>{{const li=document.createElement('li'),b=document.createElement('button');b.textContent=t.title;b.onclick=()=>load(i);li.appendChild(b);list.appendChild(li)}});document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>setView(b.dataset.view));guideToggle.onclick=toggleGuide;guideVolumeInput.oninput=()=>{{guideVolume=Number(guideVolumeInput.value)/100;guideVolumeValue.textContent=`${{guideVolumeInput.value}}%`;applyGuideVolume()}};guideBiasInput.oninput=()=>{{guideBias=Number(guideBiasInput.value)/1000;guideBiasValue.textContent=timingLabel(Number(guideBiasInput.value));resetGuide()}};[['a','ms'],['d','ms'],['s','%'],['r','ms']].forEach(([name,unit])=>{{const input=document.querySelector(`#adsr-${{name}}`),output=document.querySelector(`#adsr-${{name}}-value`);input.oninput=()=>{{guideAdsr[name]=Number(input.value);output.textContent=`${{input.value}} ${{unit}}`;applyGuideAdsr()}}}});document.querySelector('#ssli-engine').addEventListener('load',()=>{{guideStatus.textContent='SSLI warm alto hum · click to enable'}});audio.addEventListener('play',startNotation);audio.addEventListener('pause',()=>{{resetGuide();stopNotation()}});audio.addEventListener('seeking',resetGuide);audio.addEventListener('seeked',render);audio.addEventListener('ended',()=>{{resetGuide();stopNotation();if(current+1<tracks.length)load(current+1)}});document.querySelector('#track-picker').open=false;setView('staff');load(0,false);</script></main></body></html>'''
    path = output / f"{raw_album_name} - Karaoke.html"
    path.write_text(html_document, encoding="utf-8")
    return path


def main() -> int:
    """Build a fresh, hash-bound HTML player candidate and data payloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--album-manifest", required=True, type=Path)
    parser.add_argument("--transcripts", required=True, type=Path)
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Fresh rendered-candidate directory; never inferred from an untrusted manifest",
    )
    parser.add_argument("--media-root", required=True, type=Path, help="Trusted packaged-album directory containing audio")
    parser.add_argument("--stem-root", required=True, type=Path, help="Trusted root for manifest-relative stem paths")
    parser.add_argument("--ssli-root", required=True, type=Path, help="Local SSLI distribution containing index.html")
    parser.add_argument("--analysis-cache-dir", type=Path, help="Optional read-only cache; reused only on exact cache-key match")
    parser.add_argument("--analysis-overrides", type=Path, help="Reviewed per-track key and meter overrides")
    args = parser.parse_args()
    album = json.loads(args.album_manifest.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("rendered candidate directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    media_root = args.media_root.resolve(strict=True)
    stem_root = args.stem_root.resolve(strict=True)
    ssli_root = args.ssli_root.resolve(strict=True)
    try:
        ssli_manifest = validate_manifest(ssli_root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    override_payload = json.loads(args.analysis_overrides.read_text(encoding="utf-8")) if args.analysis_overrides else None
    try:
        overrides = validate_analysis_overrides(override_payload, album["tracks"]) if override_payload else {}
    except ValueError as error:
        parser.error(str(error))
    override_hash = sha256(args.analysis_overrides) if args.analysis_overrides else None
    data_dir = output / "karaoke" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    ssli_destination = output / "karaoke" / "ssli"
    for item in ssli_manifest["files"]:
        source = (ssli_root / item["path"]).resolve(strict=True)
        if not source.is_relative_to(ssli_root) or sha256(source) != item["sha256"]:
            parser.error(f"SSLI file changed during copy: {item['path']}")
        destination = ssli_destination / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copy2(ssli_root / MANIFEST_NAME, ssli_destination / MANIFEST_NAME)
    cover_source = (media_root / "cover.jpg").resolve(strict=True)
    if not cover_source.is_relative_to(media_root) or sha256(cover_source) != album["cover_sha256"]:
        parser.error("packaged cover is missing or does not match the album manifest")
    shutil.copy2(cover_source, output / "cover.jpg")
    failures = []
    player_tracks = []
    for track in album["tracks"]:
        safe_title = safe_component(track["title"], field=f"track {track['number']} title")
        transcript_path = args.transcripts / f"{track['number']:02d} - {safe_title}.whisper.json"
        if not transcript_path.exists():
            failures.append(f"missing transcript: {track['title']}")
            continue
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        if (
            transcript.get("vocal_sha256") != track["vocal_sha256"]
            or int(transcript.get("track", -1)) != int(track["number"])
        ):
            failures.append(f"transcript provenance mismatch: {track['title']}")
            continue
        # Never synthesize timing from a repeated chorus during a production
        # build. Invalid segments are counted and routed back to remediation.
        clean_transcript_segments = [
            segment for segment in transcript["segments"] if not has_unexpected_script(segment.get("text", ""))
        ]
        timing_rejected_segments = sum(bool(segment_timing_errors(segment)) for segment in clean_transcript_segments)
        words = [
            word
            for segment in clean_transcript_segments
            for word in segment["words"]
        ]
        data_path = data_dir / f"{track['number']:02d}.karaoke.json"
        transcript_hash = sha256(transcript_path)
        vocal_path = (stem_root / track["vocal_stem"]).resolve(strict=True)
        accompaniment_path = (stem_root / track["accompaniment_stem"]).resolve(strict=True)
        if not vocal_path.is_relative_to(stem_root) or not accompaniment_path.is_relative_to(stem_root):
            failures.append(f"stem path escapes --stem-root: {track['title']}")
            continue
        if sha256(vocal_path) != track["vocal_sha256"] or sha256(accompaniment_path) != track["accompaniment_sha256"]:
            failures.append(f"stem hash mismatch: {track['title']}")
            continue
        cache_key = analysis_cache_key(track, transcript_hash, override_hash)
        cache_path = args.analysis_cache_dir / f"{track['number']:02d}.karaoke.json" if args.analysis_cache_dir else None
        existing = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path and cache_path.exists() else None
        cache_valid = bool(existing and existing.get("analysis_cache_key") == cache_key)
        track_override = overrides.get(str(track["number"]), {})
        if cache_valid and "pitch_contour" in existing:
            contour = existing["pitch_contour"]
            annotated = annotate_words_from_contour(contour, words)
            pitch_method = PITCH_METHOD
        else:
            annotated, contour = analyze_pitch(vocal_path, words)
            pitch_method = PITCH_METHOD
        for word in annotated:
            word["syllables"] = lyric_syllables(str(word.get("text", "")))
        by_key = {(word["start"], word["end"], word["text"]): word for word in annotated}
        segments = []
        for segment in clean_transcript_segments:
            clean_words = [
                by_key[(w["start"], w["end"], w["text"])]
                for w in segment["words"]
                if (w["start"], w["end"], w["text"]) in by_key
            ]
            if not clean_words:
                continue
            segment_copy = {key: segment[key] for key in ["start", "end"]}
            segment_copy["text"] = " ".join(word["text"] for word in clean_words)
            segment_copy["words"] = clean_words
            segments.append(segment_copy)
        segments, lyric_status = filter_lyric_segments(segments, transcript)
        retained_word_indices = {w["word_index"] for segment in segments for w in segment["words"]}
        retained_words = [w for w in annotated if w["word_index"] in retained_word_indices]
        pitched_midis = [w["midi"] for w in retained_words if w["midi"] is not None]
        packaged_audio = (media_root / track["output_file"]).resolve(strict=True)
        if not packaged_audio.is_relative_to(media_root) or sha256(packaged_audio) != track["output_sha256"]:
            failures.append(f"packaged audio hash mismatch: {track['title']}")
            continue
        candidate_audio = output / "media" / safe_component(track["output_file"], field="packaged audio filename")
        candidate_audio.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(packaged_audio, candidate_audio)
        if sha256(candidate_audio) != track["output_sha256"]:
            failures.append(f"candidate audio copy mismatch: {track['title']}")
            continue
        data = {
            "schema": SCHEMA,
            "track": track["number"],
            "title": track["title"],
            "audio": candidate_audio.relative_to(output).as_posix(),
            "audio_sha256": track["output_sha256"],
            "vocal_sha256": track["vocal_sha256"],
            "transcript_sha256": transcript_hash,
            "analysis_cache_key": cache_key,
            "status": lyric_status,
            "timing_rejected_segments": timing_rejected_segments,
            "segments": segments,
            "words": retained_words,
            "pitch_contour": contour,
            "pitch_method": pitch_method,
            "note_events": contour_to_note_events(contour, retained_words),
            "pitch_range": {
                "min_midi": max(24, math.floor(float(np.percentile(pitched_midis, 5))) - 2) if pitched_midis else 36,
                "max_midi": min(108, math.ceil(float(np.percentile(pitched_midis, 95))) + 2) if pitched_midis else 84,
            },
            "detected_key": apply_key_override(
                existing["detected_key"] if cache_valid and existing.get("detected_key") else detect_key(accompaniment_path),
                track_override.get("key"),
            ),
            "rhythm": analyze_rhythm(accompaniment_path, track_override.get("meter")),
        }
        data_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        player_tracks.append(data)
    html = build_html(output, album, player_tracks) if len(player_tracks) == len(album["tracks"]) else None
    artifact_paths = (
        ([html] if html else [])
        + [output / "cover.jpg"]
        + sorted(data_dir.glob("*.karaoke.json"))
        + sorted((output / "media").glob("*"))
        + sorted(path for path in (output / "karaoke" / "ssli").rglob("*") if path.is_file())
    )
    report = {
        "schema": SCHEMA,
        "verified_utc": datetime.now(UTC).isoformat(),
        "tracks_expected": len(album["tracks"]),
        "tracks_built": len(player_tracks),
        "player": html.name if html else None,
        "album_manifest_sha256": sha256(args.album_manifest),
        "analysis_review_sha256": override_hash,
        "artifacts": [
            {"path": path.relative_to(output).as_posix(), "sha256": sha256(path)} for path in artifact_paths
        ],
        "failures": failures,
        "pass": not failures and html is not None,
    }
    (output / "karaoke-verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
