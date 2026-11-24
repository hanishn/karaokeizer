import functools
import http.server
import json
import os
import shutil
import sys
import threading
import wave
from pathlib import Path
from urllib.parse import quote

import pytest

from karaokeizer.audit_karaoke_player_visual import main
from karaokeizer.build_karaoke_overlay import build_html

pytestmark = pytest.mark.skipif(
    os.environ.get("KARAOKEIZER_BROWSER_TESTS") != "1",
    reason="set KARAOKEIZER_BROWSER_TESTS=1 after installing Chromium",
)


class SecureStaticHandler(http.server.SimpleHTTPRequestHandler):
    """Static fixture server with the same critical CSP as Karaokeizer."""

    def end_headers(self) -> None:
        frame_ancestors = "'self'" if self.path.split("?", 1)[0].startswith("/karaoke/ssli/") else "'none'"
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; connect-src 'none'; "
            f"frame-ancestors {frame_ancestors}; media-src 'self'",
        )
        super().end_headers()


def test_visual_audit_exercises_the_hash_bound_served_player(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Run Chromium against the actual generated player and a fake SSLI contract."""

    with wave.open(str(tmp_path / "audio.wav"), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(8_000)
        audio_file.writeframes(b"\0\0" * 80_000)
    (tmp_path / "cover.jpg").write_bytes(b"")
    ssli = tmp_path / "karaoke" / "ssli"
    ssli.mkdir(parents=True)
    ssli.joinpath("index.html").write_text(
        """<!doctype html><script>
        const slot={id:0,type:'formant',settings:{},masterOutput:{gain:{value:1}}},events=[];
        window.SynthLab={
          audio:{getCtx:()=>({currentTime:0,resume:async()=>{}}),initEffectChain:()=>{},
            getCurrentInstrument:()=>0,getInstruments:()=>[slot],timeToSlider:v=>v,
            setFormantSettings:()=>{},stopAllSustained:()=>{},clearExpression:()=>{}},
          presets:{getFormantPresets:()=>[{name:'Alto Choir'}],applyFormantPreset:()=>{}},
          formant:{init:async()=>{},scheduleNote:(m,v,i,s,e,id)=>{events.push(
            {type:'scheduleTelemetry',state:'scheduled',scheduleId:id,startFrame:s*48000,endFrame:e*48000},
            {type:'scheduleTelemetry',state:'started',scheduleId:id,requestedFrame:s*48000,actualFrame:s*48000},
            {type:'scheduleTelemetry',state:'released',scheduleId:id,requestedFrame:e*48000,actualFrame:e*48000,peak:.1});},
            cancelSchedule:id=>events.push({type:'scheduleTelemetry',state:'cancelled',scheduleId:id}),
            allNotesOff:()=>{},getScheduleTelemetry:()=>events}
        };
        </script>""",
        encoding="utf-8",
    )
    track = {
        "title": "Song", "audio": "audio.wav",
        "segments": [{"start": 0.1, "end": 0.8, "text": "hello", "words": []}],
        "words": [{"text": "hello", "start": 0.1, "end": 0.8, "word_index": 0}],
        "note_events": [{"start": 0.1, "end": 0.8, "midi": 60, "note": "C4", "word_index": 0}],
        "detected_key": {"status": "confident", "label": "C major", "display": "C major", "confidence": 1.0,
                         "runner_up": {}, "tonic": "C", "mode": "major"},
        "rhythm": {"meter": {"numerator": 4, "denominator": 4}, "measures": [{"start": 0.0, "end": 1.0}],
                   "beats": [0.0, 0.25, 0.5, 0.75]},
        "pitch_range": {"min_midi": 36, "max_midi": 84}, "status": "verified",
    }
    player = build_html(tmp_path, {"config": {"album": "Album", "artist": "Artist"}}, [track])
    handler = functools.partial(SecureStaticHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output, evidence = tmp_path / "audit.json", tmp_path / "evidence"
    try:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "karaokeizer-audit-player", "--url",
                f"http://127.0.0.1:{server.server_port}/{quote(player.name)}",
                "--player", str(player), "--output", str(output),
                "--evidence-dir", str(evidence), "--samples-per-track", "1",
            ],
        )
        assert main() == 0
    finally:
        server.shutdown()
        thread.join(timeout=2)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["pass"] is True
    assert report["guide_voice"]["pass"] is True
    assert report["guide_voice"]["telemetry"]["scheduled"] > 0
    assert report["guide_voice"]["telemetry"]["cancelled"] > 0
    assert any(event.get("peak", 0) > 0 for event in report["guide_voice"]["workletTelemetry"])
    assert report["player_sha256"] == report["served_player_sha256"]
    assert list(evidence.glob("*.png"))


@pytest.mark.skipif(not os.environ.get("KARAOKEIZER_REAL_SSLI_ROOT"), reason="real patched SSLI distribution not supplied")
def test_real_ssli_formant_worklet_is_audible_and_sample_scheduled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Prove the pinned patched SSLI worklet starts, releases, cancels, and emits audio."""

    with wave.open(str(tmp_path / "audio.wav"), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(8_000)
        audio_file.writeframes(b"\0\0" * 80_000)
    (tmp_path / "cover.jpg").write_bytes(b"")
    shutil.copytree(Path(os.environ["KARAOKEIZER_REAL_SSLI_ROOT"]), tmp_path / "karaoke" / "ssli")
    track = {
        "title": "Real Formant", "audio": "audio.wav",
        "segments": [{"start": 0.0, "end": 0.1, "text": "hum", "words": []}],
        "words": [{"text": "hum", "start": 0.1, "end": 0.35, "word_index": 0}],
        "note_events": [
            {"start": 0.1, "end": 0.35, "midi": 60, "note": "C4", "word_index": 0},
            {"start": 2.0, "end": 2.3, "midi": 64, "note": "E4", "word_index": None},
        ],
        "detected_key": {"status": "confident", "label": "C major", "display": "C major", "confidence": 1.0,
                         "runner_up": {}, "tonic": "C", "mode": "major"},
        "rhythm": {"meter": {"numerator": 4, "denominator": 4}, "measures": [{"start": 0.0, "end": 1.0}],
                   "beats": [0.0, 0.25, 0.5, 0.75]},
        "pitch_range": {"min_midi": 36, "max_midi": 84}, "status": "verified",
    }
    player = build_html(tmp_path, {"config": {"album": "Real SSLI", "artist": "Karaokeizer"}}, [track])
    handler = functools.partial(SecureStaticHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output, evidence = tmp_path / "audit.json", tmp_path / "evidence"
    try:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "karaokeizer-audit-player", "--url",
                f"http://127.0.0.1:{server.server_port}/{quote(player.name)}",
                "--player", str(player), "--output", str(output),
                "--evidence-dir", str(evidence), "--samples-per-track", "1", "--main-thread-stall-ms", "3000",
            ],
        )
        assert main() == 0
    finally:
        server.shutdown()
        thread.join(timeout=2)

    guide = json.loads(output.read_text(encoding="utf-8"))["guide_voice"]
    assert guide["pass"] is True
    assert {event["state"] for event in guide["workletTelemetry"]} >= {"scheduled", "started", "released", "cancelled"}
    assert sum(event["state"] == "started" for event in guide["workletTelemetry"]) == 2
    assert sum(event["state"] == "released" for event in guide["workletTelemetry"]) == 2
    assert max(event.get("peak", 0) for event in guide["workletTelemetry"]) > 1e-6
