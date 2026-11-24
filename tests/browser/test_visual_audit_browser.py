import functools
import http.server
import json
import os
import sys
import threading
import wave
from pathlib import Path

import pytest

from karaokeizer.audit_karaoke_player_visual import main

pytestmark = pytest.mark.skipif(
    os.environ.get("KARAOKEIZER_BROWSER_TESTS") != "1",
    reason="set KARAOKEIZER_BROWSER_TESTS=1 after installing Chromium",
)


class SecureStaticHandler(http.server.SimpleHTTPRequestHandler):
    """Static fixture server with the same critical CSP as Karaokeizer."""

    def end_headers(self) -> None:
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; connect-src 'none'; "
            "frame-ancestors 'none'; media-src 'self'",
        )
        super().end_headers()


def test_visual_audit_exercises_the_hash_bound_served_player(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Run the real Chromium audit against a minimal contract-compatible player."""

    player = tmp_path / "player.html"
    with wave.open(str(tmp_path / "audio.wav"), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(8_000)
        audio_file.writeframes(b"\0\0" * 8_000)
    player.write_text(
        """<!doctype html><html><body>
        <div id="analysis-state"></div><audio id="audio" src="audio.wav"></audio>
        <ol id="list"><li><button>Song</button></li></ol>
        <canvas id="staff" width="1000" height="400"></canvas>
        <script>
        const tracks=[{title:'Song',segments:[{start:0,end:1}],
          words:[{text:'hello',start:.1,end:.8,word_index:0}],
          note_events:[{start:.1,end:.8,midi:60,word_index:0}]}];
        let current=0,lyricBias=0;const audio=document.querySelector('#audio'),
          analysisState=document.querySelector('#analysis-state'),staff=document.querySelector('#staff'),
          sctx=staff.getContext('2d');
        function render(){} function notationWindow(){return {start:0,end:2}}
        function stableWordLaneMap(t){return new Map(t.words.map(w=>[w,0]))}
        function visualNoteEvents(t){return t.note_events} function effectiveEnd(w){return w.end}
        function drawTimeSignature(){return 100} function drawKeySignature(){return 100}
        document.querySelector('button').onclick=()=>{current=0};
        </script></body></html>""",
        encoding="utf-8",
    )
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
                f"http://127.0.0.1:{server.server_port}/player.html",
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
    assert report["player_sha256"] == report["served_player_sha256"]
    assert list(evidence.glob("*.png"))
