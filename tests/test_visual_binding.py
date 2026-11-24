import functools
import http.server
import threading
from pathlib import Path

from karaokeizer.audit_karaoke_player_visual import served_player_sha256, sha256


def test_served_player_hash_matches_exact_local_file(tmp_path: Path):
    player = tmp_path / "player.html"
    player.write_text("<html>candidate</html>", encoding="utf-8")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/player.html"
        assert served_player_sha256(url) == sha256(player)
        candidate = tmp_path / "candidate.html"
        candidate.write_text("<html>different candidate</html>", encoding="utf-8")
        assert served_player_sha256(url) != sha256(candidate)
    finally:
        server.shutdown()
        thread.join(timeout=2)
