import threading
import urllib.error
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer

import pytest

from karaokeizer.serve_karaoke import RangeRequestHandler


@pytest.fixture
def local_server(tmp_path):
    (tmp_path / "media.bin").write_bytes(b"0123456789")
    handler = partial(RangeRequestHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(server, path, *, host=None, headers=None):
    port = server.server_port
    request_headers = {"Host": host or f"127.0.0.1:{port}", **(headers or {})}
    return urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=request_headers))


def test_server_supports_ranges_and_security_headers(local_server):
    with request(local_server, "/media.bin", headers={"Range": "bytes=2-5"}) as response:
        assert response.status == 206
        assert response.read() == b"2345"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "connect-src 'none'" in response.headers["Content-Security-Policy"]


def test_server_rejects_bad_host_and_directory_listing(local_server):
    with pytest.raises(urllib.error.HTTPError) as bad_host:
        request(local_server, "/media.bin", host="attacker.example")
    assert bad_host.value.code == 421
    with pytest.raises(urllib.error.HTTPError) as listing:
        request(local_server, "/")
    assert listing.value.code == 403
