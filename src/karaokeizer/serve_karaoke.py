"""Serve local karaoke media with HTTP byte-range support."""

from __future__ import annotations

import argparse
import ipaddress
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class RangeRequestHandler(SimpleHTTPRequestHandler):
    """Serve a fixed local tree with range requests and browser hardening."""

    range_to_send: tuple[int, int] | None = None

    def _valid_host(self) -> bool:
        host = self.headers.get("Host", "").lower()
        port = self.server.server_port
        return host in {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}

    def _reject_bad_host(self) -> bool:
        if self._valid_host():
            return False
        self.send_error(421, "Unrecognized Host header")
        return True

    def do_GET(self) -> None:
        if not self._reject_bad_host():
            super().do_GET()

    def do_HEAD(self) -> None:
        if not self._reject_bad_host():
            super().do_HEAD()

    def list_directory(self, path):
        self.send_error(403, "Directory listing disabled")
        return None

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; media-src 'self'; frame-src 'self'; connect-src 'none'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
        )
        super().end_headers()

    def send_head(self):
        path = Path(self.translate_path(self.path)).resolve()
        root = Path(self.directory).resolve()
        if not path.is_relative_to(root):
            self.send_error(403, "Path escapes server root")
            return None
        if path.is_dir():
            return self.list_directory(str(path))
        try:
            # The HTTP server owns this handle after ``send_head`` returns it;
            # ``SimpleHTTPRequestHandler.copyfile`` closes it after streaming.
            source = open(path, "rb")  # noqa: SIM115
        except OSError:
            self.send_error(404, "File not found")
            return None
        size = os.fstat(source.fileno()).st_size
        start, end = 0, max(0, size - 1)
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip())
            if not match:
                source.close()
                self.send_error(416, "Invalid byte range")
                return None
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            elif last:
                start = max(0, size - int(last))
            if start >= size or start > end:
                source.close()
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return None
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.range_to_send = (start, end)
            source.seek(start)
        else:
            self.send_response(200)
            self.range_to_send = None
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(os.fstat(source.fileno()).st_mtime))
        self.end_headers()
        return source

    def copyfile(self, source, outputfile):
        if self.range_to_send is None:
            return super().copyfile(source, outputfile)
        remaining = self.range_to_send[1] - self.range_to_send[0] + 1
        while remaining:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def main() -> None:
    """Serve one player tree on loopback with range and security headers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    address = ipaddress.ip_address(args.bind)
    if not address.is_loopback:
        parser.error("Karaokeizer serves private media and only permits loopback bind addresses")
    handler = partial(RangeRequestHandler, directory=str(root))
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"Serving {args.root.resolve()} at http://{args.bind}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
