import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import weights


def test_manifest_uses_project_user_agent(monkeypatch):
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self):
            return b"{}"

    def urlopen(request, timeout):
        seen["user_agent"] = request.get_header("User-agent")
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr(weights.urllib.request, "urlopen", urlopen)

    assert weights._manifest() == {}
    assert seen == {"user_agent": "appnz-h3-cog/0.1", "timeout": 30}


def test_license_acceptance_is_explicit(monkeypatch):
    monkeypatch.delenv("MINIMAX_H3_LICENSE_ACCEPTED", raising=False)
    with pytest.raises(RuntimeError, match="reviewing and accepting"):
        weights.ensure_weights()


def test_download_checks_size_sha_and_resumes(tmp_path):
    payload = b"verified-h3-weight" * 1024

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
            body = payload[start:]
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        target = tmp_path / "model.safetensors"
        partial = target.with_suffix(target.suffix + ".part")
        partial.write_bytes(payload[:37])
        weights._download(
            f"http://127.0.0.1:{server.server_port}/model",
            target,
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        )
        assert target.read_bytes() == payload
        assert not partial.exists()
    finally:
        server.shutdown()
