"""RunPod Serverless entry point using the exact same H3 runtime as Cog."""

from __future__ import annotations

import ipaddress
import os
import shutil
import socket
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import runpod

from h3_compliance import enforce_moderation
from h3_runtime import H3Runtime
from h3_serverless import MAX_IMAGE_BYTES, decode_image_data_url, deliver_video, frame_url
from h3_tuning import authorize_tuning

_runtime = None
def _get_runtime() -> H3Runtime:
    global _runtime
    if _runtime is None:
        _runtime = H3Runtime()
    return _runtime


def _public_https(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("image URLs must use https")
    for result in socket.getaddrinfo(parsed.hostname, parsed.port or 443):
        address = ipaddress.ip_address(result[4][0])
        if not address.is_global:
            raise ValueError("image URL resolves to a private or reserved address")


def _download_image(url: str | None) -> Path | None:
    if not url:
        return None
    if url.startswith("data:"):
        data, suffix = decode_image_data_url(url)
        fd, filename = tempfile.mkstemp(prefix="h3-input-", suffix=suffix)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        return Path(filename)
    _public_https(url)
    request = urllib.request.Request(url, headers={"User-Agent": "appnz-h3-cog/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        length = int(response.headers.get("Content-Length", "0") or 0)
        if length > MAX_IMAGE_BYTES:
            raise ValueError("image exceeds 32 MiB")
        data = response.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image exceeds 32 MiB")
    suffix = Path(urllib.parse.urlparse(url).path).suffix or ".png"
    fd, filename = tempfile.mkstemp(prefix="h3-input-", suffix=suffix)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    return Path(filename)


def handler(event):
    values = event.get("input") or {}
    first = last = None
    generated = None
    moderation = None
    try:
        cache = authorize_tuning(values.get("_tuning"), values.get("_tuning_signature"))
        # The public Cog schema uses first_frame/last_frame. Keep the explicit
        # *_url aliases for direct RunPod callers while making both runtimes
        # accept the same JSON request shape.
        first = _download_image(frame_url(values, "first_frame"))
        last = _download_image(frame_url(values, "last_frame"))
        moderation = enforce_moderation(values, first, last)
        generated = _get_runtime().generate(
            prompt=values.get("prompt", ""),
            first_frame=first,
            last_frame=last,
            aspect_ratio=values.get("aspect_ratio", "16:9"),
            size=values.get("size", "balanced"),
            duration=float(values.get("duration", 5)),
            steps=int(values.get("steps", 20)),
            seed=values.get("seed"),
            structured_prompt=bool(values.get("structured_prompt", True)),
            loop=bool(values.get("loop", False)),
            include_audio=bool(values.get("include_audio", True)),
            output_codec=values.get("output_codec", "webm-av1"),
            encode_quality=int(values.get("encode_quality", 26)),
            cache=cache,
            return_metrics=True,
        )
        return {
            "outputs": [
                deliver_video(
                    generated.path,
                    values,
                    job_id=event.get("id") or event.get("jobId"),
                    ai_content_id=generated.metrics.get("ai_content_id"),
                )
            ],
            "metrics": {**generated.metrics, "moderation": moderation},
        }
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        for path in (first, last):
            if path:
                path.unlink(missing_ok=True)
        if generated:
            shutil.rmtree(generated.path.parent, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
