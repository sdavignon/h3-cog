"""Fail-closed moderation contract for the licensed US RunPod deployment."""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import io
import json
import os
import socket
import urllib.parse
import urllib.request
from pathlib import Path

MAX_MODERATION_RESPONSE_BYTES = 64 * 1024
REQUIRED_ATTESTATIONS = (
    "copyright_rights_attested",
    "likeness_rights_attested",
    "no_model_training_attested",
    "terms_accepted",
)


def _public_https(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise RuntimeError("H3_MODERATION_URL must use public HTTPS")
    for result in socket.getaddrinfo(parsed.hostname, parsed.port or 443):
        address = ipaddress.ip_address(result[4][0])
        if not address.is_global:
            raise RuntimeError("H3_MODERATION_URL resolves to a private or reserved address")


def _frame_payload(role: str, path: Path) -> dict:
    # ComfyUI installs Pillow. Import lazily so dependency-free policy tests can
    # still validate text-only requests.
    from PIL import Image

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((1024, 1024))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=82, optimize=True)
    return {
        "role": role,
        "sha256": digest,
        "media_type": "image/jpeg",
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def enforce_moderation(values: dict, first_frame: Path | None, last_frame: Path | None) -> dict:
    """Require an authenticated allow decision bound to the exact request."""
    url = os.getenv("H3_MODERATION_URL", "").strip()
    secret = os.getenv("H3_MODERATION_SECRET", "").encode()
    if not url or not secret:
        raise RuntimeError("H3 moderation is not configured; generation is blocked")
    _public_https(url)

    user_id = str(values.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("user_id is required for abuse investigation and enforcement")
    attestations = values.get("attestations") or {}
    missing = [name for name in REQUIRED_ATTESTATIONS if attestations.get(name) is not True]
    if missing:
        raise PermissionError("required compliance attestations are missing: " + ", ".join(missing))

    frames = []
    for role, path in (("first_frame", first_frame), ("last_frame", last_frame)):
        if path is not None:
            frames.append(_frame_payload(role, path))
    body = json.dumps(
        {
            "schema_version": 1,
            "model": "MiniMax H3",
            "prompt": str(values.get("prompt", "")),
            "user_id": user_id,
            "attestations": {name: True for name in REQUIRED_ATTESTATIONS},
            "source_frames": frames,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    signature = hmac.new(secret, body, hashlib.sha256).hexdigest()
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "1976-minimax-h3-runpod/1.0",
            "X-H3-Signature-SHA256": signature,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read(MAX_MODERATION_RESPONSE_BYTES + 1)
    if len(raw) > MAX_MODERATION_RESPONSE_BYTES:
        raise RuntimeError("moderation response exceeded 64 KiB")
    decision = json.loads(raw)
    decision_id = str(decision.get("decision_id", "")).strip()
    policy_version = str(decision.get("policy_version", "")).strip()
    if not decision_id or not policy_version or decision.get("allow") is not True:
        suffix = f" (decision_id={decision_id})" if decision_id else ""
        raise PermissionError("moderation denied or returned an invalid decision" + suffix)
    return {"decision_id": decision_id, "policy_version": policy_version}
