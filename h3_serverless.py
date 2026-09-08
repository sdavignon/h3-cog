"""Small dependency-free helpers shared by the RunPod entry point and tests."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path

MAX_INLINE_BYTES = 7 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024
_IMAGE_DATA_URL = re.compile(
    r"^data:(image/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=]+)$",
    re.IGNORECASE,
)
_IMAGE_SUFFIXES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}


def frame_url(values: dict, name: str) -> str | None:
    """Prefer the public Cog field while retaining the direct-RunPod alias."""
    return values.get(name) or values.get(f"{name}_url")


def decode_image_data_url(value: str) -> tuple[bytes, str]:
    """Decode a bounded PNG/JPEG/WebP data URL for direct RunPod callers."""
    match = _IMAGE_DATA_URL.fullmatch(value.strip())
    if not match:
        raise ValueError("image data URLs must be base64 PNG, JPEG, or WebP")
    media_type = match.group(1).lower()
    try:
        data = base64.b64decode(match.group(2), validate=True)
    except ValueError as exc:
        raise ValueError("image data URL contains invalid base64") from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image exceeds 32 MiB")
    signatures = {
        "image/png": data.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/jpeg": data.startswith(b"\xff\xd8\xff"),
        "image/webp": len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP",
    }
    if not signatures[media_type]:
        raise ValueError("image data does not match its declared media type")
    return data, _IMAGE_SUFFIXES[media_type]


def _safe_job_id(value: object) -> str:
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "")).strip(".-_")[:80]
    return candidate or uuid.uuid4().hex


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deliver_video(
    path: Path,
    values: dict,
    *,
    job_id: object = None,
    volume_root: Path | None = None,
    ai_content_id: str | None = None,
) -> dict:
    """Return a bounded inline result or persist it on the RunPod volume."""
    path = Path(path)
    root = Path(volume_root or os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume"))
    default_delivery = "volume" if root.is_dir() else "inline"
    delivery = str(values.get("output_delivery", default_delivery)).lower()
    if delivery not in {"volume", "inline"}:
        raise ValueError("output_delivery must be volume or inline")

    size = path.stat().st_size
    content_type = "video/webm" if path.suffix.lower() == ".webm" else "video/mp4"
    result = {
        "filename": path.name,
        "content_type": content_type,
        "size_bytes": size,
        "sha256": _sha256(path),
        "delivery": delivery,
        "ai_generated": True,
        "ai_content_id": ai_content_id,
        "disclosure": "AI GENERATED | MINIMAX H3",
    }

    if delivery == "inline":
        if size > MAX_INLINE_BYTES:
            raise ValueError(
                f"video is {size} bytes; inline delivery is limited to {MAX_INLINE_BYTES} bytes. "
                "Attach a network volume and use output_delivery=volume."
            )
        result["data"] = base64.b64encode(path.read_bytes()).decode("ascii")
        return result

    if not root.is_dir():
        raise RuntimeError(f"RunPod network volume is not mounted at {root}")
    destination_dir = root / "outputs" / _safe_job_id(job_id)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / path.name
    shutil.copy2(path, destination)
    result["volume_path"] = destination.as_posix()
    sidecar = destination.with_suffix(destination.suffix + ".json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ai_generated": True,
                "model": "MiniMax H3",
                "ai_content_id": ai_content_id,
                "filename": destination.name,
                "content_type": content_type,
                "size_bytes": size,
                "sha256": result["sha256"],
                "disclosure": "AI GENERATED | MINIMAX H3",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    result["manifest_path"] = sidecar.as_posix()
    return result
