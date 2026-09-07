"""Verified R2-first MiniMax H3 weight installation for ComfyUI."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "Comfy-Org/MiniMax-H3"
R2_BASE = os.getenv("APPNZ_MODELS_BASE", "https://appstatic.app.nz/models")
COMFY_ROOT = Path(os.getenv("COMFY_ROOT", "/opt/ComfyUI"))
FILES = {
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors": "diffusion_models",
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": "text_encoders",
    "vae/minimax_h3_video_vae_fp16.safetensors": "vae",
    "vae/minimax_h3_audio_vae_fp32.safetensors": "vae",
}


def weights_root() -> Path:
    configured = os.getenv("WEIGHTS_DIR")
    if configured:
        return Path(configured)
    if Path("/runpod-volume").is_dir():
        return Path("/runpod-volume/models")
    return Path("/weights")


def license_accepted() -> bool:
    return os.getenv("MINIMAX_H3_LICENSE_ACCEPTED", "").lower() in {"1", "true", "yes"}


def _manifest() -> dict[str, dict]:
    url = f"{R2_BASE.rstrip('/')}/{REPO}/manifest.json"
    request = urllib.request.Request(url, headers={"User-Agent": "appnz-h3-cog/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        entries = json.loads(response.read())
    if isinstance(entries, list):
        entries = {entry["path"]: entry for entry in entries}
    if not isinstance(entries, dict):
        raise ValueError("weight manifest must be an object or list")
    return entries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_size: int, expected_sha: str, retries: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(retries):
        try:
            have = partial.stat().st_size if partial.exists() else 0
            if have > expected_size:
                partial.unlink()
                have = 0
            headers = {"User-Agent": "appnz-h3-cog/0.1"}
            if have:
                headers["Range"] = f"bytes={have}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=180) as response:
                append = have > 0 and response.status == 206
                with partial.open("ab" if append else "wb") as handle:
                    for chunk in iter(lambda: response.read(8 << 20), b""):
                        handle.write(chunk)
            if partial.stat().st_size != expected_size:
                raise IOError(f"size mismatch for {destination.name}")
            if expected_sha and _sha256(partial) != expected_sha:
                raise IOError(f"sha256 mismatch for {destination.name}")
            partial.replace(destination)
            return
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(1.5 * (attempt + 1))


def _install_link(source: Path, folder: str) -> None:
    target = COMFY_ROOT / "models" / folder / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)


def ensure_weights(workers: int = 4) -> dict[str, Path]:
    if not license_accepted():
        raise RuntimeError(
            "Set MINIMAX_H3_LICENSE_ACCEPTED=1 only after reviewing and accepting "
            "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"
        )
    root = weights_root() / "MiniMax-H3"
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".download.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = _manifest()

        def fetch(relative: str) -> tuple[str, Path]:
            entry = manifest.get(relative)
            if not entry:
                raise RuntimeError(f"R2 manifest is missing {relative}")
            destination = root / relative
            expected_size = int(entry["size"])
            expected_sha = str(entry.get("sha256", ""))
            valid = destination.exists() and destination.stat().st_size == expected_size
            if valid and expected_sha:
                valid = _sha256(destination) == expected_sha
            if not valid:
                _download(
                    f"{R2_BASE.rstrip('/')}/{REPO}/{relative}",
                    destination,
                    expected_size,
                    expected_sha,
                )
            _install_link(destination, FILES[relative])
            return relative, destination

        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(FILES)))) as pool:
            installed = dict(pool.map(fetch, FILES))
    return installed
