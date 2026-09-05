"""Persistent ComfyUI process and one-request MiniMax H3 execution runtime."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

from h3_media import encode_video
from h3_prompt import format_h3_prompt
from h3_tuning import CacheTuning
from h3_workflow import aligned_frames, build_workflow, dimensions, validate_inputs
from weights import COMFY_ROOT, ensure_weights

COMFY_URL = "http://127.0.0.1:8188"


@dataclass(frozen=True)
class GenerationResult:
    path: Path
    metrics: dict


def _json_request(path: str, payload: dict | None = None, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        COMFY_URL + path,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class H3Runtime:
    def __init__(self) -> None:
        ensure_weights()
        self.process = self._start_comfy()

    def _start_comfy(self) -> subprocess.Popen:
        command = [
            sys.executable,
            str(COMFY_ROOT / "main.py"),
            "--listen", "127.0.0.1",
            "--port", "8188",
            "--disable-auto-launch",
            "--disable-metadata",
            "--lowvram",
            "--reserve-vram", os.getenv("H3_RESERVE_VRAM_GB", "1.5"),
        ]
        attention = "pytorch"
        try:
            __import__("sageattention")
            command.append("--use-sage-attention")
            attention = "sageattention"
        except ImportError:
            pass
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        process = subprocess.Popen(command, cwd=COMFY_ROOT, env=env)
        deadline = time.time() + 180
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"ComfyUI exited during startup with code {process.returncode}")
            try:
                _json_request("/system_stats", timeout=3)
                print(f"MiniMax H3 ready: attention={attention} comfy_pid={process.pid}", flush=True)
                return process
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                time.sleep(1)
        process.terminate()
        raise RuntimeError("ComfyUI did not become ready within 180 seconds")

    @staticmethod
    def _stage_image(path: Path | None, label: str) -> str | None:
        if path is None:
            return None
        source = Path(path)
        suffix = source.suffix.lower() if source.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} else ".png"
        name = f"h3-{label}-{uuid.uuid4().hex}{suffix}"
        destination = COMFY_ROOT / "input" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return name

    @staticmethod
    def _history_output(entry: dict) -> Path:
        outputs = entry.get("outputs") or {}
        for node in outputs.values():
            for key in ("videos", "video", "gifs", "images"):
                for item in node.get(key, []) if isinstance(node, dict) else []:
                    if isinstance(item, dict) and item.get("filename"):
                        folder = item.get("type", "output")
                        base = COMFY_ROOT / ("output" if folder == "output" else folder)
                        path = base / item.get("subfolder", "") / item["filename"]
                        if path.exists():
                            return path
        raise RuntimeError("ComfyUI completed without a saved video")

    def generate(
        self,
        *,
        prompt: str,
        first_frame: Path | None = None,
        last_frame: Path | None = None,
        aspect_ratio: str = "16:9",
        size: str = "balanced",
        duration: float = 5.0,
        steps: int = 20,
        seed: int | None = None,
        structured_prompt: bool = True,
        loop: bool = False,
        include_audio: bool = True,
        output_codec: str = "webm-av1",
        encode_quality: int = 26,
        cache: CacheTuning | None = None,
        return_metrics: bool = False,
    ) -> Path | GenerationResult:
        total_started = time.monotonic()
        ai_content_id = f"h3-{uuid.uuid4().hex}"
        validate_inputs(first_frame=first_frame, last_frame=last_frame, loop=loop, steps=steps, seed=seed)
        frames = aligned_frames(duration)
        width, height = dimensions(aspect_ratio, size)
        actual_seconds = frames / 24
        if seed is None:
            seed = random.SystemRandom().randint(0, 2**63 - 1)
        if loop:
            last_frame = first_frame
        formatted = format_h3_prompt(
            prompt,
            actual_seconds,
            first_frame=first_frame is not None,
            last_frame=last_frame is not None,
            structured=structured_prompt,
        )
        first_name = self._stage_image(first_frame, "first")
        last_name = self._stage_image(last_frame, "last")
        graph = build_workflow(
            prompt=formatted,
            width=width,
            height=height,
            frames=frames,
            steps=steps,
            seed=seed,
            first_image_name=first_name,
            last_image_name=last_name,
            cache=cache,
        )
        try:
            sample_started = time.monotonic()
            queued = _json_request("/prompt", {"prompt": graph})
            prompt_id = queued.get("prompt_id")
            if not prompt_id:
                raise RuntimeError(f"ComfyUI rejected workflow: {queued}")
            deadline = time.time() + 45 * 60
            while time.time() < deadline:
                time.sleep(1)
                history = _json_request("/history/" + urllib.parse.quote(prompt_id), timeout=30)
                entry = history.get(prompt_id)
                if not entry:
                    continue
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise RuntimeError("ComfyUI generation failed: " + json.dumps(status.get("messages", []))[-2000:])
                if status.get("completed"):
                    raw = self._history_output(entry)
                    sample_seconds = time.monotonic() - sample_started
                    encode_started = time.monotonic()
                    output = encode_video(
                        raw,
                        output_codec,
                        encode_quality,
                        include_audio,
                        ai_content_id=ai_content_id,
                    )
                    encode_seconds = time.monotonic() - encode_started
                    total_seconds = time.monotonic() - total_started
                    print(
                        f"generated {width}x{height} frames={frames} seconds={actual_seconds:.2f} "
                        f"steps={steps} seed={seed} loop={loop} total_seconds={total_seconds:.3f} "
                        f"cache={cache.profile if cache is not None else 'off'}",
                        flush=True,
                    )
                    if not return_metrics:
                        return output
                    metrics = {
                        "schema_version": 1,
                        "ai_generated": True,
                        "ai_content_id": ai_content_id,
                        "ai_disclosure": "AI GENERATED | MINIMAX H3",
                        "total_seconds": round(total_seconds, 3),
                        "generation_seconds": round(sample_seconds, 3),
                        "encode_seconds": round(encode_seconds, 3),
                        "output_bytes": output.stat().st_size,
                        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                        "width": width,
                        "height": height,
                        "frames": frames,
                        "duration_seconds": round(actual_seconds, 3),
                        "steps": steps,
                        "seed": seed,
                        "cache": cache.public_dict() if cache is not None else {"profile": "off"},
                    }
                    return GenerationResult(output, metrics)
            raise RuntimeError("generation exceeded 45 minute safety timeout")
        finally:
            for name in (first_name, last_name):
                if name:
                    (COMFY_ROOT / "input" / name).unlink(missing_ok=True)
