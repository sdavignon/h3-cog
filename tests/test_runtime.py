import hashlib

import h3_runtime
from h3_runtime import GenerationResult, H3Runtime


def test_generation_metrics_cover_output_identity_and_lossless_default(tmp_path, monkeypatch):
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"raw")
    encoded = tmp_path / "result.webm"
    encoded.write_bytes(b"encoded-video")
    runtime = object.__new__(H3Runtime)
    monkeypatch.setattr(runtime, "_stage_image", lambda path, label: None)
    monkeypatch.setattr(runtime, "_history_output", lambda entry: raw)
    monkeypatch.setattr(h3_runtime.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(h3_runtime, "encode_video", lambda *args, **kwargs: encoded)

    def fake_json_request(path, payload=None, timeout=60):
        if path == "/prompt":
            return {"prompt_id": "job-1"}
        return {"job-1": {"status": {"completed": True}, "outputs": {}}}

    monkeypatch.setattr(h3_runtime, "_json_request", fake_json_request)
    result = runtime.generate(prompt="test", duration=4, seed=42, return_metrics=True)
    assert isinstance(result, GenerationResult)
    assert result.path == encoded
    assert result.metrics["seed"] == 42
    assert result.metrics["cache"] == {"profile": "off"}
    assert result.metrics["output_bytes"] == len(b"encoded-video")
    assert result.metrics["output_sha256"] == hashlib.sha256(b"encoded-video").hexdigest()
    assert result.metrics["generation_seconds"] >= 0
    assert result.metrics["encode_seconds"] >= 0
