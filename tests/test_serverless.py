from pathlib import Path

import pytest

from h3_serverless import MAX_INLINE_BYTES, deliver_video, frame_url


def test_frame_url_keeps_cog_and_runpod_inputs_in_parity():
    assert frame_url({"first_frame": "https://cdn.example/cog.png"}, "first_frame") == "https://cdn.example/cog.png"
    assert frame_url({"first_frame_url": "https://cdn.example/runpod.png"}, "first_frame") == "https://cdn.example/runpod.png"
    assert frame_url(
        {
            "last_frame": "https://cdn.example/cog-last.png",
            "last_frame_url": "https://cdn.example/legacy-last.png",
        },
        "last_frame",
    ) == "https://cdn.example/cog-last.png"
    assert frame_url({}, "first_frame") is None


def test_deliver_video_defaults_to_volume_when_mounted(tmp_path: Path):
    video = tmp_path / "video.webm"
    video.write_bytes(b"video-data")
    volume = tmp_path / "volume"
    volume.mkdir()

    result = deliver_video(
        video,
        {},
        job_id="job/123",
        volume_root=volume,
        ai_content_id="h3-test",
    )

    stored = volume / "outputs" / "job-123" / "video.webm"
    assert stored.read_bytes() == b"video-data"
    assert result["delivery"] == "volume"
    assert result["volume_path"] == stored.as_posix()
    assert result["ai_content_id"] == "h3-test"
    assert (stored.with_suffix(".webm.json")).exists()
    assert "data" not in result


def test_deliver_video_allows_small_explicit_inline_result(tmp_path: Path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"small")

    result = deliver_video(
        video,
        {"output_delivery": "inline"},
        volume_root=tmp_path / "missing-volume",
    )

    assert result["delivery"] == "inline"
    assert result["data"] == "c21hbGw="
    assert result["content_type"] == "video/mp4"


def test_deliver_video_rejects_oversized_inline_result(tmp_path: Path):
    video = tmp_path / "video.mp4"
    with video.open("wb") as handle:
        handle.truncate(MAX_INLINE_BYTES + 1)

    with pytest.raises(ValueError, match="inline delivery is limited"):
        deliver_video(video, {"output_delivery": "inline"}, volume_root=tmp_path / "missing")
