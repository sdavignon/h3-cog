from h3_workflow import build_workflow
from h3_tuning import CacheTuning


def test_text_workflow_has_official_h3_sampling_path():
    graph = build_workflow(prompt="p", width=1024, height=576, frames=124, steps=20, seed=7)
    assert graph["1"]["inputs"]["unet_name"].endswith("pruned_int8_convrot.safetensors")
    assert graph["2"]["inputs"]["type"] == "minimax"
    assert graph["5"]["class_type"] == "MiniMaxH3ImageToVideo"
    assert graph["8"]["inputs"]["sampler_name"] == "res_multistep"
    assert graph["9"]["inputs"] == {"model": ["1", 0], "scheduler": "simple", "steps": 20, "denoise": 1.0}
    assert graph["13"]["inputs"]["audio"] == ["12", 0]
    assert graph["14"]["inputs"]["codec"] == "auto"


def test_first_and_last_frames_are_optional_load_nodes():
    graph = build_workflow(
        prompt="p", width=768, height=768, frames=124, steps=16, seed=9,
        first_image_name="first.png", last_image_name="last.png",
    )
    assert graph["15"] == {"class_type": "LoadImage", "inputs": {"image": "first.png"}}
    assert graph["16"] == {"class_type": "LoadImage", "inputs": {"image": "last.png"}}
    assert graph["5"]["inputs"]["first_frame"] == ["15", 0]
    assert graph["5"]["inputs"]["last_frame"] == ["16", 0]


def test_easycache_is_only_inserted_when_operator_tuning_is_supplied():
    cache = CacheTuning("balanced", 0.12, 0.15, 0.9, False, "sweep-1", "balanced")
    graph = build_workflow(
        prompt="p", width=768, height=768, frames=124, steps=20, seed=9,
        first_image_name="first.png", cache=cache,
    )
    assert graph["15"]["class_type"] == "EasyCache"
    assert graph["15"]["inputs"]["model"] == ["1", 0]
    assert graph["6"]["inputs"]["model"] == ["15", 0]
    assert graph["9"]["inputs"]["model"] == ["15", 0]
    assert graph["16"] == {"class_type": "LoadImage", "inputs": {"image": "first.png"}}
