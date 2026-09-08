# MiniMax H3 Cog for app.nz

MiniMax H3 text-to-video, image-to-video, first/last-frame video, and
keyframe-conditioned seamless loops in one portable [Cog](https://github.com/replicate/cog).
It uses MiniMax's open-weight H3 Base FL2VA model through ComfyUI's native H3
nodes, keeps the model warm between predictions, pulls four verified files from
the app.nz R2 mirror, and encodes web-ready AV1 on the GPU.

[![Deploy H3 on app.nz](https://app.nz/deploy-button.svg)](https://app.nz/deploy?template=minimax-h3)

## Important model-license boundary

This repository's adapter code is MIT licensed. **MiniMax H3 weights are not
MIT licensed.** They use the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE),
which currently defines an applicable territory that excludes the United
States, European Union, United Kingdom, Republic of Korea, and other uses in its
acceptable-use policy. Review the current upstream terms with your counsel.

The runtime refuses to download weights until the deployer explicitly sets:

```sh
MINIMAX_H3_LICENSE_ACCEPTED=1
```

Setting the variable is your acknowledgement, not app.nz's. Do not deploy the
model to an excluded territory or expose a hosted service where the license
does not permit it.

## What is included

- T2VA: prompt to 24fps video with native synchronized 32kHz stereo audio.
- I2VA/L2VA: optional first frame or optional last frame.
- FL2VA: optional first and last frames, with MiniMax's documented alignment
  prompt injected for ordinary prompts.
- Loop: reuses the supplied first frame as the last-frame condition. This is a
  generated content loop, not a cheap duplicated/reversed post-process.
- Native H3 aspect ratios and a 768px short-edge quality tier.
- Official `res_multistep` sampler and 20-step quality default; 12–16 steps and
  reduced pixel area are exposed for previews.
- optional SageAttention when supplied by the image operator, with PyTorch
  attention fallback and ComfyUI low-VRAM support
  offloading, and persistent model processes for 32GB RTX 5090 workers.
- GPU `av1_nvenc` WebM output with SVT-AV1 fallback; GPU H.264 with x264
  fallback. Native audio is remuxed as Opus or AAC.
- Standard Cog HTTP plus a RunPod Serverless handler using the same runtime.

H3's current open release is dense full-attention. MiniMax says sparse
attention will follow. The public path does not apply a lossy cache. Operator
sweeps can use ComfyUI's built-in EasyCache behind a short-lived signed
envelope, described below. We do not apply CG-Taylor or a "latent teleport"
cache: neither has been validated as an H3-compatible node against its joint
audio/video latent stream and first/last-frame fidelity. SageAttention remains
an opt-in acceleration path because its current Blackwell build is not
distributed on the configured Python index; the adapter leaves room for the
upstream sparse-attention release.

## Inputs

| Input | Default | Notes |
| --- | --- | --- |
| `prompt` | required | Scene, motion, camera, dialogue, SFX, and music |
| `first_frame` / `last_frame` | empty | Zero, one, or two H3 keyframes |
| `aspect_ratio` | `16:9` | `16:9`, `9:16`, `1:1`, `4:3`, `3:4`, `21:9` |
| `size` | `balanced` | `preview`, `balanced`, `native` |
| `duration` | `5` | 4–15 seconds; snaps upward to H3's `17k+5` frame grid |
| `steps` | `20` | 20 official; 12–16 preview; allowed 8–30 |
| `structured_prompt` | `true` | Applies the public MiniMax audiovisual prompt shape |
| `loop` | `false` | Requires `first_frame`; conditions the final frame on it |
| `include_audio` | `true` | Keep or strip H3's generated stereo audio |
| `output_codec` | `webm-av1` | `webm-av1` or `mp4-h264` |
| `encode_quality` | `26` | Lower is higher quality and larger |

The native 16:9 canvas is 1344×768. `balanced` reduces pixel area to 78%, and
`preview` to 58%, while preserving 32-pixel alignment. A requested 5 seconds is
124 frames, or 5.17 seconds, because H3 only accepts the `17k+5` grid.

## Local Cog usage

```sh
export MINIMAX_H3_LICENSE_ACCEPTED=1
cog run \
  -i prompt="A cinematic macro shot of a glass hummingbird unfolding its wings; synchronized crystalline chimes." \
  -i size=preview -i duration=4 -i steps=12
```

Image-to-video:

```sh
cog run -i first_frame=@first.png \
  -i prompt="A slow arc shot as the subject turns toward the sunrise." \
  -i size=balanced -i duration=5
```

First/last-frame loop:

```sh
cog run -i first_frame=@anchor.png -i loop=true \
  -i prompt="One continuous shot: steam curls around the cup and settles exactly into the opening composition."
```

The first build installs CUDA 12.8 / PyTorch 2.11 and pins ComfyUI commit
`9a9fdb10ed144ce760d9682cb247526ea23cc525`, the native H3 implementation tested
by this adapter. Model weights are runtime data and are not baked into the
container image.

## R2 model mirror

Only the 5090 production set is mirrored—about 42.5GB rather than every H3
variant:

```text
diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
vae/minimax_h3_video_vae_fp16.safetensors
vae/minimax_h3_audio_vae_fp32.safetensors
```

`weights.py` downloads `https://appstatic.app.nz/models/Comfy-Org/MiniMax-H3/manifest.json`,
resumes partial files with HTTP Range, verifies size and SHA-256, and links the
cache into ComfyUI. `/runpod-volume/models` is preferred when present, otherwise
`/weights`. A bad checksum never becomes a live model.

To populate the mirror after accepting the upstream license:

```sh
uv run --extra mirror python mirror_weights.py --dry-run
uv run --extra mirror python mirror_weights.py --bucket appstatic
```

Required R2 variables are `R2_ENDPOINT`, `AWS_ACCESS_KEY_ID`, and
`AWS_SECRET_ACCESS_KEY`. The upload publishes immutable weight objects and a
short-cache manifest containing exact sizes and SHA-256 hashes.

## RunPod Serverless

Build and push the same Cog image, then override the container command:

```text
python -u /src/rp_handler.py
```

The handler accepts the Cog-compatible `first_frame` and `last_frame` inputs as
public HTTPS URLs or base64 PNG/JPEG/WebP data URLs, plus the explicit
`first_frame_url` and `last_frame_url` aliases. It rejects private/reserved
network targets and caps each image at 32MiB. Data URLs make a single portable
RunPod request self-contained without adding public asset hosting. Every
request must include a pseudonymous `user_id` and affirmative copyright,
likeness, terms, and no-training attestations. The worker sends the exact prompt
and moderation-size source frames to the configured HTTPS moderation service
under HMAC authentication. Generation fails closed if the service, secret,
attestations, or authenticated allow decision is missing.

Every delivered video has a visible `AI GENERATED | MINIMAX H3` overlay, an
embedded machine-readable model/content identifier, and a JSON sidecar tied to
the output SHA-256 when volume delivery is used.

When `/runpod-volume` is mounted, output defaults to
`/runpod-volume/outputs/<job-id>/video.*` and the response contains its path,
size, and SHA-256 rather than an oversized base64 payload. Explicit
`output_delivery=inline` remains available for videos no larger than 7MiB.
RunPod network-volume S3 access does not support presigned URLs, so a separate
authenticated retrieval layer is required before this path is exposed to a
browser client. Set minimum workers to zero. A persistent network volume also
avoids re-downloading the 42.5GB model on cold workers.

### Authenticated acceleration sweeps

EasyCache is experimental and may trade temporal, audio, or keyframe fidelity
for speed. It is deliberately absent from Cog inputs and disabled by default.
Only the RunPod operator can activate it by signing a bounded, expiring `_tuning`
object with `H3_TUNING_SECRET`. The runtime never logs or returns that secret or
the request signature.

Create a fixed-seed baseline plus conservative, balanced, and aggressive
candidates:

```sh
export H3_TUNING_SECRET="$(openssl rand -hex 32)"
python h3_sweep.py request.json --sweep-id gallery-a01 > sweep.json
```

Set the same secret only on the private worker. Each candidate expires within
one hour, and the server rejects unknown fields, invalid signatures, cache
thresholds above `0.30`, and invalid cache windows. Custom settings can be
signed with `sign_tuning` from `h3_tuning.py`; the three named profiles are the
recommended first pass. Never expose the secret in the app.nz client or public
Cog schema.

Each RunPod response includes schema-versioned metrics for total generation and
encode time, dimensions, frame count, exact seed, output size and SHA-256, and
the non-secret cache configuration. Compare every cache candidate with its
same-seed `off` baseline. Promote a profile only after visual review of motion,
audio synchronization, first/last-frame alignment, and loop seam; no H3
EasyCache speedup or quality claim is made before those GPU A/B results exist.

## 5090 tuning and cost

The pruned INT8 transformer is 20.97GB, NVFP4 Qwen encoder 15.69GB, visual VAE
5.21GB, and audio VAE 0.61GB. They cannot all remain resident on a 32GB card;
ComfyUI intentionally offloads between prompt encoding, denoising, and VAE
decode. A worker therefore needs substantial system RAM and a persistent weight
volume. Do not advertise a 24GB tier for this build.

Use preview/balanced for most requests, keep the process warm for bursts, and
scale the worker to zero when idle. app.nz's H3 template publishes the final
customer rate and uses an 80% platform markup over the selected RunPod
serverless compute price; billing remains per execution second.

## Tests

```sh
uv run --extra dev pytest
python -m py_compile h3_*.py predict.py rp_handler.py weights.py mirror_weights.py
cog build -t h3-cog:local
```

Unit tests cover the H3 frame grid, native dimensions, official keyframe prompt
prefixes, workflow graph, GPU/CPU encoder fallback, explicit license gate, and
resumable SHA-verified downloads. A real GPU smoke test additionally requires
the accepted model license, 43GB of weights, a 32GB CUDA GPU, and enough host RAM.

`demo_prompts.json` is the deterministic seven-clip launch suite covering text,
image, first/last-frame, loop, vertical, square, and ultrawide output. Keep seed,
prompt, dimensions, steps, and source keyframes attached to every gallery
record. The suite remains prompt-only until the deployer accepts the weight
license and supplies a compliant worker region; no synthetic or untested clip
is presented as H3 output.

The five required source keyframes are included in `gallery-keyframes/` as
`<demo-id>-first.png` and `<demo-id>-last.png`. After the license/region gate is
cleared, render and prepare app.nz sidecars with:

```sh
python h3_gallery.py --output ./h3-gallery-renders
cd ../app-site
bun scripts/ingest-videos.mjs --in ../h3-cog/h3-gallery-renders \
  --out public/assets/videos --base-url /assets/videos --source minimax-h3
```

The second command runs from the app-site repository. It encodes delivery WebM
and poster files and upserts stable `minimax-h3:<demo-id>` gallery records. Use
`--limit 1` for the first licensed smoke render.

## Upstream references

- [MiniMax H3 model card and license](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [ComfyUI H3 workflow documentation](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
- [ComfyUI-packaged checkpoints](https://huggingface.co/Comfy-Org/MiniMax-H3)
- [MiniMax base-mode prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
