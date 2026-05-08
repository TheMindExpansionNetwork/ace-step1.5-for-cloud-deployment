# Jimsky ACE-Step XL Turbo Deployment

This fork is wired as the inference/deployment companion for the Jimsky/Hermes ACE-Step LoRA factory.

## Target model

Default high-quality model target:

```text
ACE-Step/acestep-v15-xl-turbo
local checkpoint directory: checkpoints/acestep-v15-xl-turbo
```

Why this model:

- ACE-Step 1.5 XL Turbo, 4B DiT.
- 8-step turbo generation.
- Better quality than the smaller 2B turbo lane.
- Compatible with ACE-Step LM models: 0.6B, 1.7B, and 4B.

## VRAM planning

From the Hugging Face model card:

- 12 GB: possible with CPU offload + INT8 quantization.
- 16 GB: possible with CPU offload.
- 20 GB+: recommended without offload.
- 24 GB+: best lane for XL + 4B LM.

For cheap continuous background generation, use smaller/older lanes or a T4/L4 with offload. For hero output and LoRA acceptance tests, use L4/A10G/A100 class GPUs depending on target duration and concurrency.

## Safe preflight

Run this first. It does not download weights or start a GPU:

```bash
python scripts/jimsky_xl_turbo_preflight.py
```

Expected shape:

```json
{
  "ok": true,
  "default_model": "acestep-v15-xl-turbo",
  "default_repo": "ACE-Step/acestep-v15-xl-turbo",
  "gpu_started": false,
  "model_download_started": false,
  "xl_turbo_registered": true
}
```

## Launch server

The persistent API server now defaults to XL Turbo unless overridden:

```bash
export ACESTEP_CONFIG_PATH=acestep-v15-xl-turbo
export ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B
./run_server.sh
```

The first real launch may download large model weights. Do that only when the operator approves the exact GPU/backend and storage target.

## Integration with LoRA factory

Trainer fork:

```text
https://github.com/TheMindExpansionNetwork/ace-step1.5_Trainer
```

Pipeline flow:

```text
trainer final adapter
  -> private HF adapter repo
  -> this deployment server loads base XL Turbo
  -> LoRA adapter smoke test
  -> audio QA / review card
```

## HyperFrames / show packaging

For launch/demo packaging, use HyperFrames `v0.5.4` as the pinned video composition reference. This keeps generated tracks, model cards, and review snippets easy to turn into narrated demo clips.
