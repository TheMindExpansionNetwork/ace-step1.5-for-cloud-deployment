# HyperFrames v0.5.4 Demo Packaging Plan

Pinned release:

```text
heygen-com/hyperframes v0.5.4
commit: 8de7ad7f61df55f3cc62775b43bc315f8ea143dd
release: https://github.com/heygen-com/hyperframes/releases/tag/v0.5.4
```

## Why v0.5.4 matters for this music factory

Useful release features for Jimsky/Hermes launches:

- `--composition` flag: render a specific composition from a multi-composition project.
- `doctor --json`: machine-readable health checks for agent preflight.
- `png-sequence` output: useful for contact sheets, visualizers, and frame-level QA.
- `--resolution` on `init` and `render`: one-line 4K scaffolding/rendering.
- `browserGpuMode: "auto"`: probe-once WebGL detection with software fallback.
- single-clock transport: reduces pause/play audio drift in rendered demo pieces.

## Role in the ACE-Step pipeline

HyperFrames is not the music model trainer. It is the packaging/render layer:

```text
ACE-Step LoRA training result
  -> generated audio samples
  -> model card/provenance text
  -> album art / ComfyUI visuals / CDM visuals
  -> HyperFrames composition
  -> narrated launch/demo/review video
```

## Planned first composition

`ace-lora-launch-card`:

- title: adapter name + base model `ACE-Step/acestep-v15-xl-turbo`
- visual lane: static cover art or ComfyUI queued variations
- audio lane: generated sample preview
- proof lane: dataset provenance, training settings, HF repo privacy, QA status
- CTA: private review first; public release only after approval

## Suggested commands

```bash
# machine-readable readiness
hyperframes doctor --json

# initialize 4K-capable project if needed
hyperframes init ace-lora-launch-card --resolution 4k

# render one composition
hyperframes render --composition ace-lora-launch-card --resolution 1080p

# frame sequence for visual QA/contact sheets
hyperframes render --composition ace-lora-launch-card --format png-sequence
```

## ComfyUI queue integration

When the operator provides a ComfyUI endpoint/API key, use ComfyUI for:

- many cover-art variations;
- pose/edit batches from one source image;
- visual background shuffle packs;
- inpaint/img2img edits for album/artist/persona assets.

Then feed the selected images or frame sequences into HyperFrames for final show packaging.
