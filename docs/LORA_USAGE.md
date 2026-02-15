# LoRA Usage Guide

This guide explains how to use LoRA (Low-Rank Adaptation) adapters to customize music generation in ACE-Step.

## What is LoRA?

LoRA allows you to customize the DiT model's behavior by loading pre-trained adapters without modifying the base model weights. This is useful for:

- **Style customization** (e.g., "rock style", "jazz style")
- **Artist emulation** (e.g., specific artist's sound)
- **Genre specialization** (e.g., focused on electronic music)
- **Personalization** (e.g., your own fine-tuned adapters)

## Prerequisites

- ACE-Step installed with dependencies
- Pre-trained LoRA adapter files (`.safetensors` or `.pt` format)
- LoRA adapters should be compatible with the ACE-Step DiT model

## LoRA Adapter Structure

Place your LoRA adapters in the `checkpoints/lora/` directory:

```
checkpoints/
├── lora/
│   ├── my_rock_style/
│   │   ├── adapter_model.safetensors  # LoRA weights
│   │   └── adapter_config.json        # LoRA configuration
│   ├── jazz_adapter/
│   │   └── adapter_model.safetensors
│   └── electronic_music/
│       └── adapter_model.safetensors
```

## Using LoRA with Python API

### Basic Example

```python
from acestep.handler import AceStepHandler

# Initialize handler
handler = AceStepHandler()
handler.initialize_service(
    checkpoint_dir="checkpoints",
    dit_model_path="acestep-v15-turbo",
    lm_model_path="acestep-5Hz-lm-1.7B",
)

# Load LoRA adapter
handler.load_lora(
    lora_path="checkpoints/lora/my_rock_style",
    scale=1.0  # 1.0 = full strength, 0.5 = half strength
)

# Generate music with LoRA applied
from acestep.inference import generate_music, GenerationParams

params = GenerationParams(
    caption="energetic rock guitar riff",
    lyrics="[Instrumental]",
    duration=30,
)

result = generate_music(handler, params)
print(f"Generated: {result.audio_paths[0]}")
```

### Adjust LoRA Strength

```python
# Set LoRA scale (0.0 to 1.0)
handler.set_lora_scale(0.7)  # 70% of LoRA effect

# Generate with adjusted strength
result = generate_music(handler, params)
```

### Switch Between LoRA Adapters

```python
# Load first adapter
handler.load_lora("checkpoints/lora/rock_style", scale=1.0)
result1 = generate_music(handler, params)

# Switch to different adapter
handler.unload_lora()  # Unload current
handler.load_lora("checkpoints/lora/jazz_style", scale=0.8)
result2 = generate_music(handler, params)
```

### Disable LoRA

```python
# Temporarily disable without unloading
handler.set_lora_scale(0.0)  # 0 = disabled

# Or unload completely
handler.unload_lora()
```

## Advanced Usage

### Multiple Adapters (if supported)

Some LoRA systems support multiple adapters simultaneously:

```python
# Load and activate specific adapter
handler.load_lora("checkpoints/lora/style1")
handler.set_active_lora_adapter("style1")

# Note: Check handler capabilities for multi-adapter support
```

### LoRA with Configuration Files

Save LoRA settings in your config:

```toml
# config.toml
[lora]
enabled = true
path = "checkpoints/lora/my_style"
scale = 0.8
```

Then use with CLI:

```bash
uv run python cli.py -c config.toml
```

## Finding LoRA Adapters

### Where to Get LoRA Adapters

1. **Train Your Own** - Use the original ACE-Step 1.5 repo with training code
2. **Community Sharing** - Check HuggingFace Model Hub for shared adapters
3. **Custom Fine-tuning** - Fine-tune on your specific music dataset

### Creating Your Own LoRA

For training LoRA adapters, use the [original ACE-Step 1.5 repository](https://github.com/ace-step/ACE-Step-1.5) which includes:
- Training scripts
- Dataset handling
- LoRA fine-tuning pipeline

This inference-optimized fork focuses on **using** pre-trained LoRA, not training them.

## Troubleshooting

### LoRA Not Loading

```python
# Check if LoRA is loaded
if handler.lora_loaded:
    print("LoRA is active")
else:
    print("No LoRA loaded")
```

### Incompatible LoRA

If you get errors:
- Ensure LoRA was trained for ACE-Step DiT model
- Check LoRA rank matches (typically r=8 or r=16)
- Verify adapter files are not corrupted

### Performance Issues

- Lower LoRA scale if results are too different from base model
- Try different adapters to find the best match for your use case

## Best Practices

1. **Start with low scale** (0.5-0.7) and adjust based on results
2. **Test without LoRA first** to establish a baseline
3. **Use descriptive names** for your LoRA adapter folders
4. **Keep adapters organized** by style/purpose
5. **Document your adapters** (what they do, optimal scale, etc.)

## Example Workflow

```python
from acestep.handler import AceStepHandler
from acestep.inference import generate_music, GenerationParams

# 1. Initialize
handler = AceStepHandler()
handler.initialize_service(
    checkpoint_dir="checkpoints",
    dit_model_path="acestep-v15-turbo",
    lm_model_path="acestep-5Hz-lm-1.7B",
)

# 2. Load LoRA
handler.load_lora("checkpoints/lora/electronic_music", scale=0.8)

# 3. Generate multiple variations
prompts = [
    "upbeat techno with heavy bass",
    "ambient electronic soundscape",
    "synth-driven dance track"
]

for prompt in prompts:
    params = GenerationParams(caption=prompt, duration=30)
    result = generate_music(handler, params)
    print(f"Generated: {result.audio_paths[0]}")

# 4. Cleanup
handler.unload_lora()
```

---

**Need help?** Check the main README.md or open an issue on GitHub.
