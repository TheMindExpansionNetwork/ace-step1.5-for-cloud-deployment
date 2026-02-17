# ACE-Step 1.5 API Server Guide

This guide details how to use the persistent API server for ACE-Step 1.5. The API supports **100% of the features** found in the CLI/Python interface, including advanced DiT parameters, CoT reasoning, and LoRA management.

## 🚀 Quick Start

### 1. Requirements
Ensure you have the server dependencies installed:
```bash
pip install "fastapi[all]" "uvicorn[standard]"
```

### 2. Start the Server
Run the server with the provided script. It runs on port 8000 by default.
```bash
chmod +x run_server.sh
./run_server.sh
```
*Wait for the message: `✨ Server Ready! Models are resident in VRAM.`*

### 3. Generate Music (Full Example)
You can use this full payload to test all features.

```python
import requests
import json
from IPython.display import Audio, display

# ➤ FULL PAYLOAD CONFIGURATION
payload = {
    # --- CORE INPUTS ---
    "caption": "female vocals, rap, modern, hip hop, Indian fusion, whispered, plucked synth melody intro",
    "lyrics": "[Intro]\n(Plucked Synth Melody)\n\n[Verse 1]\nThe sun melts over the endless plain...",
    "instrumental": False,          # Set True to ignore lyrics aiming for instrumental track
    
    # --- MUSICAL ATTRIBUTES (Optional overrides) ---
    "bpm": 95,                      # Force specific speed (e.g., 90-140 for hip hop)
    "keyscale": "C Minor",          # Force musical key (e.g., "Am", "F# Major")
    "timesignature": "4/4",         # "4/4", "3/4", "6/8" etc.
    "vocal_language": "en",         # "en", "zh", "ja", "ko", "fr", "de", "es", "it"
    "duration": 45.0,               # Duration in seconds (10.0 to 600.0)

    # --- ADVANCED GENERATION SETTINGS ---
    "inference_steps": 50,          # Quality vs Speed. 8=fast, 25=standard, 50+=high quality
    "guidance_scale": 7.5,          # How strictly to follow the text caption (5.0 - 9.0)
    "seed": 42,                     # specific seed for reproducibility (-1 = random)
    "batch_size": 1,                # How many variations to generate at once
    "audio_format": "flac",         # "flac", "wav", "mp3"

    # --- LLM / BRAIN SETTINGS (Chain-of-Thought) ---
    "thinking": True,               # Enable the "Brain" to plan the song structure
    "lm_temperature": 0.85,         # Creativity of the planning (0.5=focused, 1.2=chaotic)
    "lm_cfg_scale": 2.0,            # How strictly the brain follows instructions
    "use_cot_caption": True,        # Let AI refine your simple caption into a detailed one
    "use_cot_metas": True,          # Let AI decide missing BPM/Key if you didn't provide them
    "use_cot_language": True,       # Let AI detect language from lyrics
    "use_constrained_decoding": True, # Ensure metadata follows strict format

    # --- AUDIO POST-PROCESSING ---
    "enable_normalization": True,   # Maximize volume without clipping
    "normalization_db": -1.0,       # Target peak volume (-1.0 dB is standard)

    # --- ADVANCED DIT (Diffusion) CONTROLS ---
    "use_adg": False,               # Adaptive Dual Guidance (experimental, for base model)
    "shift": 1.0,                   # Timestep shift (controls noise schedule)
    "latent_shift": 0.0,            # Shift latents before decoding (rarely used)
    "latent_rescale": 1.0,          # Rescale latents before decoding
    
    # --- TASK SPECIFIC (For Editing/Remixing) ---
    "task_type": "text2music",      # Options: "text2music", "cover", "repaint", "extract", "lego", "complete"
    
    # If task_type="cover" (Style Transfer):
    # "reference_audio": "/path/to/original.mp3",
    # "audio_cover_strength": 0.6,  # 0.1 (strong change) to 0.9 (keep original melody)
    
    # If task_type="repaint" (Edit a section):
    # "src_audio": "/path/to/source.mp3",
    # "repainting_start": 10.0,     # Start editing at 10s
    # "repainting_end": 20.0,       # Stop editing at 20s

    # If task_type="extract" (Isolate Instrument):
    # "src_audio": "/path/to/source.mp3",
    # "caption": "drum kit, percussion only", # Describe what to KEEP
    # "audio_cover_strength": 0.85, # High strength to match original timing
}

# ➤ SEND REQUEST
print(f"🎵 Sending Request (Task: {payload['task_type']})...")

try:
    response = requests.post("http://localhost:8000/generate", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        
        # Print Timing Stats
        timings = data.get('timings', {})
        print(f"⏱️  Total Time: {timings.get('pipeline_total_time', 'N/A')}s")
        
        # Play Audios
        for idx, audio in enumerate(data['audios']):
            print(f"\n✅ Track {idx+1}: {audio['path']}")
            display(Audio(audio['path']))
            
    else:
        print(f"❌ Error {response.status_code}:\n{json.dumps(response.json(), indent=2)}")

except Exception as e:
    print(f"🚨 Connection Failed: {e}")
```

---

## 📚 API Reference

### 1. `POST /generate`

The main endpoint for generating music. It accepts a JSON payload mapping to `GenerationParams`.

#### **Schema Introspection**
To see the exact supported fields and their types dynamically:
```bash
curl http://localhost:8000/generation-schema
```

#### **Important Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `caption` | string | **Required** | Main text prompt. |
| `task_type` | string | `text2music` | `text2music`, `cover`, `repaint`, `extract`, `lego`, `complete` |
| `duration` | float | `-1` | Duration in seconds (<0 for auto). |
| `thinking` | bool | `true` | Enable LLM reasoning (Chain-of-Thought). |
| `instrumental` | bool | `false` | Force instrumental generation. |
| `keyscale` | string | `""` | e.g. "C Major", "Am" (Note: NOT `key_scale`) |
| `timesignature` | string | `""` | e.g. "4/4", "3/4" (Note: NOT `time_signature`) |

---

### 2. `GET /health`

Returns server status and VRAM usage. Useful for monitoring.
```json
{
  "status": "running",
  "vram_used_gb": 4.5,
  "vram_total_gb": 24.0,
  "dit_loaded": true,
  "llm_loaded": true,
  "device": "cuda"
}
```

---

### 3. `POST /v1/models/lora/load`

Load a LoRA adapter dynamically without restarting.
```json
{ "lora_path": "/path/to/adapter" }
```

### 4. `POST /v1/models/lora/unload`

Unload the current LoRA and restore the base model.
```json
{}
```

---

## ⚙️ Configuration

The server respects standard ACE-Step environment variables. You can set these in `run_server.sh` or your shell.

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| `ACESTEP_CONFIG_PATH` | Model config to load (e.g. `acestep-v15-turbo`) | `acestep-v15-turbo` |
| `ACESTEP_DEVICE` | Compute device (`cuda`, `mps`, `cpu`) | `auto` |
| `ACESTEP_COMPILE_MODEL`| Compile DiT for faster inference (slow startup) | `false` |
| `ACESTEP_OFFLOAD_TO_CPU`| Offload model parts to CPU to save VRAM | `false` |
| `ACESTEP_LM_MODEL_PATH` | Path/Name of the LLM model | `acestep-5Hz-lm-1.7B` |

---

## 🛡️ Model Verification & Persistence

- **Auto-Download**: If models are missing, the server will automatically download them to the `checkpoints/` directory on first launch.
- **Persistence**: Models are loaded **once** at startup. Subsequent requests reuse the same models in VRAM, ensuring instant response times (no loading overhead per request).
- **Safety**: The server includes VRAM guards. If a request is too large for your GPU, it will attempt to reduce batch sizes dynamically rather than crashing.
