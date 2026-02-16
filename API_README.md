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

### 3. Generate Music (Simple)
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"caption": "A cinematic epic orchestral track", "duration": 15}'
```

---

## 📚 API Reference

### 1. `POST /generate`

The main endpoint for generating music. It accepts a JSON payload mapping to `GenerationParams`.

#### **Common Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `caption` | string | **Required** | Main text prompt. |
| `task_type` | string | `text2music` | `text2music`, `cover`, `repaint`, `extract`, `lego`, `complete` |
| `duration` | float | `-1` | Duration in seconds (<0 for auto). |
| `thinking` | bool | `true` | Enable LLM reasoning (Chain-of-Thought). |
| `instrumental` | bool | `false` | Force instrumental generation. |

#### **Feature Examples**

**A. Style Transfer / Cover**
Use a reference audio file to guide the generation style.
```json
{
  "task_type": "cover",
  "caption": "Remix this in a jazz style",
  "reference_audio": "/abs/path/to/reference.mp3",
  "audio_cover_strength": 0.6
}
```

**B. In-Painting (Repaint)**
Regenerate a specific section of audio.
```json
{
  "task_type": "repaint",
  "caption": "Add drum fill",
  "src_audio": "/abs/path/to/source.mp3",
  "repainting_start": 10.0,
  "repainting_end": 15.0
}
```

**C. Advanced Generation**
Full control over diffusion and LLM parameters.
```json
{
  "caption": "Experimental electronic",
  "inference_steps": 30,
  "guidance_scale": 5.5,
  "use_adg": true,
  "latent_rescale": 0.9,
  "lm_temperature": 0.9,
  "cot_caption": "Override the LLM's caption thinking"
}
```

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

---

## 🐍 Python Client Example

```python
import requests

def generate_song():
    url = "http://localhost:8000/generate"
    payload = {
        "caption": "A synthwave track",
        "duration": 10,
        "thinking": True
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        print("Audio saved at:", result['audios'][0]['path'])
        print("Timings:", result['timings'])
    else:
        print("Error:", response.text)

if __name__ == "__main__":
    generate_song()
```
