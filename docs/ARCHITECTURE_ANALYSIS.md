# ACE-Step 1.5: Deep Architectural Analysis

> **Purpose:** Map exactly how models are loaded, managed, and VRAM is utilized across a full CLI generation invocation. Identify the minimal refactoring path to keep models resident in VRAM across multiple generations.

---

## Table of Contents

1. [High-Level Execution Flow](#1-high-level-execution-flow)
2. [Model Inventory & VRAM Budget](#2-model-inventory--vram-budget)
3. [Detailed Model Loading Lifecycle](#3-detailed-model-loading-lifecycle)
4. [VRAM Management Mechanisms](#4-vram-management-mechanisms)
5. [Why Models Are Unloaded Today](#5-why-models-are-unloaded-today)
6. [Singleton / Persistent Server Refactoring Plan](#6-singleton--persistent-server-refactoring-plan)
7. [Quick Reference: Key File Locations](#7-quick-reference-key-file-locations)

---

## 1. High-Level Execution Flow

```
cli.py main()
  │
  ├── GPU config detection (get_gpu_config / set_global_gpu_config)
  ├── Argument parsing (argparse + TOML config loading)
  ├── Interactive wizard (run_wizard) if no config provided
  │
  ├─── [1] AceStepHandler()           ← Empty shell, no models loaded yet
  ├─── [2] LLMHandler()               ← Empty shell, no models loaded yet
  │
  ├─── [3] dit_handler.initialize_service(...)
  │      ├── AutoModel.from_pretrained()     → DiT model (~2-4 GB VRAM)
  │      ├── AutoencoderOobleck.from_pretrained() → VAE (~0.5-1 GB VRAM)
  │      ├── AutoModel + AutoTokenizer       → Text encoder + tokenizer (~0.3 GB VRAM)
  │      └── silence_latent.pt               → Small tensor (~negligible)
  │
  ├─── [4] llm_handler.initialize(...)  (if requires_lm)
  │      ├── AutoTokenizer.from_pretrained() → LM tokenizer (~80-90s load time!)
  │      ├── MetadataConstrainedLogitsProcessor → Constrained decoding setup
  │      └── vLLM engine OR PyTorch model    → LM weights (~1-3 GB VRAM)
  │
  ├─── [5] LM pre-generation (sample_mode / format / CoT lyrics)
  ├─── [6] Prompt editing hook
  │
  ├─── [7] generate_music(dit_handler, llm_handler, params, config)
  │      ├── LM phase: llm_handler.generate_with_stop_condition()
  │      │      → Generates metadata + audio codes
  │      └── DiT phase: dit_handler.generate_music()
  │             ├── _prepare_batch()     → Text encoding, audio encoding
  │             ├── Diffusion loop       → DiT inference steps
  │             └── tiled_decode()       → VAE decode latents → audio
  │
  └─── [8] Process exit → ALL VRAM released
```

**Key insight:** Steps [3] and [4] are the expensive cold-start operations. Step [8] is where all models leave VRAM — not because of explicit cleanup, but because the **Python process terminates**.

---

## 2. Model Inventory & VRAM Budget

| Component | Class/File | Loading Method | Approx. VRAM (bf16) | Device Placement |
|-----------|-----------|----------------|---------------------|------------------|
| **DiT Model** | `AceStepHandler.model` | `AutoModel.from_pretrained()` | 2-4 GB | GPU (or CPU if offloading) |
| **VAE (Oobleck)** | `AceStepHandler.vae` | `AutoencoderOobleck.from_pretrained()` | 0.5-1 GB | GPU (or CPU if offloading) |
| **Text Encoder** (Qwen3-Embedding-0.6B) | `AceStepHandler.text_encoder` | `AutoModel.from_pretrained()` | ~0.3 GB | GPU (or CPU if offloading) |
| **Text Tokenizer** | `AceStepHandler.text_tokenizer` | `AutoTokenizer.from_pretrained()` | CPU only | CPU |
| **Silence Latent** | `AceStepHandler.silence_latent` | `torch.load()` | Negligible | Always GPU |
| **5Hz LM** (0.6B–1.7B) | `LLMHandler.llm` | vLLM engine or `AutoModel.from_pretrained()` | 1-3 GB | GPU |
| **LM Tokenizer** | `LLMHandler.llm_tokenizer` | `AutoTokenizer.from_pretrained()` | CPU only | CPU |
| **Constrained Processor** | `LLMHandler.constrained_processor` | Constructor | Negligible | CPU |

**Total estimated VRAM footprint:** 4–8 GB (depending on model variant and whether LM is enabled)

### LoRA State (Optional)

When LoRA adapters are loaded:
- `AceStepHandler._base_decoder` — CPU backup of original decoder weights
- `AceStepHandler._lora_adapter_registry` — Registry of adapter configs
- Additional PEFT/LyCORIS adapter weights merged into `self.model`

---

## 3. Detailed Model Loading Lifecycle

### 3.1 DiT Handler Initialization (`handler.py:497-859`)

```python
# handler.py line 678
self.model = AutoModel.from_pretrained(
    acestep_v15_checkpoint_path,
    trust_remote_code=True,
    attn_implementation=candidate,  # "flash_attention_2" → "sdpa" → "eager"
    torch_dtype=self.dtype,         # bf16 on CUDA, fp32 on CPU/MPS
)
```

**Attention fallback chain:** Flash Attention → SDPA → Eager (tries each, falls back on error)

**Device placement logic (handler.py:698-706):**
```
offload_to_cpu=False     → model.to(device).to(dtype)           # Full GPU
offload_to_cpu=True, offload_dit_to_cpu=False → model.to(device)  # GPU but VAE/text_enc on CPU
offload_to_cpu=True, offload_dit_to_cpu=True  → model.to("cpu")   # CPU, moved to GPU on-demand
```

**Post-load steps:**
1. `model.eval()` — Sets to inference mode
2. Optional `torch.compile()` — Only if `compile_model=True`
3. Optional quantization (int8/fp8/w8a8) via `torchao`
4. Load `silence_latent.pt` — Always kept on GPU

**VAE loading (handler.py:767-791):**
```python
self.vae = AutoencoderOobleck.from_pretrained(vae_checkpoint_path)
# Same offload logic as DiT
self.vae.eval()
```

**Text encoder loading (handler.py:793-803):**
```python
self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
self.text_encoder = AutoModel.from_pretrained(text_encoder_path)
self.text_encoder.eval()
```

### 3.2 LLM Handler Initialization (`llm_inference.py:437-720+`)

**Two possible backends:**

1. **vLLM backend (CUDA only):**
   - Creates a `nanovllm.LLM` engine with GPU memory utilization management
   - Handles KV cache allocation internally
   - VRAM gated: requires ≥ 2 GB free VRAM (`VRAM_SAFE_FREE_GB`)

2. **PyTorch backend (fallback):**
   ```python
   self.llm = AutoModelForCausalLM.from_pretrained(
       model_path,
       torch_dtype=self.dtype,
       device_map=device,  # Direct device placement
   )
   self.llm.eval()
   ```

**Tokenizer loading is SLOW (~80-90s):** This is a known bottleneck documented with a TODO:
```python
# llm_inference.py line 537-539
logger.info("loading 5Hz LM tokenizer... it may take 80~90s")
# TODO: load tokenizer too slow, not found solution yet
```

### 3.3 Summary: What Happens on `cli.py main()` Exit

When `main()` returns (or the process exits):

1. Python's garbage collector destroys `dit_handler` and `llm_handler` local variables
2. Their `__del__` methods (if any) or reference count drops trigger cleanup
3. PyTorch CUDA contexts are released
4. **The OS reclaims ALL GPU memory** allocated by the process
5. No explicit `gc.collect()` or `torch.cuda.empty_cache()` is called in the happy path

**There is exactly ONE explicit cleanup path:** `LLMHandler.unload()` (llm_inference.py:81-114) — but it is **never called** by `cli.py` during normal execution.

---

## 4. VRAM Management Mechanisms

### 4.1 CPU Offloading (handler.py)

The `offload_to_cpu` flag controls a **manual offloading strategy:**

- When `offload_to_cpu=True` and `offload_dit_to_cpu=False`: DiT stays on GPU, VAE and text encoder live on CPU and are moved to GPU on-demand during generation
- When both are True: Everything starts on CPU, moved to GPU per-operation

**Offloading is tracked:** `self.current_offload_cost` accumulates the time spent moving tensors between CPU↔GPU.

### 4.2 VRAM Guard System (memory_utils.py:105-155)

Before each generation, `_vram_guard_reduce_batch()` estimates whether the requested batch size fits in free VRAM:

```python
# Per-sample VRAM estimate: 0.5 GB baseline + 0.15 GB per minute over 60s
per_sample_gb = 0.5 + max(0.0, 0.15 * (duration_sec - 60.0) / 60.0)
# Base models use 2x
if "base" in model_name.lower():
    per_sample_gb *= 2.0
```

If insufficient VRAM, it auto-reduces batch size rather than OOMing.

### 4.3 Tiled VAE Decoding (handler.py:1776-1970)

For long audio, the VAE decodes in chunks to avoid OOM:
- `_get_auto_decode_chunk_size()` selects chunk size based on free VRAM (128-512)
- `_should_offload_wav_to_cpu()` decides whether to move decoded audio to CPU immediately
- Three fallback paths: GPU full → CPU offload → full CPU decode

### 4.4 Explicit VRAM Cleanup Calls

| Location | When Called | What It Does |
|----------|-----------|-------------|
| `handler.py:649-654` | Before DiT load in `initialize_service` | Deletes old model, `empty_cache`, `synchronize` |
| `llm_inference.py:81-114` | `LLMHandler.unload()` | Full cleanup: model, tokenizer, gc.collect, empty_cache |
| `llm_inference.py:533-535` | Before LM load in `initialize()` | `empty_cache`, `synchronize` |
| `llm_inference.py:670` | After vLLM engine fails, before PyTorch fallback | `empty_cache` |
| `llm_inference.py:2299` | During LM generation error recovery | `empty_cache` |
| `llm_inference.py:3872` | During batch generation cleanup | `empty_cache` |

**Notable absence:** There is NO `torch.cuda.empty_cache()` call at the end of `generate_music()` or `main()`. Cleanup happens only through process termination.

---

## 5. Why Models Are Unloaded Today

### Root Cause: Single-Shot CLI Architecture

```python
# cli.py lines 1993-1994
if __name__ == "__main__":
    main()
```

`cli.py` is a **single-shot script:**
1. Parse args → Load models → Generate → Exit
2. The Python process terminates after one generation
3. All VRAM is released by OS process cleanup

### Cost of Each Invocation

| Phase | Duration | Notes |
|-------|----------|-------|
| GPU config detection | <1s | Fast |
| DiT model loading | ~5-15s | `from_pretrained` + device transfer |
| VAE loading | ~2-5s | Smaller model |
| Text encoder loading | ~2-3s | 0.6B model |
| **LM tokenizer loading** | **~80-90s** | **Known bottleneck, dominates cold start** |
| LM model loading | ~5-20s | Depends on vLLM vs PT, model size |
| **Total cold start** | **~95-135s** | Before any audio is generated |
| Actual generation | ~10-60s | Depends on duration, steps, batch size |

**The tokenizer load alone (80-90s) makes repeated invocations extremely wasteful.**

---

## 6. Singleton / Persistent Server Refactoring Plan

### Option A: FastAPI Persistent Server (Recommended)

**Architecture:**

```
server.py
  │
  ├── Startup event: Load models ONCE
  │   ├── dit_handler = AceStepHandler()
  │   ├── dit_handler.initialize_service(...)
  │   ├── llm_handler = LLMHandler()
  │   └── llm_handler.initialize(...)
  │
  ├── POST /generate  → generate_music(dit_handler, llm_handler, params, config)
  ├── POST /understand → understand_music(llm_handler, audio_codes, ...)
  ├── POST /sample    → create_sample(llm_handler, ...)
  ├── GET  /health    → VRAM status, model status
  └── POST /unload    → Explicit cleanup for LoRA swaps etc.
```

**Minimal changes required:**

1. **New file: `server.py`** — FastAPI app with model singletons
2. **Zero changes to `acestep/handler.py`** — Already supports reuse (has `self.model`, `self.vae`, etc.)
3. **Zero changes to `acestep/llm_inference.py`** — Already has `unload()` method
4. **Zero changes to `acestep/inference.py`** — `generate_music()` already takes handler instances as parameters
5. **`cli.py` unchanged** — Can coexist with the server

**Key design insight:** The `generate_music()` function in `inference.py` already takes `dit_handler` and `llm_handler` as parameters — it doesn't create them internally. This means the refactoring is trivial: just keep the handlers alive between invocations.

### Option B: Interactive Loop CLI

If a full HTTP server is unnecessary, a simpler approach:

```python
# keep_models_loaded.py
def main():
    # Load once
    dit_handler, llm_handler = load_models(args)
    
    print("Models loaded. Enter 'generate' to create music, 'quit' to exit.")
    while True:
        command = input("> ")
        if command == "quit":
            break
        elif command == "generate":
            params = get_params_from_user()
            result = generate_music(dit_handler, llm_handler, params, config)
            print(f"Saved: {result.audios}")
```

**Pros:** Simpler, no HTTP overhead
**Cons:** Single-user, no remote access, harder to integrate with other tools

### Option C: Singleton Pattern (Weakest)

```python
class ModelSingleton:
    _instance = None
    
    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.dit_handler = AceStepHandler()
            cls._instance.dit_handler.initialize_service(...)
            # ... etc
        return cls._instance
```

**Limitation:** Still dies when the process exits. Only useful if embedded in a longer-running application.

### Recommended Approach: Option A (FastAPI Server)

**Dependencies already present:** `fastapi` and `uvicorn` are already in `pyproject.toml`.

**Implementation outline:**

```python
# server.py (new file)
import uvicorn
from fastapi import FastAPI
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import generate_music, GenerationParams, GenerationConfig

app = FastAPI(title="ACE-Step Inference Server")

# Global model instances — loaded once at startup
dit_handler: AceStepHandler = None
llm_handler: LLMHandler = None

@app.on_event("startup")
async def startup():
    global dit_handler, llm_handler
    dit_handler = AceStepHandler()
    dit_handler.initialize_service(
        project_root=".",
        config_path="acestep-v15-turbo",
        device="cuda",
    )
    llm_handler = LLMHandler()
    llm_handler.initialize(
        checkpoint_dir="./checkpoints",
        lm_model_path="acestep-5Hz-lm-1.7B",
        backend="vllm",
        device="cuda",
    )

@app.post("/generate")
async def generate(params: dict):
    gen_params = GenerationParams(**params)
    config = GenerationConfig(**params.get("config", {}))
    result = generate_music(dit_handler, llm_handler, gen_params, config)
    return result.to_dict()

@app.get("/health")
async def health():
    return {
        "dit_loaded": dit_handler.model is not None,
        "llm_loaded": llm_handler.llm_initialized,
        "device": dit_handler.device,
    }
```

---

## 7. Quick Reference: Key File Locations

| Concern | File | Key Lines |
|---------|------|-----------|
| CLI entry point | `cli.py` | `main()` at line 989 |
| DiT handler class | `acestep/handler.py` | `AceStepHandler` at line 63 |
| DiT model loading | `acestep/handler.py` | `initialize_service()` at line 497 |
| DiT generation | `acestep/handler.py` | `generate_music()` at line 2293 |
| LLM handler class | `acestep/llm_inference.py` | `LLMHandler` at line 46 |
| LLM model loading | `acestep/llm_inference.py` | `initialize()` at line 437 |
| LLM cleanup | `acestep/llm_inference.py` | `unload()` at line 81 |
| Inference orchestrator | `acestep/inference.py` | `generate_music()` at line 310 |
| VRAM guard | `acestep/core/generation/handler/memory_utils.py` | Full file |
| GPU config | `acestep/gpu_config.py` | GPU tier detection |
| Model download | `acestep/model_downloader.py` | Auto-download logic |
| CPU↔GPU offload | `acestep/handler.py` | Lines 698-706, 770-801 |
| Tiled VAE decode | `acestep/handler.py` | `tiled_decode()` at line 1776 |
| LoRA management | `acestep/core/generation/handler/lora_manager.py` | LoRA lifecycle |

### Handler Dependency Graph

```
cli.py main()
  └── acestep/inference.py::generate_music()
        ├── AceStepHandler  (acestep/handler.py)
        │   ├── Mixins from acestep/core/generation/handler/
        │   │   ├── init_service.py    (InitServiceMixin)
        │   │   ├── memory_utils.py    (MemoryUtilsMixin)
        │   │   ├── diffusion.py       (DiffusionMixin)
        │   │   ├── io_audio.py        (IoAudioMixin)
        │   │   ├── lora_manager.py    (LoraManagerMixin)
        │   │   ├── metadata_utils.py  (MetadataMixin)
        │   │   ├── padding_utils.py   (PaddingMixin)
        │   │   ├── progress.py        (ProgressMixin)
        │   │   ├── prompt_utils.py    (PromptMixin)
        │   │   └── task_utils.py      (TaskUtilsMixin)
        │   └── Models: DiT + VAE + TextEncoder + Tokenizer
        │
        └── LLMHandler  (acestep/llm_inference.py)
            └── Models: 5Hz LM (vLLM or PyTorch) + Tokenizer
```

---

## Summary

The codebase is **already well-architected for persistent mode:**

1. **`generate_music()`** takes handler instances as parameters — no internal model creation
2. **`AceStepHandler`** and **`LLMHandler`** are stateful classes that maintain loaded models
3. **`LLMHandler.unload()`** already exists for explicit cleanup
4. **FastAPI + Uvicorn** are already dependencies

The only missing piece is a **server entry point** that:
- Loads models once at startup
- Exposes HTTP endpoints that reuse the loaded handlers
- Optionally supports LoRA hot-swapping and model switching

**Estimated refactoring effort:** ~100-200 lines of new code in a single `server.py` file. Zero modifications to existing inference code.
