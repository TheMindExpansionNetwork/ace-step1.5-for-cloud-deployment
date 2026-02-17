"""
ACE-Step 1.5 Persistent Inference Server

This server loads models once at startup and keeps them resident in VRAM along with
API endpoints for generation. It uses the exact same inference logic as the CLI.

Usage:
    python server.py
    # OR
    uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
"""

import os
import sys
import time
import asyncio
import gc
import json
import torch
import dataclasses
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Union, Set

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import (
    generate_music,
    GenerationParams,
    GenerationConfig,
    GenerationResult
)
from acestep.gpu_config import get_gpu_config, set_global_gpu_config

# --- Global Persistent State ---
# These hold the model references so they are never garbage collected
_dit_handler: Optional[AceStepHandler] = None
_llm_handler: Optional[LLMHandler] = None

# --- Configuration defaults ---
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"

# --- Helper Functions for Dynamic Safety ---

def get_dataclass_fields(cls) -> Set[str]:
    """Dynamically retrieve valid field names from a dataclass."""
    return {f.name for f in dataclasses.fields(cls)}

def safe_instantiate(data_cls, **kwargs):
    """
    Instantiate a dataclass using only valid fields from kwargs.
    Filters out unsupported arguments to prevent TypeErrors.
    """
    valid_fields = get_dataclass_fields(data_cls)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
    
    # Log dropped fields for debugging (explicitly ignoring 'save_dir' which is handled separately)
    dropped = set(kwargs.keys()) - set(filtered_kwargs.keys())
    # remove known non-fields to reduce noise
    dropped.discard('save_dir') 
    
    if dropped:
        logger.debug(f"ℹ️ Dropped arguments for {data_cls.__name__}: {dropped}")
        
    return data_cls(**filtered_kwargs)

# --- Pydantic Models ---

class GenerateRequest(BaseModel):
    # Core prompts
    caption: str = Field(..., description="Main text prompt for music generation")
    lyrics: Union[str, List[str]] = Field("", description="Lyrics (string or list of strings)")
    instrumental: bool = Field(False, description="Generate instrumental music regardless of lyrics")

    # Musical parameters (Matched to GenerationParams names)
    bpm: Optional[int] = Field(None, description="Beats per minute")
    keyscale: str = Field("", description="Key and scale (e.g. 'C Major')")
    timesignature: str = Field("", description="Time signature (e.g. '4/4')")
    vocal_language: str = Field("unknown", description="Vocal language code (en, zh, ja, etc.)")
    
    # Generation parameters
    inference_steps: int = Field(8, description="Number of diffusion steps (8 for turbo, 32+ for base)")
    guidance_scale: float = Field(7.0, description="CFG scale")
    seed: int = Field(-1, description="Random seed (-1 for random)")
    duration: float = Field(-1.0, description="Duration in seconds (<0 for auto)")
    batch_size: int = Field(1, description="Number of variations to generate")
    
    # Audio Post-Processing
    enable_normalization: bool = Field(True, description="Enable loudness normalization")
    normalization_db: float = Field(-1.0, description="Target loudness in dB")
    
    # Latent Post-Processing
    latent_shift: float = Field(0.0, description="Additive shift on DiT latents")
    latent_rescale: float = Field(1.0, description="Multiplicative rescale on DiT latents")
    
    # Advanced DiT
    use_adg: bool = Field(False, description="Use Adaptive Dual Guidance (base model only)")
    cfg_interval_start: float = Field(0.0, description="CFG start ratio")
    cfg_interval_end: float = Field(1.0, description="CFG end ratio")
    shift: float = Field(1.0, description="Timestep shift factor")
    infer_method: str = Field("ode", description="Diffusion method: 'ode' or 'sde'")
    timesteps: Optional[List[float]] = Field(None, description="Custom timesteps list")

    # Advanced LM / Thinking
    thinking: bool = Field(True, description="Enable LLM thinking/refinement")
    lm_temperature: float = Field(0.85, description="LLM sampling temperature")
    lm_cfg_scale: float = Field(2.0, description="LLM CFG scale")
    lm_top_k: int = Field(0, description="LLM top-k (0 disables)")
    lm_top_p: float = Field(0.9, description="LLM top-p nucleus sampling")
    lm_negative_prompt: str = Field("NO USER INPUT", description="LLM negative prompt")
    
    # CoT Control
    use_cot_metas: bool = Field(True, description="Use CoT for metadata")
    use_cot_caption: bool = Field(True, description="Use CoT for caption refinement")
    use_cot_lyrics: bool = Field(False, description="Use CoT for lyrics generation")
    use_cot_language: bool = Field(True, description="Use CoT for language detection")
    use_constrained_decoding: bool = Field(True, description="Use constrained decoding for specific fields")

    # CoT Overrides (if thinking=True)
    cot_bpm: Optional[int] = Field(None, description="Override CoT BPM")
    cot_keyscale: str = Field("", description="Override CoT Key")
    cot_timesignature: str = Field("", description="Override CoT Time Signature")
    cot_duration: Optional[float] = Field(None, description="Override CoT Duration")
    cot_vocal_language: str = Field("unknown", description="Override CoT Language")
    cot_caption: str = Field("", description="Override CoT Caption")
    cot_lyrics: str = Field("", description="Override CoT Lyrics")

    # Config
    audio_format: str = Field("flac", description="Output format (wav, mp3, flac)")
    allow_lm_batch: bool = Field(True, description="Allow LM batch processing")
    use_random_seed: bool = Field(False, description="Use random seed per batch item")
    seeds: Optional[Union[int, List[int]]] = Field(None, description="Specific seeds for batch")
    
    # Task specific
    task_type: str = Field("text2music", description="Task type (text2music, cover, repaint, lego, extract, complete)")
    
    # Task Inputs (Absolute file paths on server)
    reference_audio: Optional[str] = Field(None, description="Path to reference audio (for cover/style transfer)")
    src_audio: Optional[str] = Field(None, description="Path to source audio (for repaint, lego, extract, complete)")
    audio_codes: Optional[str] = Field(None, description="Pre-computed audio codes string")
    
    # Task Controls
    repainting_start: float = Field(0.0, description="Start time for repainting (seconds)")
    repainting_end: float = Field(-1.0, description="End time for repainting (seconds)")
    audio_cover_strength: float = Field(1.0, description="Strength of cover reference (0.0-1.0)")
    cover_noise_strength: float = Field(0.0, description="Cover noise strength (0=pure noise, 1=closest to src)")

class HealthResponse(BaseModel):
    status: str
    vram_used_gb: float
    vram_total_gb: float
    dit_loaded: bool
    llm_loaded: bool
    device: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Event:
    1. Loads models at startup (persisting in global vars).
    2. Keeps them alive during the server runtime.
    3. Cleans up on shutdown.
    """
    global _dit_handler, _llm_handler

    logger.info("🚀 Starting ACE-Step Persistence Server...")
    
    # 1. GPU Setup
    gpu_config = get_gpu_config()
    set_global_gpu_config(gpu_config)
    logger.info(f"GPU Detected: {gpu_config.gpu_memory_gb:.2f} GB VRAM")

    # 2. Configuration from Env or Defaults
    project_root = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    config_path = os.getenv("ACESTEP_CONFIG_PATH", "acestep-v15-turbo")
    device = os.getenv("ACESTEP_DEVICE", "auto")
    
    # Environment flags
    # Note: We default offload to False to ensure persistence, unless explicit
    offload_to_cpu = os.getenv("ACESTEP_OFFLOAD_TO_CPU", "").lower() in ("true", "1", "yes")
    offload_dit_to_cpu = os.getenv("ACESTEP_OFFLOAD_DIT_TO_CPU", "").lower() in ("true", "1", "yes")
    compile_model = os.getenv("ACESTEP_COMPILE_MODEL", "").lower() in ("true", "1", "yes")
    
    # 3. Initialize DiT Handler
    logger.info(f"Initializing DiT Handler ({config_path})...")
    _dit_handler = AceStepHandler()
    
    t0 = time.time()
    status, ok = _dit_handler.initialize_service(
        project_root=project_root,
        config_path=config_path,
        device=device,
        offload_to_cpu=offload_to_cpu,
        offload_dit_to_cpu=offload_dit_to_cpu,
        compile_model=compile_model
    )
    
    if not ok:
        logger.error(f"❌ Failed to load DiT: {status}")
        # We don't exit here to allow /reload to fix it later via API if needed
    else:
        logger.info(f"✅ DiT Loaded in {time.time() - t0:.2f}s")

    # 4. Initialize LLM Handler
    init_llm = os.getenv("ACESTEP_INIT_LLM", "true").lower() not in ("false", "0", "no")
    
    if init_llm:
        logger.info("Initializing LLM Handler...")
        _llm_handler = LLMHandler()
        lm_model_path = os.getenv("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-1.7B")
        backend = os.getenv("ACESTEP_LM_BACKEND", "vllm")
        
        t1 = time.time()
        lm_status, lm_ok = _llm_handler.initialize(
            checkpoint_dir=checkpoint_dir,
            lm_model_path=lm_model_path,
            backend=backend,
            device=device,
            offload_to_cpu=offload_to_cpu
        )
        
        if lm_ok:
            logger.info(f"✅ LLM Loaded in {time.time() - t1:.2f}s")
        else:
            logger.warning(f"⚠️ LLM Load Warning: {lm_status}")
    
    logger.info("✨ Server Ready! Models are resident in VRAM.")
    
    yield  # Application runs here...
    
    # Shutdown / Cleanup
    logger.info("🛑 Shutting down server, unloading models...")
    _dit_handler = None
    _llm_handler = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(title="ACE-Step API", version="1.5.0", lifespan=lifespan)

# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Returns model status and VRAM usage."""
    vram_used = 0.0
    vram_total = 0.0
    device_name = "cpu"
    
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1e9
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        device_name = torch.cuda.get_device_name(0)
    
    return HealthResponse(
        status="running",
        vram_used_gb=round(vram_used, 2),
        vram_total_gb=round(vram_total, 2),
        dit_loaded=(_dit_handler is not None and _dit_handler.model is not None),
        llm_loaded=(_llm_handler is not None and _llm_handler.llm_initialized),
        device=device_name
    )

@app.get("/generation-schema")
async def get_generation_schema():
    """
    Returns the supported fields and their types for GenerationParams and GenerationConfig.
    Useful for clients to dynamically adapt to backend changes.
    """
    def get_fields_info(cls):
        info = {}
        for f in dataclasses.fields(cls):
            type_name = str(f.type)
            # Try to get cleaner name if possible
            if hasattr(f.type, "__name__"):
                type_name = f.type.__name__
            # Handle Optional/Union typing reprs
            if "typing." in type_name:
                type_name = str(f.type).replace("typing.", "")
            
            info[f.name] = {
                "type": type_name,
                "default": str(f.default) if f.default != dataclasses.MISSING else None
            }
        return info

    return {
        "GenerationParams": get_fields_info(GenerationParams),
        "GenerationConfig": get_fields_info(GenerationConfig)
    }

@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    Generates music using the pre-loaded models.
    """
    if _dit_handler is None or _dit_handler.model is None:
        raise HTTPException(status_code=503, detail="DiT model is not loaded. Check server logs.")

    logger.info(f"🎨 Generating: {req.caption[:50]}... (Steps: {req.inference_steps}, Batch: {req.batch_size})")

    try:
        # Convert API request to dict
        req_dict = req.model_dump()
        
        # 1. Construct GenerationParams dynamically
        # This filters out any keys in req_dict that aren't in GenerationParams
        # It ensures we never get a TypeError for unexpected arguments
        params = safe_instantiate(GenerationParams, **req_dict)
        
        # 2. Construct GenerationConfig dynamically
        # Note: 'save_dir' must NOT be passed here as it is not in GenerationConfig dataclass
        config = safe_instantiate(GenerationConfig, **req_dict)
        
        # 3. Determine output directory
        save_dir = os.path.join(os.path.dirname(__file__), "output")
        
        # 4. Run generation
        # Using the same generate_music function as CLI
        result: GenerationResult = generate_music(
            dit_handler=_dit_handler,
            llm_handler=_llm_handler,
            params=params,
            config=config,
            save_dir=save_dir # Passed as separate argument, not in config
        )
        
        if not result.success:
            # Check for bad inputs (like file not found) vs server errors
            status_code = 500
            if "File not found" in str(result.error):
                status_code = 400
            
            error_detail = result.error or result.status_message
            logger.error(f"Generation failed: {error_detail}")
            raise HTTPException(status_code=status_code, detail=error_detail)
            
        return {
            "success": True,
            "audios": result.audios,
            "timings": result.extra_outputs.get("time_costs", {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Generation failed with unexpected error")
        # Return structured error
        return {
             "success": False,
             "error": "Internal Server Error",
             "details": str(e),
             "type": type(e).__name__
        }

@app.post("/reload")
async def reload_models():
    """Force re-initialization of the model (useful for changing checkpoints)."""
    global _dit_handler
    
    if _dit_handler is None:
        raise HTTPException(status_code=503, detail="Handler not created")
        
    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.getenv("ACESTEP_CONFIG_PATH", "acestep-v15-turbo")
    
    logger.info("Reloading DiT model...")
    # initialize_service handles unloading the old model safely first
    status, ok = _dit_handler.initialize_service(
        project_root=project_root,
        config_path=config_path,
        device=os.getenv("ACESTEP_DEVICE", "auto")
    )
    
    if not ok:
        raise HTTPException(status_code=500, detail=f"Reload failed: {status}")
        
    return {"message": "Model reloaded successfully", "config": config_path}

class LoraLoadRequest(BaseModel):
    lora_path: str = Field(..., description="Absolute path to the LoRA adapter directory")

@app.post("/v1/models/lora/load")
async def load_lora_adapter(req: LoraLoadRequest):
    """Load a LoRA adapter into the active DiT model."""
    if _dit_handler is None:
        raise HTTPException(status_code=503, detail="DiT model not initialized")
    
    logger.info(f"Loading LoRA from: {req.lora_path}")
    # The handler.load_lora method returns a string message starting with ✅ or ❌
    result = _dit_handler.load_lora(req.lora_path)
    
    if "❌" in result:
        # 400 Bad Request for user errors (file not found), 500 for others?
        # The message usually contains strict validation errors.
        raise HTTPException(status_code=400, detail=result)
    
    return {"message": result, "lora_loaded": True}

@app.post("/v1/models/lora/unload")
async def unload_lora_adapter():
    """Unload the currently active LoRA adapter."""
    if _dit_handler is None:
        raise HTTPException(status_code=503, detail="DiT model not initialized")
        
    logger.info("Unloading LoRA adapter...")
    result = _dit_handler.unload_lora()
    
    if "❌" in result:
        raise HTTPException(status_code=500, detail=result)
        
    return {"message": result, "lora_loaded": False}

if __name__ == "__main__":
    # Standard entry point if running `python server.py`
    uvicorn.run(
        "server:app", 
        host=DEFAULT_HOST, 
        port=DEFAULT_PORT, 
        workers=1,
        log_level="info"
    )
