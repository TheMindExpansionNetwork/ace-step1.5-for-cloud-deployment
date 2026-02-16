# Work Handoff Log: ACE-Step 1.5 Persistent Server Implementation

**Date:** 2026-02-16
**Status:** Analysis Complete, Implementation Ready
**Objective:** Transform the CLI-only ACE-Step 1.5 repository into a persistent API service that keeps models loaded in VRAM.

---

## ✅ Completed Work

1.  **Repository Analysis & Comparison**
    *   Cloned the reference repository (`https://github.com/ace-step/ACE-Step-1.5.git`) to `ace-step-ref/`.
    *   Performed a deep structural comparison between the local user repository and the reference.
    *   Verified that core inference logic (`handler.py`, `llm_inference.py`) is **functionally identical** in both.
    *   Identified that the local repository is missing the `acestep/api_server.py` and `acestep/ui/` components, which contain the persistence logic.

2.  **Root Cause Identification**
    *   Confirmed that the current `cli.py` behaves correctly by design: it initializes models, generates audio, and then exits, causing the OS to reclaim VRAM.
    *   Confirmed that `offload_to_cpu` flags are NOT the cause of the unloading issue.
    *   Identified that the reference repository achieves persistence via a long-running `uvicorn` process and a FastAPI `lifespan` handler that stores model instances in `app.state`.

3.  **Architecture Design for Solution**
    *   Designed a "Production-Grade Persistent Service Redesign" (Option C from the deep report).
    *   Decided to implement a new `server.py` rather than porting the massive `api_server.py` from the reference logic. This functionality will:
        *   Load models once at startup.
        *   Expose `/generate`, `/health`, and `/reload` endpoints.
        *   Reuse the existing `acestep.inference.generate_music` function.
        *   Maintain all VRAM safety guards (memory checks, batch reduction).

4.  **Documentation**
    *   Created `docs/ARCHITECTURE_ANALYSIS.md`: Detailed breakdown of VRAM usage and model loading flow.
    *   Created `docs/DEEP_COMPARISON_REPORT.md`: Comprehensive comparison and technical recommendations.

---

## 🧩 Current State

*   **Local Repository:** Contains the functional CLI and model layer. Missing the API server layer.
*   **Reference Clone:** Currently located at `ace-step-ref/` (can be deleted once implementation is done, or kept for reference).
*   **Context:** We have mapped exactly how `api_server.py` in the reference repo initializes the `AceStepHandler` and `LLMHandler` and stores them in `app.state`.

---

## 📋 Next Steps (For the Next Agent)

**Goal:** Implement the `server.py` file to enable persistent model serving.

### 1. Create `server.py` in the root directory
Implement the FastAPI server code. Key requirements for the code:
*   **Imports:** Import `AceStepHandler` from `acestep.handler` and `LLMHandler` from `acestep.llm_inference`.
*   **Lifespan Context:** Use `@asynccontextmanager` to initialize handlers *once* and store them in global variables or `app.state`.
*   **Environment Handling:** Read `ACESTEP_CONFIG_PATH`, `ACESTEP_DEVICE`, etc., to configure the handlers.
*   **Endpoints:**
    *   `POST /generate`: Accept JSON payload (caption, lyrics, etc.), wrap them in `GenerationParams`/`GenerationConfig`, call `generate_music(...)`, and return the result.
    *   `GET /health`: access `torch.cuda` memory stats and check if handlers are not None.
    *   `POST /reload`: Allow re-running `initialize_service()` on the existing handler to swap checkpoints.
*   **VRAM Safety:** Ensure the server runs with `workers=1` (standard `uvicorn` arg) to prevent multiple processes trying to allocate GPU memory.

### 2. Verify Dependencies
*   Check if `fastapi` and `uvicorn` are installed in the user's environment.
*   If not, suggest installing them: `pip install "uvicorn[standard]" fastapi`.

### 3. Verification
*   Start the server: `python server.py` (or `uvicorn server:app ...`).
*   Test with `curl` to ensure the model stays loaded between requests (monitoring VRAM via `watch -n 1 nvidia-smi` or the `/health` endpoint).

### 4. Cleanup
*   Remove the `ace-step-ref/` directory to save space and clean up the workspace.

---

## 🔗 Reference Code Snippet (Blueprint)

The prompt for the next agent should rely on this blueprint derived from the analysis:

```python
# server.py blueprint
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from acestep.handler import AceStepHandler
# ... imports ...

_dit = None  # Global handler reference

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _dit
    _dit = AceStepHandler()
    _dit.initialize_service(...) # Load once
    yield
    # Cleanup on exit

app = FastAPI(lifespan=lifespan)

@app.post("/generate")
def generate(req):
    # Reuse _dit
    return generate_music(_dit, ...)
```
