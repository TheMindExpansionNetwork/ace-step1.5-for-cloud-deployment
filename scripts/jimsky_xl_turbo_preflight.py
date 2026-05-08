#!/usr/bin/env python3
"""CPU-only preflight for the Jimsky ACE-Step XL Turbo deployment fork."""
from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Any

DEFAULT_MODEL = "acestep-v15-xl-turbo"
DEFAULT_REPO = "ACE-Step/acestep-v15-xl-turbo"


def main() -> int:
    """Verify local wiring without downloading model weights or starting a GPU."""
    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    result: dict[str, Any] = {
        "ok": True,
        "repo_root": str(root),
        "default_model": DEFAULT_MODEL,
        "default_repo": DEFAULT_REPO,
        "gpu_started": False,
        "model_download_started": False,
        "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")),
    }
    # Avoid importing the full ACE-Step package here: dependency imports such as
    # loguru/torch may not exist on a clean control host, and this preflight must
    # remain CPU-only/no-install/no-download. Validate the registry by source text.
    downloader_source = (root / "acestep" / "model_downloader.py").read_text()
    server_source = (root / "server.py").read_text()
    result.update(
        xl_turbo_registered=f'"{DEFAULT_MODEL}": "{DEFAULT_REPO}"' in downloader_source,
        model_in_available=DEFAULT_MODEL in downloader_source,
        server_defaults_to_xl=f'os.getenv("ACESTEP_CONFIG_PATH", "{DEFAULT_MODEL}")' in server_source,
        available_xl_models=sorted({
            token.strip('"')
            for token in downloader_source.replace("'", '"').split()
            if "acestep-v15-xl" in token
        }),
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") and result.get("xl_turbo_registered") and result.get("server_defaults_to_xl") else 1


if __name__ == "__main__":
    raise SystemExit(main())
