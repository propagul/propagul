"""propagul.mesh.backends.llamacpp — llama.cpp backend adapter.

llama.cpp (llama-server) exposes a REST API on port 8080 by default.
Models are loaded at server start via CLI (e.g. `llama-server -m model.gguf`).

This adapter is read-only: it collects telemetry (loaded model, health,
active slot count) but cannot modify models remotely.

Default endpoint: http://localhost:8080

API Surface (llama-server ≥ b2000):
    GET /health     → {"status": "ok"} or {"status": "loading model"}
    GET /slots      → [{...slot data...}] (optional, requires --slots flag)
    GET /props      → {"default_generation_settings": {...}}
    GET /v1/models  → OpenAI-compatible model list (newer builds)

Edge Cases:
    - /slots may not exist (requires --slots CLI flag)
    - /v1/models may not exist (older builds)
    - Port 8080 is shared with TGI — detect.py handles disambiguation
      via /info (TGI) vs /health (llama.cpp) ordering
"""

import json
import logging
import os.path
import urllib.request
import urllib.error
from typing import Optional

logger = logging.getLogger("propagul.mesh.backends.llamacpp")

DEFAULT_URL = "http://localhost:8080"


def _http_get(url: str, timeout: float = 5.0) -> Optional[bytes]:
    """GET request via stdlib. Returns response body or None."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as e:
        logger.debug("llama.cpp request failed: %s → %s", url, e)
        return None


def check_health(base_url: str = DEFAULT_URL, timeout: float = 3.0) -> dict:
    """Check llama.cpp server health.

    Returns dict with 'healthy' bool and 'status' string.
    Possible status values: "ok", "loading model", "error", "no slots available"
    """
    raw = _http_get(f"{base_url}/health", timeout=timeout)
    if raw is None:
        return {"healthy": False, "status": "unreachable"}

    try:
        data = json.loads(raw)
        status = data.get("status", "unknown")
        return {"healthy": status == "ok", "status": status}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"healthy": False, "status": "parse_error"}


def get_props(base_url: str = DEFAULT_URL, timeout: float = 5.0) -> dict:
    """Fetch server properties from /props endpoint.

    Returns generation settings including model path and context size.
    Empty dict if unavailable.
    """
    raw = _http_get(f"{base_url}/props", timeout=timeout)
    if raw is None:
        return {}

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def get_slots(base_url: str = DEFAULT_URL, timeout: float = 5.0) -> list[dict]:
    """Fetch active inference slots from /slots endpoint.

    Requires llama-server started with --slots flag.
    Returns list of slot dicts or empty list if unavailable.
    """
    raw = _http_get(f"{base_url}/slots", timeout=timeout)
    if raw is None:
        return []

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []


def get_models_v1(base_url: str = DEFAULT_URL, timeout: float = 5.0) -> list[dict]:
    """Fetch models from OpenAI-compatible /v1/models endpoint.

    Available in newer llama.cpp builds. Returns list of model dicts
    or empty list if unavailable.
    """
    raw = _http_get(f"{base_url}/v1/models", timeout=timeout)
    if raw is None:
        return []

    try:
        data = json.loads(raw)
        return data.get("data", [])
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []


def _extract_model_name(props: dict) -> str:
    """Extract a human-readable model name from server properties.

    llama.cpp /props returns the full file path in default_generation_settings.
    We extract just the filename and strip the extension.
    """
    settings = props.get("default_generation_settings", {})
    model_path = settings.get("model", "")

    if not model_path:
        return "unknown"

    # Extract filename from full path
    # e.g., "/models/llama-3-8b-q4_k_m.gguf" → "llama-3-8b-q4_k_m"
    basename = os.path.basename(model_path)
    # Strip .gguf extension
    if basename.lower().endswith(".gguf"):
        basename = basename[:-5]
    return basename or "unknown"


def collect_telemetry(base_url: str = DEFAULT_URL, timeout: float = 5.0) -> dict:
    """Collect full telemetry from a llama.cpp server.

    Returns a dict suitable for merging into the agent's heartbeat payload.
    Format matches the Ollama/vLLM adapter schema.

    Strategy:
        1. /health → server status
        2. /v1/models → OpenAI-compatible model list (preferred)
        3. /props → fallback for model name + context size
        4. /slots → active inference slot count (optional)
    """
    health = check_health(base_url, timeout)
    v1_models = get_models_v1(base_url, timeout)
    props = get_props(base_url, timeout)
    slots = get_slots(base_url, timeout)

    model_list = []

    if v1_models:
        # Prefer OpenAI-compatible endpoint
        for m in v1_models:
            model_list.append({
                "name": m.get("id", "unknown"),
                "size_gb": 0,  # llama.cpp doesn't expose file size via API
                "parameter_size": "",
                "quantization": "",
                "backend": "llama_cpp",
            })
    elif props:
        # Fallback: extract from /props
        model_name = _extract_model_name(props)
        settings = props.get("default_generation_settings", {})
        model_list.append({
            "name": model_name,
            "size_gb": 0,
            "parameter_size": "",
            "quantization": "",
            "backend": "llama_cpp",
            "context_size": settings.get("n_ctx", 0),
        })

    # Slot metrics (if available)
    active_slots = sum(1 for s in slots if s.get("is_processing", False))
    total_slots = len(slots)

    result = {
        "backend": "llama_cpp",
        "url": base_url,
        "healthy": health["healthy"],
        "status": health["status"],
        "models": model_list,
        "model_count": len(model_list),
        # llama.cpp loads one model at a time — it's either running or not
        "running_count": len(model_list) if health["healthy"] else 0,
    }

    if slots:
        result["slots"] = {
            "total": total_slots,
            "active": active_slots,
            "idle": total_slots - active_slots,
        }

    return result
