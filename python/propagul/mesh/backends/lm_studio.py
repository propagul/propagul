"""propagul.mesh.backends.lm_studio — LM Studio backend adapter.

LM Studio exposes an OpenAI-compatible API on port 1234.
No pull/delete — models are managed through the LM Studio GUI.

This adapter is read-only: it collects telemetry (model list, health)
but cannot modify models remotely.

Default endpoint: http://localhost:1234
"""

import json
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("propagul.mesh.backends.lm_studio")

DEFAULT_URL = "http://localhost:1234"


@dataclass
class LmStudioModelInfo:
    """Model metadata from LM Studio /v1/models endpoint."""
    id: str = ""
    object: str = "model"
    owned_by: str = ""


def _http_get(url: str, timeout: float = 5.0) -> Optional[bytes]:
    """GET request via stdlib. Returns response body or None."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as e:
        logger.debug("LM Studio request failed: %s → %s", url, e)
        return None


def get_models(base_url: str = DEFAULT_URL, timeout: float = 5.0) -> List[LmStudioModelInfo]:
    """Fetch loaded models from LM Studio /v1/models endpoint.

    Returns list of LmStudioModelInfo. Empty list if server unreachable.
    """
    raw = _http_get(f"{base_url}/v1/models", timeout=timeout)
    if raw is None:
        return []

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []

    models = []
    for item in data.get("data", []):
        models.append(LmStudioModelInfo(
            id=item.get("id", ""),
            object=item.get("object", "model"),
            owned_by=item.get("owned_by", ""),
        ))
    return models


def check_health(base_url: str = DEFAULT_URL, timeout: float = 3.0) -> bool:
    """Check LM Studio health by probing /v1/models.

    LM Studio doesn't have a dedicated /health endpoint.
    """
    raw = _http_get(f"{base_url}/v1/models", timeout=timeout)
    return raw is not None


def collect_telemetry(base_url: str = DEFAULT_URL, timeout: float = 5.0) -> dict:
    """Collect full telemetry from an LM Studio server.

    Returns a dict suitable for merging into the agent's heartbeat payload.
    """
    models = get_models(base_url, timeout)
    healthy = check_health(base_url, timeout)

    model_list = []
    for m in models:
        model_list.append({
            "name": m.id,
            "size_gb": 0,
            "parameter_size": "",
            "quantization": "",
            "backend": "lm_studio",
        })

    return {
        "backend": "lm_studio",
        "url": base_url,
        "healthy": healthy,
        "models": model_list,
        "model_count": len(models),
        "running_count": len(models),
    }
