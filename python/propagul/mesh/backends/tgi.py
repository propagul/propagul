"""propagul.mesh.backends.tgi — HuggingFace Text Generation Inference adapter.

TGI serves a single model per instance. No pull/delete — the model is
specified at container/server start via `--model-id`.

This adapter is read-only: it collects telemetry (model info, health,
Prometheus metrics) but cannot modify models remotely.

Default endpoint: http://localhost:8080
Disambiguation: Port 8080 may also be llama.cpp. TGI has /info endpoint
that returns {"model_id": ...}, while llama.cpp has /health → {"status": "ok"}.
"""

import json
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("propagul.mesh.backends.tgi")

DEFAULT_URL = "http://localhost:8080"


@dataclass
class TgiModelInfo:
    """Model metadata from TGI /info endpoint."""
    model_id: str = ""
    model_dtype: str = ""
    model_device_type: str = ""
    max_total_tokens: int = 0
    max_input_length: int = 0
    max_batch_total_tokens: int = 0
    sha: str = ""
    docker_label: str = ""
    version: str = ""


@dataclass
class TgiMetrics:
    """Parsed Prometheus metrics from TGI /metrics endpoint."""
    queue_size: int = 0
    batch_current_size: int = 0
    inference_duration_sum: float = 0.0
    total_tokens_generated: int = 0


def _http_get(url: str, timeout: float = 5.0) -> Optional[bytes]:
    """GET request via stdlib. Returns response body or None."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as e:
        logger.debug("TGI request failed: %s → %s", url, e)
        return None


def get_info(base_url: str = DEFAULT_URL, timeout: float = 5.0) -> Optional[TgiModelInfo]:
    """Fetch model info from TGI /info endpoint.

    This is the primary way to identify TGI vs llama.cpp on port 8080.
    TGI returns {"model_id": "...", "model_dtype": "...", ...}.
    llama.cpp does NOT have a /info endpoint.
    """
    raw = _http_get(f"{base_url}/info", timeout=timeout)
    if raw is None:
        return None

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    # TGI signature: must have "model_id" field
    if "model_id" not in data:
        return None

    return TgiModelInfo(
        model_id=data.get("model_id", ""),
        model_dtype=data.get("model_dtype", ""),
        model_device_type=data.get("model_device_type", ""),
        max_total_tokens=data.get("max_total_tokens", 0),
        max_input_length=data.get("max_input_length", 0),
        max_batch_total_tokens=data.get("max_batch_total_tokens", 0),
        sha=data.get("sha", ""),
        docker_label=data.get("docker_label", ""),
        version=data.get("version", ""),
    )


def check_health(base_url: str = DEFAULT_URL, timeout: float = 3.0) -> bool:
    """Check TGI health. Returns True if server is responding."""
    raw = _http_get(f"{base_url}/health", timeout=timeout)
    return raw is not None


def get_metrics(base_url: str = DEFAULT_URL, timeout: float = 5.0) -> Optional[TgiMetrics]:
    """Parse Prometheus metrics from TGI /metrics endpoint."""
    raw = _http_get(f"{base_url}/metrics", timeout=timeout)
    if raw is None:
        return None

    text = raw.decode("utf-8", errors="replace")
    metrics = TgiMetrics()

    for line in text.splitlines():
        if line.startswith("#"):
            continue
        try:
            if "tgi_queue_size" in line and not line.startswith("#"):
                metrics.queue_size = int(float(line.split()[-1]))
            elif "tgi_batch_current_size" in line:
                metrics.batch_current_size = int(float(line.split()[-1]))
            elif "tgi_request_inference_duration_sum" in line:
                metrics.inference_duration_sum = float(line.split()[-1])
            elif "tgi_request_generated_tokens_total" in line:
                metrics.total_tokens_generated = int(float(line.split()[-1]))
        except (ValueError, IndexError):
            continue

    return metrics


def collect_telemetry(base_url: str = DEFAULT_URL, timeout: float = 5.0) -> dict:
    """Collect full telemetry from a TGI server.

    Returns a dict suitable for merging into the agent's heartbeat payload.
    """
    info = get_info(base_url, timeout)
    healthy = check_health(base_url, timeout)
    metrics = get_metrics(base_url, timeout)

    model_list = []
    if info:
        model_list.append({
            "name": info.model_id,
            "size_gb": 0,
            "parameter_size": "",
            "quantization": info.model_dtype or "",
            "backend": "tgi",
        })

    result = {
        "backend": "tgi",
        "url": base_url,
        "healthy": healthy,
        "models": model_list,
        "model_count": len(model_list),
        "running_count": len(model_list),  # TGI: if loaded, it's running
    }

    if info:
        result["model_info"] = {
            "model_id": info.model_id,
            "dtype": info.model_dtype,
            "device": info.model_device_type,
            "max_tokens": info.max_total_tokens,
            "version": info.version,
        }

    if metrics:
        result["metrics"] = {
            "queue_size": metrics.queue_size,
            "batch_size": metrics.batch_current_size,
            "tokens_generated": metrics.total_tokens_generated,
        }

    return result


def is_tgi(base_url: str = DEFAULT_URL, timeout: float = 2.0) -> bool:
    """Disambiguate TGI from llama.cpp on port 8080.

    TGI has /info with "model_id". llama.cpp does not.
    """
    info = get_info(base_url, timeout)
    return info is not None
