"""propagul.mesh.backends.vllm — vLLM backend adapter.

vLLM exposes an OpenAI-compatible API. No pull/delete — models are loaded
via CLI arguments at server start (e.g. `vllm serve meta-llama/Llama-3-8B`).

This adapter is read-only: it collects telemetry (model list, health,
Prometheus metrics) but cannot modify models remotely.

Default endpoint: http://localhost:8000
"""

import json
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("propagul.mesh.backends.vllm")

# Default vLLM server URL
DEFAULT_URL = "http://localhost:8000"


@dataclass
class VllmModelInfo:
    """Model metadata from vLLM /v1/models endpoint."""
    id: str
    object: str = "model"
    owned_by: str = ""
    root: str = ""
    max_model_len: int = 0


@dataclass
class VllmMetrics:
    """Parsed Prometheus metrics from vLLM /metrics endpoint."""
    num_requests_running: int = 0
    num_requests_waiting: int = 0
    gpu_cache_usage_perc: float = 0.0
    cpu_cache_usage_perc: float = 0.0
    avg_prompt_throughput: float = 0.0
    avg_generation_throughput: float = 0.0


@dataclass
class VllmStatus:
    """Aggregated vLLM server status."""
    healthy: bool = False
    models: List[VllmModelInfo] = field(default_factory=list)
    metrics: Optional[VllmMetrics] = None
    error: str = ""


def _http_get(url: str, timeout: float = 5.0) -> Optional[bytes]:
    """GET request via stdlib. Returns response body or None."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as e:
        logger.debug("vLLM request failed: %s → %s", url, e)
        return None


def get_models(base_url: str = DEFAULT_URL, timeout: float = 5.0) -> List[VllmModelInfo]:
    """Fetch loaded models from vLLM /v1/models endpoint.

    Returns list of VllmModelInfo. Empty list if server unreachable.
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
        models.append(VllmModelInfo(
            id=item.get("id", ""),
            object=item.get("object", "model"),
            owned_by=item.get("owned_by", ""),
            root=item.get("root", ""),
            max_model_len=item.get("max_model_len", 0),
        ))
    return models


def check_health(base_url: str = DEFAULT_URL, timeout: float = 3.0) -> bool:
    """Check vLLM health. Returns True if server is responding."""
    raw = _http_get(f"{base_url}/health", timeout=timeout)
    return raw is not None


def get_metrics(base_url: str = DEFAULT_URL, timeout: float = 5.0) -> Optional[VllmMetrics]:
    """Parse Prometheus metrics from vLLM /metrics endpoint.

    vLLM exposes Prometheus text format. We parse the key metrics.
    Returns VllmMetrics or None if unavailable.
    """
    raw = _http_get(f"{base_url}/metrics", timeout=timeout)
    if raw is None:
        return None

    text = raw.decode("utf-8", errors="replace")
    metrics = VllmMetrics()

    for line in text.splitlines():
        if line.startswith("#"):
            continue
        try:
            if line.startswith("vllm:num_requests_running"):
                metrics.num_requests_running = int(float(line.split()[-1]))
            elif line.startswith("vllm:num_requests_waiting"):
                metrics.num_requests_waiting = int(float(line.split()[-1]))
            elif line.startswith("vllm:gpu_cache_usage_perc"):
                metrics.gpu_cache_usage_perc = float(line.split()[-1])
            elif line.startswith("vllm:cpu_cache_usage_perc"):
                metrics.cpu_cache_usage_perc = float(line.split()[-1])
            elif line.startswith("vllm:avg_prompt_throughput"):
                metrics.avg_prompt_throughput = float(line.split()[-1])
            elif line.startswith("vllm:avg_generation_throughput"):
                metrics.avg_generation_throughput = float(line.split()[-1])
        except (ValueError, IndexError):
            continue

    return metrics


def collect_telemetry(base_url: str = DEFAULT_URL, timeout: float = 5.0) -> dict:
    """Collect full telemetry from a vLLM server.

    Returns a dict suitable for merging into the agent's heartbeat payload.
    Format matches the Ollama adapter's return schema.
    """
    models = get_models(base_url, timeout)
    healthy = check_health(base_url, timeout)
    metrics = get_metrics(base_url, timeout)

    model_list = []
    for m in models:
        model_list.append({
            "name": m.id,
            "size_gb": 0,  # vLLM doesn't expose model file size
            "parameter_size": "",
            "quantization": "",
            "backend": "vllm",
        })

    result = {
        "backend": "vllm",
        "url": base_url,
        "healthy": healthy,
        "models": model_list,
        "model_count": len(models),
        "running_count": len(models),  # vLLM: all loaded models are running
    }

    if metrics:
        result["metrics"] = {
            "requests_running": metrics.num_requests_running,
            "requests_waiting": metrics.num_requests_waiting,
            "gpu_cache_usage": round(metrics.gpu_cache_usage_perc * 100, 1),
            "throughput_prompt": round(metrics.avg_prompt_throughput, 1),
            "throughput_generation": round(metrics.avg_generation_throughput, 1),
        }

    return result
