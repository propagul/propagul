"""propagul.mesh.backends.ollama — Ollama inference engine adapter.

Polls the local Ollama HTTP API to collect:
- Installed models (name, size, quantization, modified date)
- Running models (currently loaded in VRAM)
- Server version and status

Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import json
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import ipaddress as _ipaddress

logger = logging.getLogger("propagul.mesh.backends.ollama")

DEFAULT_OLLAMA_URL = "http://localhost:11434"

# P2-01: SSRF prevention — only allow connections to local/private addresses
_ALLOWED_HOSTS = frozenset({
    "localhost",
    "127.0.0.1",
    "::1",
    "0.0.0.0",
})

# Tailscale/CGNAT range — Python <3.11 doesn't classify as is_private
_CGNAT_NETWORK = _ipaddress.IPv4Network("100.64.0.0/10")


def _validate_url(url: str) -> None:
    """Validate that a URL points to a local/private address.

    Uses ipaddress module for robust validation — prevents SSRF bypasses
    that would pass naive prefix checks.

    Allowed:
        - Known hostnames (localhost, 127.0.0.1, ::1, 0.0.0.0)
        - Loopback (127.0.0.0/8, ::1)
        - Private (RFC1918: 10/8, 172.16/12, 192.168/16; ULA: fc00::/7)
        - CGNAT/Tailscale (100.64.0.0/10, RFC 6598)

    Blocked explicitly:
        - Link-local (169.254.0.0/16) — includes cloud metadata 169.254.169.254

    Raises ValueError if the URL points to a non-local address.
    """
    parsed = urlparse(url)
    host = parsed.hostname or ""

    if host in _ALLOWED_HOSTS:
        return

    try:
        addr = _ipaddress.ip_address(host)
    except ValueError:
        raise ValueError(
            f"SSRF blocked: '{host}' is not a recognized local hostname or IP. "
            f"Only localhost and private/loopback IPs are allowed."
        )

    # Block link-local BEFORE is_private — Python 3.9 classifies
    # 169.254.0.0/16 as is_private=True, which would allow the cloud
    # metadata endpoint 169.254.169.254 through the guard.
    if addr.is_link_local:
        raise ValueError(
            f"SSRF blocked: {host} is a link-local address. "
            f"Cloud metadata endpoint 169.254.169.254 is explicitly blocked."
        )

    if addr.is_loopback or addr.is_private:
        return

    # Allow Tailscale/CGNAT (100.64.0.0/10) — not classified as
    # is_private on Python <3.11.
    if isinstance(addr, _ipaddress.IPv4Address) and addr in _CGNAT_NETWORK:
        return

    raise ValueError(
        f"SSRF blocked: {host} is not a local/private address. "
        f"Only localhost, loopback, RFC1918/ULA, and Tailscale (100.64/10) IPs are allowed."
    )


@dataclass
class OllamaModel:
    """A model installed on an Ollama instance."""
    name: str
    size_bytes: int
    parameter_size: str  # e.g. "8B", "70B"
    quantization: str  # e.g. "Q4_K_M", "F16"
    family: str  # e.g. "llama", "mistral"
    modified_at: str  # ISO timestamp
    digest: str  # SHA256 prefix

    @property
    def size_gb(self) -> float:
        return round(self.size_bytes / (1024 ** 3), 2)


@dataclass
class OllamaRunningModel:
    """A model currently loaded in VRAM."""
    name: str
    size_bytes: int
    vram_bytes: int
    expires_at: str  # When it will be unloaded


@dataclass
class OllamaStatus:
    """Complete status snapshot of a local Ollama instance."""
    online: bool
    version: str
    url: str
    models: list[OllamaModel] = field(default_factory=list)
    running: list[OllamaRunningModel] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def model_count(self) -> int:
        return len(self.models)

    @property
    def running_count(self) -> int:
        return len(self.running)

    @property
    def total_model_size_gb(self) -> float:
        return round(sum(m.size_bytes for m in self.models) / (1024 ** 3), 2)

    @property
    def total_vram_used_bytes(self) -> int:
        return sum(r.vram_bytes for r in self.running)

    def to_dict(self) -> dict:
        """Serialize for telemetry push."""
        return {
            "backend": "ollama",
            "online": self.online,
            "version": self.version,
            "url": self.url,
            "error": self.error,
            "model_count": self.model_count,
            "running_count": self.running_count,
            "total_model_size_gb": self.total_model_size_gb,
            "total_vram_used_bytes": self.total_vram_used_bytes,
            "models": [
                {
                    "name": m.name,
                    "size_gb": m.size_gb,
                    "parameter_size": m.parameter_size,
                    "quantization": m.quantization,
                    "family": m.family,
                    "digest": m.digest[:12],
                }
                for m in self.models
            ],
            "running": [
                {
                    "name": r.name,
                    "size_bytes": r.size_bytes,
                    "vram_bytes": r.vram_bytes,
                }
                for r in self.running
            ],
        }


def _http_get(url: str, timeout: float = 5.0) -> dict:
    """HTTP GET with stdlib only. Returns parsed JSON or raises."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def poll(base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 5.0) -> OllamaStatus:
    """Poll the local Ollama instance for current status.

    This is the main entry point — call every 10 seconds from the agent.
    Returns OllamaStatus with all available information.
    Never raises; returns offline status on any error.
    """
    # P2-01: SSRF guard — only allow local/private URLs
    try:
        _validate_url(base_url)
    except ValueError as e:
        return OllamaStatus(
            online=False, version="", url=base_url,
            error=str(e),
        )

    # 1. Check version / health
    try:
        version_resp = _http_get(f"{base_url}/api/version", timeout=timeout)
        version = version_resp.get("version", "unknown")
    except Exception as e:
        return OllamaStatus(
            online=False,
            version="",
            url=base_url,
            error=f"Connection failed: {e}",
        )

    # 2. Get installed models
    models: list[OllamaModel] = []
    try:
        tags_resp = _http_get(f"{base_url}/api/tags", timeout=timeout)
        for m in tags_resp.get("models", []):
            details = m.get("details", {})
            models.append(OllamaModel(
                name=m.get("name", "unknown"),
                size_bytes=m.get("size", 0),
                parameter_size=details.get("parameter_size", ""),
                quantization=details.get("quantization_level", ""),
                family=details.get("family", ""),
                modified_at=m.get("modified_at", ""),
                digest=m.get("digest", ""),
            ))
    except Exception as e:
        logger.warning("Failed to list models: %s", e)

    # 3. Get running models (loaded in VRAM)
    running: list[OllamaRunningModel] = []
    try:
        ps_resp = _http_get(f"{base_url}/api/ps", timeout=timeout)
        for r in ps_resp.get("models", []):
            running.append(OllamaRunningModel(
                name=r.get("name", "unknown"),
                size_bytes=r.get("size", 0),
                vram_bytes=r.get("size_vram", 0),
                expires_at=r.get("expires_at", ""),
            ))
    except Exception as e:
        logger.warning("Failed to list running models: %s", e)

    return OllamaStatus(
        online=True,
        version=version,
        url=base_url,
        models=models,
        running=running,
    )


def execute_command(
    command: str,
    model: str,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = 120.0,
) -> dict:
    """Execute a command on the local Ollama instance.

    Supported commands:
        pull    — Download a model (ollama pull <model>)
        delete  — Remove a model (ollama delete <model>)

    Returns {"status": "ok"} or {"status": "error", "error": "..."}.
    """
    # P2-01: SSRF guard
    try:
        _validate_url(base_url)
    except ValueError as e:
        return {"status": "error", "error": str(e)}

    if command == "pull":
        url = f"{base_url}/api/pull"
        body = json.dumps({"name": model, "stream": False}).encode("utf-8")
    elif command == "delete":
        url = f"{base_url}/api/delete"
        body = json.dumps({"name": model}).encode("utf-8")
    else:
        return {"status": "error", "error": f"Unknown command: {command}"}

    try:
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST" if command == "pull" else "DELETE",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return {"status": "ok", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def eject_all(
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = 10.0,
) -> dict:
    """Unload ALL running models from VRAM immediately.

    Ollama has no single 'unload-all' endpoint. The documented mechanism
    is: POST /api/generate with {"model": "<name>", "keep_alive": 0}
    for each running model. This triggers immediate VRAM release.

    Algorithm:
        1. GET /api/ps → list all running models
        2. For each: POST /api/generate with keep_alive=0
        3. Collect per-model results with error isolation

    Returns {
        "status": "ok" | "partial" | "error",
        "ejected": ["model1", "model2"],
        "failed": [{"model": "x", "error": "..."}],
        "running_before": 3,
        "running_after": 0,
    }

    Timeout is per-model (not total), so worst case with N models
    is N × timeout. Acceptable: Ollama unload is sub-second per model.
    """
    # SSRF guard
    try:
        _validate_url(base_url)
    except ValueError as e:
        return {"status": "error", "error": str(e),
                "ejected": [], "failed": [], "running_before": 0, "running_after": 0}

    # Step 1: Discover running models
    try:
        ps_resp = _http_get(f"{base_url}/api/ps", timeout=timeout)
    except Exception as e:
        return {"status": "error", "error": f"Failed to list running models: {e}",
                "ejected": [], "failed": [], "running_before": 0, "running_after": 0}

    running_models = ps_resp.get("models", [])
    running_before = len(running_models)

    if running_before == 0:
        return {"status": "ok", "ejected": [], "failed": [],
                "running_before": 0, "running_after": 0}

    # Step 2: Unload each model individually (error-isolated)
    ejected: list[str] = []
    failed: list[dict] = []

    for model_info in running_models:
        model_name = model_info.get("name", "")
        if not model_name:
            continue

        try:
            # POST /api/generate with keep_alive=0 triggers immediate unload.
            # No prompt required — Ollama processes the keep_alive directive
            # and returns an empty response after releasing VRAM.
            body = json.dumps({
                "model": model_name,
                "keep_alive": 0,
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url}/api/generate",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp.read()  # Consume response body to release connection
            ejected.append(model_name)
            logger.info("Ejected model from VRAM: %s", model_name)
        except Exception as e:
            failed.append({"model": model_name, "error": str(e)})
            logger.warning("Failed to eject %s: %s", model_name, e)

    # Step 3: Verify — re-check /api/ps for remaining models
    running_after = running_before  # pessimistic default
    try:
        verify_resp = _http_get(f"{base_url}/api/ps", timeout=timeout)
        running_after = len(verify_resp.get("models", []))
    except Exception:
        pass  # Non-fatal: verification is best-effort

    if failed:
        status = "partial" if ejected else "error"
    else:
        status = "ok"

    return {
        "status": status,
        "ejected": ejected,
        "failed": failed,
        "running_before": running_before,
        "running_after": running_after,
    }


def execute_pull_streaming(
    model: str,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = 600.0,
):
    """Pull a model with streaming progress (generator).

    Ollama /api/pull with stream=true returns NDJSON lines:
        {"status": "pulling manifest"}
        {"status": "downloading ...", "digest": "sha256:...",
         "total": 4109895168, "completed": 1234567}
        {"status": "verifying sha256 digest"}
        {"status": "success"}

    Yields dict for each progress event. Raises on error.
    Timeout is 10 min (large models can be 40+ GB).
    """
    try:
        _validate_url(base_url)
    except ValueError as e:
        yield {"status": "error", "error": str(e)}
        return

    url = f"{base_url}/api/pull"
    body = json.dumps({"name": model, "stream": True}).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    yield event
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        yield {"status": "error", "error": str(e)}

