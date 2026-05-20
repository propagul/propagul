"""propagul.mesh.backends.detect — Auto-detect local inference engines.

Probes common ports to find which inference engines are running locally.
Returns a list of detected backends with their URLs.
"""

import logging
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("propagul.mesh.backends.detect")


@dataclass
class DetectedBackend:
    """A detected local inference engine."""
    name: str  # "ollama", "vllm", "llama_cpp", "tgi", "lm_studio", "localai"
    url: str  # Base URL
    version: str
    confidence: float  # 0.0 - 1.0


# Probe order: most common first, TGI before llama.cpp (both use 8080)
_PROBES = [
    {
        "name": "ollama",
        "urls": ["http://localhost:11434"],
        "health_path": "/api/version",
        "version_key": "version",
        "sig_header": None,
        "sig_body_key": "version",
    },
    {
        "name": "vllm",
        "urls": ["http://localhost:8000"],
        "health_path": "/v1/models",
        "version_key": None,
        "sig_header": None,
        "sig_body_key": "data",  # vLLM returns {"data": [...]}
    },
    {
        # TGI MUST come before llama.cpp — both use port 8080.
        # TGI has /info with "model_id" field; llama.cpp does not.
        "name": "tgi",
        "urls": ["http://localhost:8080"],
        "health_path": "/info",
        "version_key": "version",
        "sig_header": None,
        "sig_body_key": "model_id",  # TGI signature: {"model_id": "..."}
    },
    {
        "name": "llama_cpp",
        "urls": ["http://localhost:8080"],
        "health_path": "/health",
        "version_key": None,
        "sig_header": None,
        "sig_body_key": "status",  # llama.cpp returns {"status": "ok"}
    },
    {
        "name": "lm_studio",
        "urls": ["http://localhost:1234"],
        "health_path": "/v1/models",
        "version_key": None,
        "sig_header": None,
        "sig_body_key": "data",
    },
    {
        "name": "localai",
        "urls": ["http://localhost:8080"],
        "health_path": "/v1/models",
        "version_key": None,
        "sig_header": None,
        "sig_body_key": "data",
    },
]


def _probe_url(url: str, path: str, timeout: float = 2.0) -> Optional[dict]:
    """Try to reach a URL. Returns parsed JSON or None."""
    import json
    try:
        req = urllib.request.Request(
            f"{url}{path}",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def detect(timeout: float = 2.0) -> list[DetectedBackend]:
    """Auto-detect all local inference engines.

    Probes common ports in parallel-ish fashion (sequential but fast with
    short timeouts). Returns list of detected backends.
    """
    detected: list[DetectedBackend] = []
    seen_urls: set[str] = set()

    for probe in _PROBES:
        for url in probe["urls"]:
            if url in seen_urls:
                continue

            resp = _probe_url(url, probe["health_path"], timeout=timeout)
            if resp is None:
                continue

            # Verify it's the right backend by checking signature
            sig_key = probe["sig_body_key"]
            if sig_key and sig_key not in resp:
                continue

            version = ""
            if probe["version_key"] and probe["version_key"] in resp:
                version = str(resp[probe["version_key"]])

            detected.append(DetectedBackend(
                name=probe["name"],
                url=url,
                version=version,
                confidence=0.9 if version else 0.7,
            ))
            seen_urls.add(url)
            logger.info("Detected %s at %s (v%s)", probe["name"], url, version)

    if not detected:
        logger.warning("No local inference engines detected")

    return detected
