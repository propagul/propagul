from __future__ import annotations

"""propagul.mesh.proxy — Local OpenAI-compatible Reverse Proxy.

Provides a unified OpenAI-compatible endpoint (localhost:8787/v1) that
routes requests to whichever local inference backend is detected:

    Ollama   → Translates /v1/chat/completions → /api/chat (+ streaming)
    vLLM     → Passthrough (already OpenAI-compatible)
    TGI      → Passthrough /v1/chat/completions (TGI ≥1.4 supports it)
    LM Studio → Passthrough (already OpenAI-compatible)
    llama.cpp → Passthrough (server mode supports /v1/)

Architecture:
    - asyncio TCP server (no external deps)
    - HTTP/1.1 request parsing + response forwarding
    - SSE stream translation for Ollama (NDJSON → SSE data: lines)
    - SSRF protection: backend_url validated against localhost/RFC1918 allowlist

Zero external dependencies. Pure stdlib.
"""

import asyncio
import json
import logging
import threading
import time
import urllib.request
import urllib.error
from typing import Optional
from urllib.parse import urlparse

import ipaddress as _ipaddress

logger = logging.getLogger("propagul.mesh.proxy")

DEFAULT_PROXY_PORT = 8787
DEFAULT_PROXY_HOST = "127.0.0.1"

# Backends that already speak OpenAI API natively
_OPENAI_NATIVE_BACKENDS = frozenset({"vllm", "lm_studio", "tgi", "llama_cpp", "localai"})


# Max request body size: 10 MB (prevents memory DoS)
_MAX_BODY_SIZE = 10 * 1024 * 1024

# Streaming backpressure: max chunks queued before reader thread blocks
_STREAM_QUEUE_MAXSIZE = 64

# Sentinel object for end-of-stream signaling between reader thread and async consumer
_STREAM_EOF = object()

# SSRF: only these hostnames are allowed as-is (non-IP)
_SSRF_ALLOWED_HOSTNAMES = frozenset({"localhost"})

# ---------------------------------------------------------------------------
# Thinking Model Budget Inflation
# ---------------------------------------------------------------------------
# Ollama's `num_predict` controls TOTAL tokens (thinking + visible content).
# OpenAI's `max_tokens` controls only VISIBLE output.
# For reasoning models (qwen3, deepseek-r1, QwQ), a low max_tokens means
# all tokens are consumed by reasoning, producing empty visible output.
#
# Fix: Detect thinking models and add a budget to num_predict.
# Default: 4096 extra tokens for thinking (conservative, configurable).
_DEFAULT_THINKING_BUDGET = 4096

# Low max_tokens threshold: for unknown models with max_tokens below this,
# we omit num_predict entirely and let Ollama use its generous default.
# This prevents empty output on first request to an unknown thinking model.
_LOW_MAX_TOKENS_THRESHOLD = 200

# Module-level cache: model names known to support thinking.
# Populated reactively from response `thinking` field.
# Thread-safe: set.add() is atomic in CPython (GIL).
_thinking_models: set = set()


def _validate_backend_url(url: str) -> None:
    """Validate that backend_url points to a local/private address.

    Uses the stdlib `ipaddress` module for robust validation —
    prevents SSRF bypasses like "fd-example.com" that would pass
    naive prefix checks.

    Allowed:
        - "localhost" (hostname)
        - Any loopback IP (127.0.0.0/8, ::1)
        - Any private IP (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, fc00::/7)
        - Tailscale/CGNAT (100.64.0.0/10) — Python <3.11 doesn't classify
          this as private, but Tailscale uses it for mesh networking.

    Blocked explicitly:
        - Link-local (169.254.0.0/16, fe80::/10) — includes cloud metadata

    Raises ValueError if the URL points to a non-local address.
    """
    if not url:
        return  # Empty = no backend, will return 503
    parsed = urlparse(url)
    host = parsed.hostname or ""

    # Allow known safe hostnames
    if host in _SSRF_ALLOWED_HOSTNAMES:
        return

    # Parse as IP address — reject hostnames that aren't "localhost"
    try:
        addr = _ipaddress.ip_address(host)
    except ValueError:
        # Not a valid IP and not in allowed hostnames
        raise ValueError(
            f"SSRF blocked: '{host}' is not a recognized local hostname or IP. "
            f"Only localhost and private/loopback IPs are allowed."
        )

    # Allow loopback (127.0.0.0/8, ::1) and private (RFC1918, ULA fc00::/7)
    if addr.is_loopback or addr.is_private:
        return

    # Block link-local (169.254.0.0/16, fe80::/10) — cloud metadata
    # endpoint 169.254.169.254 must not be reachable. No legitimate
    # use-case for fleet backends on link-local IPs.
    if addr.is_link_local:
        raise ValueError(
            f"SSRF blocked: {host} is a link-local address. "
            f"Cloud metadata endpoint 169.254.169.254 is explicitly blocked."
        )

    # Allow Tailscale/CGNAT range (100.64.0.0/10, RFC 6598).
    # Python <3.11 does NOT classify this as is_private.
    # Tailscale assigns all nodes IPs in 100.64.0.0/10 for mesh routing.
    if isinstance(addr, _ipaddress.IPv4Address):
        _CGNAT = _ipaddress.IPv4Network("100.64.0.0/10")
        if addr in _CGNAT:
            return

    raise ValueError(
        f"SSRF blocked: {host} is not a local/private address. "
        f"Only localhost, loopback, RFC1918/ULA, and Tailscale (100.64/10) IPs are allowed."
    )


class ProxyConfig:
    """Configuration for the local proxy.

    Args:
        host: Bind address (default 127.0.0.1).
        port: Listen port (default 8787).
        backend_name: Backend type ("ollama", "vllm", etc.).
        backend_url: Backend base URL.
        backend_auth: Optional auth header value for the backend
            (e.g. "Bearer sk-lm-..."). This is NOT client pass-through.
            The proxy uses this fixed value for all backend requests.
            Useful for LM Studio or other backends that require auth.
        thinking_budget: Extra tokens added to num_predict for thinking
            models (qwen3, deepseek-r1, etc.). Default: 4096.
            Set to 0 to disable budget inflation.
    """

    def __init__(
        self,
        host: str = DEFAULT_PROXY_HOST,
        port: int = DEFAULT_PROXY_PORT,
        backend_name: str = "",
        backend_url: str = "",
        backend_auth: str = "",
        thinking_budget: int = _DEFAULT_THINKING_BUDGET,
    ):
        _validate_backend_url(backend_url)
        self.host = host
        self.port = port
        self.backend_name = backend_name
        self.backend_url = backend_url.rstrip("/")
        self.backend_auth = backend_auth  # Fixed auth for backend requests
        self.thinking_budget = thinking_budget


# ---------------------------------------------------------------------------
# HTTP Request Parsing (stdlib, no deps)
# ---------------------------------------------------------------------------

class HttpRequest:
    """Minimal parsed HTTP request."""

    def __init__(self):
        self.method: str = ""
        self.path: str = ""
        self.headers: dict[str, str] = {}
        self.body: bytes = b""
        self.http_version: str = "HTTP/1.1"


async def _read_http_request(reader: asyncio.StreamReader) -> Optional[HttpRequest]:
    """Parse an HTTP/1.1 request from an asyncio StreamReader.

    Returns None on connection close or parse failure.
    """
    req = HttpRequest()

    # Read request line
    try:
        request_line = await asyncio.wait_for(reader.readline(), timeout=30.0)
    except (asyncio.TimeoutError, ConnectionResetError):
        return None

    if not request_line:
        return None

    request_line_str = request_line.decode("utf-8", errors="replace").strip()
    if not request_line_str:
        return None

    parts = request_line_str.split(" ", 2)
    if len(parts) < 2:
        return None

    req.method = parts[0].upper()
    req.path = parts[1]
    if len(parts) > 2:
        req.http_version = parts[2]

    # Read headers
    while True:
        line = await asyncio.wait_for(reader.readline(), timeout=10.0)
        line_str = line.decode("utf-8", errors="replace").strip()
        if not line_str:
            break
        if ":" in line_str:
            key, value = line_str.split(":", 1)
            req.headers[key.strip().lower()] = value.strip()

    # Read body if Content-Length present
    content_length = req.headers.get("content-length")
    if content_length and content_length.isdigit():
        length = int(content_length)
        if length > _MAX_BODY_SIZE:
            return None  # Body too large, reject
        if length > 0:
            req.body = await asyncio.wait_for(
                reader.readexactly(length), timeout=30.0,
            )

    # Reject chunked transfer encoding (not supported)
    if req.headers.get("transfer-encoding", "").lower() == "chunked":
        return None

    return req


def _write_http_response(
    status: int,
    body: bytes,
    content_type: str = "application/json",
    extra_headers: Optional[dict[str, str]] = None,
) -> bytes:
    """Build a complete HTTP/1.1 response as bytes."""
    status_text = {
        200: "OK", 400: "Bad Request", 404: "Not Found",
        405: "Method Not Allowed", 413: "Content Too Large",
        500: "Internal Server Error", 501: "Not Implemented",
        502: "Bad Gateway", 503: "Service Unavailable",
    }.get(status, "Unknown")

    headers = {
        "Content-Type": content_type,
        "Content-Length": str(len(body)),
        "Connection": "close",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    }
    if extra_headers:
        headers.update(extra_headers)

    header_lines = f"HTTP/1.1 {status} {status_text}\r\n"
    for k, v in headers.items():
        header_lines += f"{k}: {v}\r\n"
    header_lines += "\r\n"

    return header_lines.encode("utf-8") + body


def _json_error(status: int, message: str) -> bytes:
    """Build an OpenAI-style error JSON response."""
    body = json.dumps({
        "error": {
            "message": message,
            "type": "proxy_error",
            "code": status,
        }
    }).encode("utf-8")
    return _write_http_response(status, body)


# ---------------------------------------------------------------------------
# Ollama → OpenAI Translation
# ---------------------------------------------------------------------------

def _translate_openai_to_ollama_chat(
    openai_body: dict,
    thinking_budget: int = _DEFAULT_THINKING_BUDGET,
) -> dict:
    """Translate OpenAI /v1/chat/completions request → Ollama /api/chat.

    OpenAI format:
        {"model": "llama3.1:8b", "messages": [...], "stream": true, ...}

    Ollama format:
        {"model": "llama3.1:8b", "messages": [...], "stream": true, ...}

    Ollama's /api/chat format is very close to OpenAI's. Key differences:
    - Response format differs (Ollama wraps in {"message": {...}})
    - Streaming format differs (NDJSON vs SSE)
    - Some parameters have different names

    Thinking model budget inflation:
        OpenAI's ``max_tokens`` controls only visible output.
        Ollama's ``num_predict`` controls total tokens (thinking + output).
        For known thinking models, we inflate ``num_predict`` by
        ``thinking_budget`` so the model has room to reason AND produce
        visible output. For unknown models with low max_tokens,
        we omit ``num_predict`` entirely to let Ollama use its default.
    """
    model = openai_body.get("model", "")
    ollama_req = {
        "model": model,
        "messages": openai_body.get("messages", []),
        "stream": openai_body.get("stream", False),
    }

    # Forward compatible parameters
    if "temperature" in openai_body:
        ollama_req["options"] = ollama_req.get("options", {})
        ollama_req["options"]["temperature"] = openai_body["temperature"]
    if "top_p" in openai_body:
        ollama_req["options"] = ollama_req.get("options", {})
        ollama_req["options"]["top_p"] = openai_body["top_p"]
    if "max_tokens" in openai_body:
        max_tokens = openai_body["max_tokens"]
        num_predict = _compute_num_predict(model, max_tokens, thinking_budget)
        if num_predict is not None:
            ollama_req["options"] = ollama_req.get("options", {})
            ollama_req["options"]["num_predict"] = num_predict
        # else: omit num_predict — let Ollama use its default
    if "stop" in openai_body:
        ollama_req["options"] = ollama_req.get("options", {})
        ollama_req["options"]["stop"] = openai_body["stop"]

    return ollama_req


def _compute_num_predict(
    model: str,
    max_tokens: int,
    thinking_budget: int,
) -> "Optional[int]":
    """Compute num_predict from max_tokens with thinking model awareness.

    Returns:
        int: The adjusted num_predict value.
        None: Omit num_predict (let Ollama use its default).

    Strategy:
        - Known thinking model → max_tokens + thinking_budget
        - Known non-thinking model → max_tokens (1:1)
        - Unknown model + low max_tokens → None (safety net)
        - Unknown model + high max_tokens → max_tokens (1:1, safe)
    """
    if model in _thinking_models:
        # Known thinking model: inflate budget
        inflated = max_tokens + thinking_budget
        logger.debug(
            "Thinking model '%s': num_predict=%d (max_tokens=%d + budget=%d)",
            model, inflated, max_tokens, thinking_budget,
        )
        return inflated

    # Unknown model: if max_tokens is low, omit num_predict entirely.
    # This prevents empty output on first request to an unknown thinking model.
    # For non-thinking models, Ollama stops at natural completion anyway.
    if max_tokens < _LOW_MAX_TOKENS_THRESHOLD:
        logger.debug(
            "Unknown model '%s' with low max_tokens=%d: omitting num_predict "
            "(safety net for potential thinking model)",
            model, max_tokens,
        )
        return None

    # Unknown model with generous max_tokens: pass through as-is.
    # Even if it's a thinking model, there's likely enough budget.
    return max_tokens


def _register_thinking_model(model: str) -> None:
    """Register a model as a thinking model (thread-safe via GIL)."""
    if model and model not in _thinking_models:
        _thinking_models.add(model)
        logger.info("Registered thinking model: '%s'", model)


def _translate_ollama_response_to_openai(
    ollama_resp: dict,
    model: str,
) -> dict:
    """Translate Ollama /api/chat response → OpenAI /v1/chat/completions.

    Ollama returns:
        {"message": {"role": "assistant", "content": "...",
                     "thinking": "..." (optional, reasoning models)},
         "done": true, "total_duration": ..., "eval_count": ..., ...}

    OpenAI expects:
        {"id": "...", "object": "chat.completion", "model": "...",
         "choices": [{"message": {"role": "assistant", "content": "..."},
                      "finish_reason": "stop", "index": 0}],
         "usage": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}}

    Thinking models (qwen3, deepseek-r1, etc.):
        Ollama v0.24+ returns a separate "thinking" field in the message.
        We forward it as-is — clients that understand it can use it,
        others simply ignore the extra field (non-breaking extension).
    """
    message = ollama_resp.get("message", {})
    eval_count = ollama_resp.get("eval_count", 0)
    prompt_eval_count = ollama_resp.get("prompt_eval_count", 0)

    # Build message dict — include thinking if present
    msg_dict: dict = {
        "role": message.get("role", "assistant"),
        "content": message.get("content", ""),
    }
    if "thinking" in message and message["thinking"]:
        msg_dict["thinking"] = message["thinking"]
        _register_thinking_model(model)  # Cache for future budget inflation

    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": msg_dict,
            "finish_reason": "stop" if ollama_resp.get("done") else None,
        }],
        "usage": {
            "prompt_tokens": prompt_eval_count,
            "completion_tokens": eval_count,
            "total_tokens": prompt_eval_count + eval_count,
        },
    }


def _translate_ollama_stream_chunk_to_openai(
    ollama_chunk: dict,
    model: str,
    stream_id: str = "",
) -> str:
    """Translate a single Ollama stream NDJSON line → OpenAI SSE data: line.

    Ollama streams:
        {"message": {"role": "assistant", "content": "tok"}, "done": false}
        ...
        {"message": {"role": "assistant", "content": ""}, "done": true}

    Thinking models additionally stream:
        {"message": {"content": "", "thinking": "tok"}, "done": false}
    during the reasoning phase (content is empty, thinking has tokens).

    OpenAI expects:
        data: {"id": "...", "choices": [{"delta": {"content": "tok"}}]}
        ...
        data: [DONE]

    stream_id: Stable ID shared across all chunks in a stream.
    """
    if ollama_chunk.get("done"):
        return "data: [DONE]\n\n"

    chunk_id = stream_id or f"chatcmpl-{int(time.time()*1000)}"
    message = ollama_chunk.get("message", {})
    delta: dict = {
        "content": message.get("content", ""),
    }
    # Forward thinking tokens for reasoning models
    thinking = message.get("thinking")
    if thinking:
        delta["thinking"] = thinking
        _register_thinking_model(model)  # Cache for future budget inflation

    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": None,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _translate_ollama_models_to_openai(ollama_models: dict) -> dict:
    """Translate Ollama /api/tags → OpenAI /v1/models.

    Ollama returns: {"models": [{"name": "llama3:8b", ...}, ...]}
    OpenAI expects: {"data": [{"id": "llama3:8b", "object": "model", ...}], "object": "list"}
    """
    models = []
    for m in ollama_models.get("models", []):
        models.append({
            "id": m.get("name", "unknown"),
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        })
    return {"object": "list", "data": models}


# ---------------------------------------------------------------------------
# Backend Routing
# ---------------------------------------------------------------------------

class BackendRouter:
    """Routes OpenAI API requests to the correct local backend.

    Supports two modes:
    - Local-only: routes to the single configured backend (backward compat)
    - Fleet-aware: uses FleetState to route to the best backend in the fleet
      (model-affinity + least-connections + automatic failover)
    """

    def __init__(self, config: ProxyConfig, fleet_state=None, local_node_id: str = "", routing_log=None):
        self.config = config
        self._fleet_router = None
        self._routing_log = routing_log
        self._local_node_id = local_node_id
        if fleet_state is not None:
            from propagul.mesh.router import RequestRouter, ActiveConnectionTracker
            self._tracker = ActiveConnectionTracker()
            self._fleet_router = RequestRouter(
                fleet_state=fleet_state,
                local_node_id=local_node_id,
                tracker=self._tracker,
            )
        else:
            self._tracker = None

    def _backend_url(self) -> str:
        return self.config.backend_url

    def _backend_auth(self) -> str:
        """Return the configured backend auth header value (or empty)."""
        return self.config.backend_auth

    def _is_ollama(self) -> bool:
        return self.config.backend_name == "ollama"

    def _is_native_openai(self) -> bool:
        return self.config.backend_name in _OPENAI_NATIVE_BACKENDS

    async def handle_request(
        self,
        req: HttpRequest,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Route an incoming OpenAI API request to the backend."""
        path = req.path.split("?")[0]  # Strip query params

        # CORS preflight
        if req.method == "OPTIONS":
            writer.write(_write_http_response(200, b""))
            await writer.drain()
            return

        # Route based on path
        if path == "/v1/models" or path == "/v1/models/":
            await self._handle_models(req, writer)
        elif path == "/v1/chat/completions":
            await self._handle_chat_completions(req, writer)
        elif path == "/v1/completions":
            await self._handle_completions(req, writer)
        elif path == "/health" or path == "/":
            await self._handle_health(req, writer)
        else:
            writer.write(_json_error(404, f"Unknown endpoint: {path}"))
            await writer.drain()

    async def _handle_health(
        self, req: HttpRequest, writer: asyncio.StreamWriter,
    ) -> None:
        """Health endpoint for the proxy itself."""
        body = json.dumps({
            "status": "ok",
            "proxy": "propagul-mesh",
            "backend": self.config.backend_name,
            "backend_url": self.config.backend_url,
        }).encode("utf-8")
        writer.write(_write_http_response(200, body))
        await writer.drain()

    async def _handle_models(
        self, req: HttpRequest, writer: asyncio.StreamWriter,
    ) -> None:
        """GET /v1/models — list available models."""
        if req.method not in ("GET", "HEAD"):
            writer.write(_json_error(405, "GET required for /v1/models"))
            await writer.drain()
            return

        if not self._backend_url():
            writer.write(_json_error(503, "No backend detected"))
            await writer.drain()
            return

        loop = asyncio.get_running_loop()
        auth = self._backend_auth()
        if self._is_ollama():
            # Translate Ollama /api/tags → OpenAI /v1/models
            try:
                data = await loop.run_in_executor(
                    None,
                    lambda: _http_get_sync(
                        f"{self._backend_url()}/api/tags", auth_header=auth,
                    ),
                )
                openai_resp = _translate_ollama_models_to_openai(data)
                body = json.dumps(openai_resp).encode("utf-8")
                writer.write(_write_http_response(200, body))
            except Exception as e:
                writer.write(_json_error(502, f"Backend error: {e}"))
        else:
            # Native OpenAI backend — passthrough
            try:
                data = await loop.run_in_executor(
                    None,
                    lambda: _http_get_sync(
                        f"{self._backend_url()}/v1/models", auth_header=auth,
                    ),
                )
                body = json.dumps(data).encode("utf-8")
                writer.write(_write_http_response(200, body))
            except Exception as e:
                writer.write(_json_error(502, f"Backend error: {e}"))

        await writer.drain()

    async def _handle_chat_completions(
        self, req: HttpRequest, writer: asyncio.StreamWriter,
    ) -> None:
        """POST /v1/chat/completions — the main inference endpoint.

        Routing priority:
        1. Fleet-aware: if FleetState is available and has a better backend
           for the requested model, route directly to that remote backend.
        2. Local fallback: use the configured local backend.
        """
        if req.method != "POST":
            writer.write(_json_error(400, "POST required"))
            await writer.drain()
            return

        try:
            openai_body = json.loads(req.body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            writer.write(_json_error(400, f"Invalid JSON: {e}"))
            await writer.drain()
            return

        stream = openai_body.get("stream", False)
        model = openai_body.get("model", "")

        # Fleet-aware routing: check if a remote backend is better
        if self._fleet_router and model:
            target = self._fleet_router.select(model)
            if target is not None:
                # SSRF protection: validate remote backend URL.
                # Even though fleet_routing comes from our dashboard,
                # a compromised agent could inject a public URL.
                try:
                    _validate_backend_url(target.backend_url)
                except ValueError as ssrf_err:
                    logger.warning(
                        "Fleet target %s blocked by SSRF: %s",
                        target.node_id, ssrf_err,
                    )
                    target = None  # Fall through to local backend
            if target is not None:
                # Route to the fleet target
                self._tracker.acquire(target.node_id)
                try:
                    backend_name = target.backend_name
                    backend_url = target.backend_url.rstrip("/")

                    if backend_name == "ollama":
                        if stream:
                            await self._ollama_chat_stream(
                                openai_body, model, writer,
                                backend_url_override=backend_url,
                            )
                        else:
                            await self._ollama_chat_sync(
                                openai_body, model, writer,
                                backend_url_override=backend_url,
                            )
                    elif backend_name in _OPENAI_NATIVE_BACKENDS:
                        await self._passthrough_post(
                            f"{backend_url}/v1/chat/completions",
                            req.body, stream, writer,
                            auth_override=target.backend_auth,
                        )
                    else:
                        # Unknown backend type from fleet — try passthrough
                        await self._passthrough_post(
                            f"{backend_url}/v1/chat/completions",
                            req.body, stream, writer,
                            auth_override=target.backend_auth,
                        )

                    logger.info(
                        "Fleet-routed: model=%s → node=%s (%s)",
                        model, target.node_id, backend_name,
                    )
                    if self._routing_log:
                        from propagul.mesh.router import RoutingEvent
                        self._routing_log.record(RoutingEvent(
                            timestamp=time.time(),
                            model=model,
                            target_node=target.node_id,
                            target_backend=backend_name,
                            reason="model_affinity",
                        ))
                    return
                except Exception as e:
                    logger.warning(
                        "Fleet routing to %s failed: %s. Falling back to local.",
                        target.node_id, e,
                    )
                    if self._routing_log:
                        from propagul.mesh.router import RoutingEvent
                        self._routing_log.record(RoutingEvent(
                            timestamp=time.time(),
                            model=model,
                            target_node=target.node_id,
                            target_backend=backend_name,
                            reason="fleet_error",
                        ))
                finally:
                    self._tracker.release(target.node_id)

        # Local fallback (original behavior)
        if not self._backend_url():
            writer.write(_json_error(503, "No backend detected"))
            await writer.drain()
            return

        if self._is_ollama():
            if stream:
                await self._ollama_chat_stream(openai_body, model, writer)
            else:
                await self._ollama_chat_sync(openai_body, model, writer)
        elif self._is_native_openai():
            # Passthrough to OpenAI-compatible backend
            await self._passthrough_post(
                f"{self._backend_url()}/v1/chat/completions",
                req.body, stream, writer,
            )
        else:
            writer.write(_json_error(501, f"Unsupported backend: {self.config.backend_name}"))
            await writer.drain()
            return

        # Record local fallback event
        if self._routing_log and model:
            from propagul.mesh.router import RoutingEvent
            self._routing_log.record(RoutingEvent(
                timestamp=time.time(),
                model=model,
                target_node=self._local_node_id or "local",
                target_backend=self.config.backend_name or "unknown",
                reason="local_fallback",
            ))

    async def _handle_completions(
        self, req: HttpRequest, writer: asyncio.StreamWriter,
    ) -> None:
        """POST /v1/completions — text completions (less common)."""
        if req.method != "POST":
            writer.write(_json_error(405, "POST required for /v1/completions"))
            await writer.drain()
            return

        if not self._backend_url():
            writer.write(_json_error(503, "No backend detected"))
            await writer.drain()
            return

        if self._is_native_openai():
            stream = False
            try:
                body = json.loads(req.body.decode("utf-8"))
                stream = body.get("stream", False)
            except Exception:
                pass
            await self._passthrough_post(
                f"{self._backend_url()}/v1/completions",
                req.body, stream, writer,
            )
        else:
            writer.write(_json_error(501, "Text completions not supported for this backend"))
            await writer.drain()

    # -------------------------------------------------------------------
    # Ollama-specific handlers
    # -------------------------------------------------------------------

    async def _ollama_chat_sync(
        self,
        openai_body: dict,
        model: str,
        writer: asyncio.StreamWriter,
        backend_url_override: str = "",
    ) -> None:
        """Non-streaming Ollama /api/chat request.

        Runs the blocking HTTP call in a thread executor to avoid
        blocking the asyncio event loop.
        """
        ollama_req = _translate_openai_to_ollama_chat(openai_body, self.config.thinking_budget)
        ollama_req["stream"] = False

        auth = self._backend_auth()
        backend_url = backend_url_override or self._backend_url()
        try:
            loop = asyncio.get_running_loop()
            req_body = json.dumps(ollama_req).encode("utf-8")
            ollama_resp = await loop.run_in_executor(
                None,
                lambda: _http_post_sync(
                    f"{backend_url}/api/chat", req_body,
                    timeout=300.0, auth_header=auth,
                ),
            )
            openai_resp = _translate_ollama_response_to_openai(ollama_resp, model)
            body = json.dumps(openai_resp).encode("utf-8")
            writer.write(_write_http_response(200, body))
        except Exception as e:
            writer.write(_json_error(502, f"Ollama error: {e}"))

        await writer.drain()

    async def _ollama_chat_stream(
        self,
        openai_body: dict,
        model: str,
        writer: asyncio.StreamWriter,
        backend_url_override: str = "",
    ) -> None:
        """Streaming Ollama /api/chat → OpenAI SSE translation.

        Ollama streams NDJSON lines. We translate each to SSE data: lines
        in OpenAI format and flush immediately for low latency.

        Uses a single reader thread (via run_in_executor) that reads all
        lines and pushes them to an asyncio.Queue with backpressure.
        The async consumer awaits queue.get() — zero per-line executor calls.
        """
        ollama_req = _translate_openai_to_ollama_chat(openai_body, self.config.thinking_budget)
        ollama_req["stream"] = True

        # Stable stream ID (all chunks share same ID per OpenAI spec)
        stream_id = f"chatcmpl-{int(time.time()*1000)}"

        # Write SSE response headers
        sse_headers = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n"
        )
        writer.write(sse_headers.encode("utf-8"))
        await writer.drain()

        loop = asyncio.get_running_loop()
        auth = self._backend_auth()
        backend_url = backend_url_override or self._backend_url()
        try:
            req_body = json.dumps(ollama_req).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if auth:
                headers["Authorization"] = auth
            http_req = urllib.request.Request(
                f"{backend_url}/api/chat",
                data=req_body,
                headers=headers,
                method="POST",
            )

            # Open connection in executor (blocking)
            resp = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(http_req, timeout=300),
            )

            # P-03: Ensure resp is always closed, even if queue/reader setup fails
            reader_future = None
            try:
                # Single-thread stream: reader thread → asyncio.Queue → async consumer
                q: asyncio.Queue = asyncio.Queue(maxsize=_STREAM_QUEUE_MAXSIZE)
                stop_evt = threading.Event()

                def _reader():
                    """Blocking reader in executor thread. Pushes lines to async queue."""
                    try:
                        for raw in iter(resp.readline, b""):
                            if stop_evt.is_set():
                                break
                            # Backpressure: blocks here if queue is full
                            fut = asyncio.run_coroutine_threadsafe(q.put(raw), loop)
                            fut.result()  # propagates RuntimeError if loop is closed
                    except Exception as exc:
                        try:
                            fut = asyncio.run_coroutine_threadsafe(q.put(exc), loop)
                            fut.result()
                        except RuntimeError:
                            pass
                    finally:
                        try:
                            fut = asyncio.run_coroutine_threadsafe(q.put(_STREAM_EOF), loop)
                            fut.result()
                        except RuntimeError:
                            pass

                reader_future = loop.run_in_executor(None, _reader)

                while True:
                    item = await q.get()
                    if item is _STREAM_EOF:
                        break
                    if isinstance(item, Exception):
                        raise item

                    line = item.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        sse_data = _translate_ollama_stream_chunk_to_openai(
                            chunk, model, stream_id=stream_id,
                        )
                        writer.write(sse_data.encode("utf-8"))
                        await writer.drain()
                    except json.JSONDecodeError:
                        continue
            except asyncio.CancelledError:
                stop_evt.set()
                raise
            finally:
                stop_evt.set()
                try:
                    resp.close()
                except Exception:
                    pass
                # Wait for reader thread to exit cleanly
                if reader_future is not None:
                    try:
                        await asyncio.shield(reader_future)
                    except Exception:
                        pass

        except Exception as e:
            # Send error event + [DONE] so clients don't hang
            error_event = f"data: {json.dumps({'error': str(e)})}\n\n"
            writer.write(error_event.encode("utf-8"))
            writer.write(b"data: [DONE]\n\n")
            await writer.drain()

    # -------------------------------------------------------------------
    # Passthrough for OpenAI-native backends (vLLM, LM Studio, etc.)
    # -------------------------------------------------------------------

    async def _passthrough_post(
        self,
        url: str,
        body: bytes,
        stream: bool,
        writer: asyncio.StreamWriter,
        auth_override: str = "",
    ) -> None:
        """Forward a POST request directly to an OpenAI-native backend."""
        auth = auth_override or self._backend_auth()
        try:
            if stream:
                await self._passthrough_stream(url, body, writer, auth_override=auth)
            else:
                loop = asyncio.get_running_loop()
                resp = await loop.run_in_executor(
                    None,
                    lambda: _http_post_sync(
                        url, body, timeout=300.0, auth_header=auth,
                    ),
                )
                resp_body = json.dumps(resp).encode("utf-8")
                writer.write(_write_http_response(200, resp_body))
                await writer.drain()
        except Exception as e:
            writer.write(_json_error(502, f"Backend error: {e}"))
            await writer.drain()

    async def _passthrough_stream(
        self,
        url: str,
        body: bytes,
        writer: asyncio.StreamWriter,
        auth_override: str = "",
    ) -> None:
        """Stream passthrough for OpenAI-native backends.

        The backend already emits SSE data: lines — we just forward them.
        Uses a single reader thread with asyncio.Queue for backpressure
        instead of per-line run_in_executor calls.
        """
        sse_headers = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n"
        )
        writer.write(sse_headers.encode("utf-8"))
        await writer.drain()

        loop = asyncio.get_running_loop()
        auth = auth_override or self._backend_auth()
        try:
            headers = {"Content-Type": "application/json"}
            if auth:
                headers["Authorization"] = auth
            http_req = urllib.request.Request(
                url,
                data=body,
                headers=headers,
                method="POST",
            )

            resp = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(http_req, timeout=300),
            )

            # Single-thread stream: reader thread → asyncio.Queue → async consumer
            q: asyncio.Queue = asyncio.Queue(maxsize=_STREAM_QUEUE_MAXSIZE)
            stop_evt = threading.Event()

            def _reader():
                """Blocking reader in executor thread. Pushes raw lines to async queue."""
                try:
                    for raw in iter(resp.readline, b""):
                        if stop_evt.is_set():
                            break
                        fut = asyncio.run_coroutine_threadsafe(q.put(raw), loop)
                        fut.result()
                except Exception as exc:
                    try:
                        fut = asyncio.run_coroutine_threadsafe(q.put(exc), loop)
                        fut.result()
                    except RuntimeError:
                        pass
                finally:
                    try:
                        fut = asyncio.run_coroutine_threadsafe(q.put(_STREAM_EOF), loop)
                        fut.result()
                    except RuntimeError:
                        pass

            reader_future = loop.run_in_executor(None, _reader)

            try:
                while True:
                    item = await q.get()
                    if item is _STREAM_EOF:
                        break
                    if isinstance(item, Exception):
                        raise item
                    writer.write(item)
                    await writer.drain()
            except asyncio.CancelledError:
                stop_evt.set()
                raise
            finally:
                stop_evt.set()
                try:
                    resp.close()
                except Exception:
                    pass
                try:
                    await asyncio.shield(reader_future)
                except Exception:
                    pass

        except Exception as e:
            error_event = f"data: {json.dumps({'error': str(e)})}\n\n"
            writer.write(error_event.encode("utf-8"))
            writer.write(b"data: [DONE]\n\n")
            await writer.drain()


# ---------------------------------------------------------------------------
# Synchronous HTTP helpers (run in executor for asyncio compat)
# ---------------------------------------------------------------------------

def _http_get_sync(
    url: str,
    timeout: float = 10.0,
    auth_header: str = "",
) -> dict:
    """Synchronous HTTP GET → parsed JSON.

    Args:
        auth_header: Optional Authorization header value for the backend.
    """
    headers = {"Accept": "application/json"}
    if auth_header:
        headers["Authorization"] = auth_header
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_sync(
    url: str,
    body: bytes,
    timeout: float = 60.0,
    auth_header: str = "",
) -> dict:
    """Synchronous HTTP POST → parsed JSON.

    Args:
        auth_header: Optional Authorization header value for the backend.
    """
    headers = {"Content-Type": "application/json"}
    if auth_header:
        headers["Authorization"] = auth_header
    req = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Server Lifecycle
# ---------------------------------------------------------------------------

async def _handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    router: BackendRouter,
) -> None:
    """Handle a single HTTP connection."""
    peer = writer.get_extra_info("peername", ("?", 0))
    try:
        req = await _read_http_request(reader)
        if req is None:
            return

        logger.debug("%s %s from %s", req.method, req.path, peer)
        await router.handle_request(req, writer)

    except Exception as e:
        logger.error("Connection error from %s: %s", peer, e)
        try:
            writer.write(_json_error(500, f"Internal proxy error: {e}"))
            await writer.drain()
        except Exception:
            pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def start_proxy(
    config: ProxyConfig,
    ready_event: Optional[asyncio.Event] = None,
    fleet_state=None,
    local_node_id: str = "",
    routing_log=None,
) -> None:
    """Start the local reverse proxy server.

    This is an asyncio coroutine — meant to be run as a task alongside
    the agent's telemetry loops.

    Args:
        config: Proxy configuration (host, port, backend info)
        ready_event: Optional event that gets set when the server is listening.
        fleet_state: Optional FleetState for fleet-aware routing.
            If provided, the proxy can route to remote backends in the LAN.
            If None, the proxy routes only to the local backend (backward compat).
        local_node_id: This node's ID (for local-preference in routing).
        routing_log: Optional RoutingEventLog for recording routing decisions.
    """
    router = BackendRouter(
        config,
        fleet_state=fleet_state,
        local_node_id=local_node_id,
        routing_log=routing_log,
    )

    server = await asyncio.start_server(
        lambda r, w: _handle_connection(r, w, router),
        config.host,
        config.port,
    )

    addr = server.sockets[0].getsockname() if server.sockets else (config.host, config.port)
    fleet_mode = "fleet-aware" if fleet_state is not None else "local-only"
    logger.info(
        "Local proxy listening on http://%s:%d/v1 → %s (%s) [%s]",
        addr[0], addr[1], config.backend_url, config.backend_name, fleet_mode,
    )

    if ready_event:
        ready_event.set()

    async with server:
        await server.serve_forever()
