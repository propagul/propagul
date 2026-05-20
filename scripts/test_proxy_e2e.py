#!/usr/bin/env python3
"""Propagul Proxy E2E Test — Run locally on machine where Ollama is installed.

Usage:
    python scripts/test_proxy_e2e.py [--ollama-url http://localhost:11434]

This script:
1. Verifies Ollama is reachable
2. Starts the Propagul local proxy on a random port
3. Tests health, models listing, non-streaming chat, streaming chat
4. Prints PASS/FAIL for each test
5. Shuts down the proxy

Requirements: Python 3.9+, Ollama running locally. Zero external dependencies.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import urllib.request
import urllib.error

# Ensure propagul is importable — adjust path if running from repo root
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from propagul.mesh.proxy import ProxyConfig, BackendRouter, start_proxy, _read_http_request


# ---------------------------------------------------------------------------
# Test Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434"
PROXY_HOST = "127.0.0.1"


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Propagul Proxy E2E Test")
    parser.add_argument(
        "--ollama-url", default=OLLAMA_URL,
        help=f"Ollama base URL (default: {OLLAMA_URL})",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: float = 10.0) -> tuple[int, bytes]:
    """Simple GET, returns (status_code, body_bytes)."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        return 0, str(e).encode()


def _http_post(url: str, body: dict, timeout: float = 60.0) -> tuple[int, bytes]:
    """Simple POST, returns (status_code, body_bytes)."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        return 0, str(e).encode()


def _http_post_stream(url: str, body: dict, timeout: float = 60.0) -> list[str]:
    """POST with streaming — collects SSE lines."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    lines = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

results: list[tuple[str, bool, str]] = []


def _report(name: str, passed: bool, detail: str = ""):
    icon = "✅" if passed else "❌"
    results.append((name, passed, detail))
    print(f"  {icon} {name}" + (f" — {detail}" if detail else ""))


async def run_tests(ollama_url: str):
    """Run all E2E tests.

    IMPORTANT: All blocking HTTP calls MUST run in a thread executor.
    The proxy server runs on the SAME event loop — blocking calls would
    prevent the server from processing requests (deadlock).
    """
    loop = asyncio.get_running_loop()

    # Async wrappers for blocking HTTP helpers (prevent event loop deadlock)
    async def aget(url: str, timeout: float = 10.0):
        return await loop.run_in_executor(None, lambda: _http_get(url, timeout))

    async def apost(url: str, body: dict, timeout: float = 60.0):
        return await loop.run_in_executor(None, lambda: _http_post(url, body, timeout))

    async def apost_stream(url: str, body: dict, timeout: float = 60.0):
        return await loop.run_in_executor(None, lambda: _http_post_stream(url, body, timeout))

    # ------------------------------------------------------------------
    # 0. Pre-flight: Ollama reachable?
    # ------------------------------------------------------------------
    print("\n  Pre-flight: Checking Ollama...")
    status, body = await aget(f"{ollama_url}/api/version")
    if status != 200:
        print(f"  ❌ Ollama not reachable at {ollama_url} (status={status})")
        print(f"     Make sure Ollama is running: ollama serve")
        sys.exit(1)

    version = json.loads(body).get("version", "?")
    print(f"  ✅ Ollama v{version} at {ollama_url}\n")

    # Get first available model for chat tests
    _, tags_body = await aget(f"{ollama_url}/api/tags")
    models = json.loads(tags_body).get("models", [])
    if not models:
        print("  ❌ No models installed. Run: ollama pull llama3.2:1b")
        sys.exit(1)

    test_model = models[0]["name"]
    print(f"  Using model: {test_model}\n")

    # ------------------------------------------------------------------
    # 1. Start proxy on random port
    # ------------------------------------------------------------------
    config = ProxyConfig(
        host=PROXY_HOST,
        port=0,  # OS picks a free port
        backend_name="ollama",
        backend_url=ollama_url,
    )
    router = BackendRouter(config)

    async def _handler(r, w):
        try:
            req = await _read_http_request(r)
            if req:
                await router.handle_request(req, w)
        except Exception:
            pass
        finally:
            try:
                w.close()
                await w.wait_closed()
            except Exception:
                pass

    server = await asyncio.start_server(_handler, PROXY_HOST, 0)
    port = server.sockets[0].getsockname()[1]
    base = f"http://{PROXY_HOST}:{port}"
    print(f"  Proxy listening on {base}\n")

    try:
        # ------------------------------------------------------------------
        # Test 1: Health endpoint
        # ------------------------------------------------------------------
        status, body = await aget(f"{base}/health")
        try:
            data = json.loads(body)
            ok = status == 200 and data.get("status") == "ok"
            _report("GET /health", ok, f"status={status}, body={data}")
        except Exception as e:
            _report("GET /health", False, str(e))

        # ------------------------------------------------------------------
        # Test 2: Models listing (OpenAI format)
        # ------------------------------------------------------------------
        status, body = await aget(f"{base}/v1/models")
        try:
            data = json.loads(body)
            ok = (
                status == 200
                and data.get("object") == "list"
                and len(data.get("data", [])) > 0
                and all(m.get("object") == "model" for m in data["data"])
            )
            _report(
                "GET /v1/models",
                ok,
                f"{len(data.get('data', []))} models, format={'OpenAI' if ok else 'WRONG'}",
            )
        except Exception as e:
            _report("GET /v1/models", False, str(e))

        # ------------------------------------------------------------------
        # Test 3: Non-streaming chat completion
        # ------------------------------------------------------------------
        chat_body = {
            "model": test_model,
            "messages": [{"role": "user", "content": "Say hello in exactly one word."}],
            "stream": False,
            "temperature": 0.1,
            "max_tokens": 500,  # High enough for thinking models (qwen3, deepseek-r1)
        }
        status, body = await apost(f"{base}/v1/chat/completions", chat_body, timeout=120)
        try:
            data = json.loads(body)
            ok = (
                status == 200
                and data.get("object") == "chat.completion"
                and len(data.get("choices", [])) > 0
                and data["choices"][0].get("message", {}).get("content", "") != ""
            )
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")[:50]
            _report("POST /v1/chat/completions (sync)", ok, f'response="{content}"')
        except Exception as e:
            _report("POST /v1/chat/completions (sync)", False, str(e))

        # ------------------------------------------------------------------
        # Test 4: Streaming chat completion
        # ------------------------------------------------------------------
        stream_body = {
            "model": test_model,
            "messages": [{"role": "user", "content": "Count from 1 to 3."}],
            "stream": True,
            "temperature": 0.1,
            "max_tokens": 500,  # High enough for thinking models
        }
        try:
            lines = await apost_stream(
                f"{base}/v1/chat/completions", stream_body, timeout=120,
            )
            has_data = any(l.startswith("data: {") for l in lines)
            has_done = any(l == "data: [DONE]" for l in lines)

            # Parse content from chunks
            content_parts = []
            for l in lines:
                if l.startswith("data: {"):
                    chunk = json.loads(l[6:])
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    c = delta.get("content", "")
                    if c:
                        content_parts.append(c)

            full_content = "".join(content_parts)[:60]
            ok = has_data and has_done and len(content_parts) > 0
            _report(
                "POST /v1/chat/completions (stream)",
                ok,
                f'{len(lines)} SSE lines, done={has_done}, content="{full_content}"',
            )
        except Exception as e:
            _report("POST /v1/chat/completions (stream)", False, str(e))

        # ------------------------------------------------------------------
        # Test 5: 404 on unknown path
        # ------------------------------------------------------------------
        status, _ = await aget(f"{base}/v1/unknown")
        _report("GET /v1/unknown → 404", status == 404, f"status={status}")

    finally:
        server.close()
        await server.wait_closed()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"  Result: {passed}/{total} tests passed")
    if passed == total:
        print("  🎉 All E2E tests passed!")
    else:
        print("  ⚠️  Some tests failed — check output above")
    print("=" * 60)

    return passed == total


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    print("=" * 60)
    print("  PROPAGUL PROXY — E2E Test")
    print("=" * 60)

    loop = asyncio.new_event_loop()
    try:
        success = loop.run_until_complete(run_tests(args.ollama_url))
    finally:
        loop.close()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
