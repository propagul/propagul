#!/usr/bin/env python3
"""Universal E2E test script for Propagul Mesh backend adapters.

Tests: detection, model listing, sync chat, streaming chat, proxy passthrough.
Zero external dependencies — pure stdlib.

Usage:
    python scripts/test_backend_e2e.py --backend ollama --url http://localhost:11434
    python scripts/test_backend_e2e.py --backend vllm --url http://localhost:8000
    python scripts/test_backend_e2e.py --backend vllm --url http://localhost:8000 --proxy-url http://localhost:8787
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error


# ─── Test Configuration ──────────────────────────────────────────
BACKENDS = {
    "ollama": {
        "health_path": "/api/version",
        "models_path": "/api/tags",
        "chat_path": "/v1/chat/completions",
        "models_key": "models",
    },
    "vllm": {
        "health_path": "/v1/models",
        "models_path": "/v1/models",
        "chat_path": "/v1/chat/completions",
        "models_key": "data",
    },
    "tgi": {
        "health_path": "/info",
        "models_path": "/info",
        "chat_path": "/v1/chat/completions",
        "models_key": None,  # TGI /info returns model_id, not a list
    },
    "llama_cpp": {
        "health_path": "/health",
        "models_path": "/v1/models",
        "chat_path": "/v1/chat/completions",
        "models_key": "data",
    },
    "lm_studio": {
        "health_path": "/v1/models",
        "models_path": "/v1/models",
        "chat_path": "/v1/chat/completions",
        "models_key": "data",
    },
}


def _request(url: str, method: str = "GET", body: dict = None,
             headers: dict = None, timeout: float = 15.0) -> tuple:
    """Make HTTP request. Returns (status_code, parsed_json_or_None, raw_text)."""
    hdrs = {"Accept": "application/json"}
    if headers:
        hdrs.update(headers)

    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        hdrs["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw), raw
            except json.JSONDecodeError:
                return resp.status, None, raw
    except urllib.error.HTTPError as e:
        body_text = ""
        try:
            body_text = e.read().decode("utf-8")
        except Exception:
            pass
        return e.code, None, body_text
    except Exception as e:
        return 0, None, str(e)


def _stream_request(url: str, body: dict, timeout: float = 30.0) -> tuple:
    """Make streaming request. Returns (status_code, chunks_list, first_token_ms)."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    chunks = []
    first_token_ms = None
    start = time.monotonic()

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        chunks.append(chunk)
                        if first_token_ms is None:
                            first_token_ms = (time.monotonic() - start) * 1000
                    except json.JSONDecodeError:
                        pass
            return resp.status, chunks, first_token_ms
    except urllib.error.HTTPError as e:
        return e.code, [], None
    except Exception as e:
        return 0, [], None


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.duration_ms = 0.0
        self.detail = ""
        self.error = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "duration_ms": round(self.duration_ms, 1),
            "detail": self.detail,
            "error": self.error,
        }


def run_test(name: str, func) -> TestResult:
    """Run a single test function and capture result."""
    result = TestResult(name)
    start = time.monotonic()
    try:
        func(result)
    except Exception as e:
        result.error = f"Exception: {e}"
        result.passed = False
    result.duration_ms = (time.monotonic() - start) * 1000
    return result


# ─── Test Functions ──────────────────────────────────────────────

def test_health(base_url: str, backend: str) -> callable:
    cfg = BACKENDS[backend]

    def _test(result: TestResult):
        url = f"{base_url}{cfg['health_path']}"
        status, body, raw = _request(url)
        if status == 200:
            result.passed = True
            result.detail = f"HTTP 200 from {url}"
            if body:
                # Show version or model_id if available
                for key in ("version", "model_id", "status"):
                    if key in body:
                        result.detail += f" ({key}={body[key]})"
                        break
        else:
            result.error = f"HTTP {status}: {raw[:200]}"

    return _test


def test_models(base_url: str, backend: str) -> callable:
    cfg = BACKENDS[backend]

    def _test(result: TestResult):
        url = f"{base_url}{cfg['models_path']}"
        status, body, raw = _request(url)
        if status != 200:
            result.error = f"HTTP {status}: {raw[:200]}"
            return

        if cfg["models_key"] and body:
            models = body.get(cfg["models_key"], [])
            if isinstance(models, list):
                names = []
                for m in models[:5]:
                    if isinstance(m, dict):
                        names.append(m.get("id") or m.get("name") or m.get("model", "?"))
                result.passed = True
                result.detail = f"{len(models)} model(s): {', '.join(names)}"
            else:
                result.error = f"Unexpected format for models key '{cfg['models_key']}'"
        elif backend == "tgi" and body:
            # TGI returns {model_id: "..."}
            result.passed = True
            result.detail = f"model_id={body.get('model_id', '?')}"
        else:
            result.passed = True
            result.detail = f"HTTP 200 (no model list expected)"

    return _test


def test_chat_sync(base_url: str, backend: str, model: str = "") -> callable:
    def _test(result: TestResult):
        url = f"{base_url}/v1/chat/completions"
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
            "max_tokens": 10,
            "stream": False,
        }
        status, resp, raw = _request(url, body=body, timeout=30.0)
        if status != 200:
            result.error = f"HTTP {status}: {raw[:300]}"
            return

        if resp and "choices" in resp:
            content = resp["choices"][0].get("message", {}).get("content", "")
            result.passed = True
            result.detail = f"Response: '{content[:50]}'"
        else:
            result.error = f"Unexpected response format: {raw[:200]}"

    return _test


def test_chat_stream(base_url: str, backend: str, model: str = "") -> callable:
    def _test(result: TestResult):
        url = f"{base_url}/v1/chat/completions"
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Count from 1 to 3."}],
            "max_tokens": 30,
            "stream": True,
        }
        status, chunks, ttft = _stream_request(url, body)
        if status != 200:
            result.error = f"HTTP {status}"
            return

        if chunks:
            result.passed = True
            result.detail = f"{len(chunks)} chunks, TTFT={ttft:.0f}ms"
        else:
            result.error = "No chunks received"

    return _test


def test_detection(backend: str) -> callable:
    """Test that the Propagul backend detection module works."""
    def _test(result: TestResult):
        try:
            from propagul.mesh.backends.detect import detect
            detected = detect(timeout=3.0)
            found = [d for d in detected if d.name == backend]
            if found:
                result.passed = True
                result.detail = f"Detected {backend} at {found[0].url} (v{found[0].version}, conf={found[0].confidence})"
            else:
                names = [d.name for d in detected]
                result.error = f"Backend '{backend}' not detected. Found: {names}"
        except ImportError as e:
            result.error = f"Import error (run from project root): {e}"

    return _test


def test_proxy(proxy_url: str, model: str = "") -> callable:
    """Test the Propagul local proxy passthrough."""
    def _test(result: TestResult):
        # Test /v1/models
        status, body, raw = _request(f"{proxy_url}/v1/models")
        if status != 200:
            result.error = f"Proxy /v1/models returned HTTP {status}: {raw[:200]}"
            return

        # Test sync chat
        chat_body = {
            "model": model,
            "messages": [{"role": "user", "content": "Say 'proxy works' and nothing else."}],
            "max_tokens": 10,
            "stream": False,
        }
        status, resp, raw = _request(f"{proxy_url}/v1/chat/completions", body=chat_body, timeout=30.0)
        if status != 200:
            result.error = f"Proxy chat returned HTTP {status}: {raw[:200]}"
            return

        if resp and "choices" in resp:
            content = resp["choices"][0].get("message", {}).get("content", "")
            result.passed = True
            result.detail = f"Proxy response: '{content[:50]}'"
        else:
            result.error = f"Unexpected proxy response: {raw[:200]}"

    return _test


# ─── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Propagul Mesh E2E Backend Tester")
    parser.add_argument("--backend", required=True, choices=list(BACKENDS.keys()),
                        help="Backend to test")
    parser.add_argument("--url", required=True,
                        help="Base URL of the backend (e.g., http://localhost:11434)")
    parser.add_argument("--model", default="",
                        help="Model name for chat tests (auto-detected if empty)")
    parser.add_argument("--proxy-url", default="",
                        help="Local proxy URL to test (e.g., http://localhost:8787)")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    parser.add_argument("--skip-detection", action="store_true",
                        help="Skip Propagul detection test (useful on remote machines)")
    args = parser.parse_args()

    results = []

    # 1. Health check
    results.append(run_test("health", test_health(args.url, args.backend)))

    # 2. Model listing
    results.append(run_test("models", test_models(args.url, args.backend)))

    # Auto-detect model if not specified
    model = args.model
    if not model:
        cfg = BACKENDS[args.backend]
        status, body, _ = _request(f"{args.url}{cfg['models_path']}")
        if status == 200 and body:
            if cfg["models_key"] and isinstance(body.get(cfg["models_key"]), list):
                models_list = body[cfg["models_key"]]
                if models_list:
                    m = models_list[0]
                    model = m.get("id") or m.get("name") or m.get("model", "")
            elif args.backend == "tgi":
                model = body.get("model_id", "")
        if model:
            print(f"  Auto-detected model: {model}")

    # 3. Sync chat
    if model:
        results.append(run_test("chat_sync", test_chat_sync(args.url, args.backend, model)))
    else:
        r = TestResult("chat_sync")
        r.error = "No model available for chat test"
        results.append(r)

    # 4. Streaming chat
    if model:
        results.append(run_test("chat_stream", test_chat_stream(args.url, args.backend, model)))
    else:
        r = TestResult("chat_stream")
        r.error = "No model available for stream test"
        results.append(r)

    # 5. Detection
    if not args.skip_detection:
        results.append(run_test("detection", test_detection(args.backend)))

    # 6. Proxy test
    if args.proxy_url:
        results.append(run_test("proxy", test_proxy(args.proxy_url, model)))

    # Output
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    if args.json:
        output = {
            "backend": args.backend,
            "url": args.url,
            "model": model,
            "timestamp": time.time(),
            "passed": passed,
            "total": total,
            "results": [r.to_dict() for r in results],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\n{'=' * 60}")
        print(f"  Propagul E2E: {args.backend} @ {args.url}")
        print(f"{'=' * 60}")
        for r in results:
            icon = "✅" if r.passed else "❌"
            print(f"  {icon} {r.name:20s} {r.duration_ms:7.1f}ms  {r.detail or r.error}")
        print(f"{'=' * 60}")
        print(f"  Result: {passed}/{total} passed")
        print(f"{'=' * 60}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
