"""Tests for propagul.mesh.proxy — Local OpenAI-compatible Reverse Proxy.

Tests cover:
- HTTP request parsing
- Ollama ↔ OpenAI translation (request + response + streaming chunks)
- Model listing translation
- Router dispatch (health, models, chat completions)
- Error handling (no backend, bad JSON, unknown path)
- Server lifecycle (start, connect, shutdown)

Compatible with Python 3.9+ and pytest without pytest-asyncio.
Uses asyncio.run() (not deprecated get_event_loop()) for cross-test isolation.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import MagicMock

from propagul.mesh.proxy import (
    HttpRequest,
    ProxyConfig,
    BackendRouter,
    _read_http_request,
    _write_http_response,
    _json_error,
    _translate_openai_to_ollama_chat,
    _translate_ollama_response_to_openai,
    _translate_ollama_stream_chunk_to_openai,
    _translate_ollama_models_to_openai,
    _compute_num_predict,
    _register_thinking_model,
    _thinking_models,
    _STREAM_EOF,
    _STREAM_QUEUE_MAXSIZE,
    _LOW_MAX_TOKENS_THRESHOLD,
    _DEFAULT_THINKING_BUDGET,
    DEFAULT_PROXY_PORT,
    DEFAULT_PROXY_HOST,
)


# ---------------------------------------------------------------------------
# Translation Tests
# ---------------------------------------------------------------------------

class TestOpenAIToOllamaTranslation:
    """Test OpenAI → Ollama request translation."""

    def test_basic_chat_request(self):
        openai_body = {
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        result = _translate_openai_to_ollama_chat(openai_body)
        assert result["model"] == "llama3.1:8b"
        assert result["messages"] == [{"role": "user", "content": "Hello"}]
        assert result["stream"] is False

    def test_with_temperature(self):
        openai_body = {"model": "llama3", "messages": [], "temperature": 0.7}
        result = _translate_openai_to_ollama_chat(openai_body)
        assert result["options"]["temperature"] == 0.7

    def test_with_max_tokens(self):
        """max_tokens → num_predict (Ollama naming)."""
        openai_body = {"model": "llama3", "messages": [], "max_tokens": 512}
        result = _translate_openai_to_ollama_chat(openai_body)
        assert result["options"]["num_predict"] == 512

    def test_with_top_p_and_stop(self):
        openai_body = {
            "model": "llama3", "messages": [],
            "top_p": 0.9, "stop": ["\n", "END"],
        }
        result = _translate_openai_to_ollama_chat(openai_body)
        assert result["options"]["top_p"] == 0.9
        assert result["options"]["stop"] == ["\n", "END"]

    def test_multiple_options_share_dict(self):
        """Ensure temperature + max_tokens + top_p land in same options dict.

        Uses max_tokens=500 (above LOW_THRESHOLD) so num_predict is set.
        """
        openai_body = {
            "model": "m", "messages": [],
            "temperature": 0.5, "max_tokens": 500, "top_p": 0.8,
        }
        result = _translate_openai_to_ollama_chat(openai_body)
        opts = result["options"]
        assert opts["temperature"] == 0.5
        assert opts["num_predict"] == 500
        assert opts["top_p"] == 0.8

    def test_low_max_tokens_omits_num_predict(self):
        """Low max_tokens on unknown model → num_predict omitted (safety net)."""
        openai_body = {
            "model": "unknown-model", "messages": [],
            "max_tokens": 100,
        }
        result = _translate_openai_to_ollama_chat(openai_body)
        # num_predict should NOT be in options for low max_tokens on unknown model
        assert "num_predict" not in result.get("options", {})

    def test_empty_body(self):
        result = _translate_openai_to_ollama_chat({})
        assert result["model"] == ""
        assert result["messages"] == []
        assert result["stream"] is False


class TestOllamaToOpenAITranslation:
    """Test Ollama → OpenAI response translation."""

    def test_basic_response(self):
        ollama_resp = {
            "message": {"role": "assistant", "content": "Hi there!"},
            "done": True,
            "eval_count": 5,
            "prompt_eval_count": 10,
        }
        result = _translate_ollama_response_to_openai(ollama_resp, "llama3")
        assert result["object"] == "chat.completion"
        assert result["model"] == "llama3"
        assert result["choices"][0]["message"]["content"] == "Hi there!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15

    def test_incomplete_response(self):
        ollama_resp = {
            "message": {"role": "assistant", "content": "partial"},
            "done": False,
        }
        result = _translate_ollama_response_to_openai(ollama_resp, "m")
        assert result["choices"][0]["finish_reason"] is None

    def test_response_has_id_and_created(self):
        ollama_resp = {"message": {"content": ""}, "done": True}
        result = _translate_ollama_response_to_openai(ollama_resp, "m")
        assert result["id"].startswith("chatcmpl-")
        assert result["created"] > 0


class TestStreamChunkTranslation:
    """Test Ollama NDJSON → OpenAI SSE translation."""

    def test_content_chunk(self):
        chunk = {"message": {"content": "Hello"}, "done": False}
        result = _translate_ollama_stream_chunk_to_openai(chunk, "llama3")
        assert result.startswith("data: ")
        assert result.endswith("\n\n")
        parsed = json.loads(result[6:])  # Strip "data: "
        assert parsed["choices"][0]["delta"]["content"] == "Hello"
        assert parsed["object"] == "chat.completion.chunk"

    def test_done_chunk(self):
        chunk = {"message": {"content": ""}, "done": True}
        result = _translate_ollama_stream_chunk_to_openai(chunk, "llama3")
        assert result == "data: [DONE]\n\n"


class TestModelsTranslation:
    """Test Ollama /api/tags → OpenAI /v1/models."""

    def test_basic_models(self):
        ollama_models = {
            "models": [
                {"name": "llama3:8b", "size": 4000000000},
                {"name": "mistral:latest", "size": 3500000000},
            ]
        }
        result = _translate_ollama_models_to_openai(ollama_models)
        assert result["object"] == "list"
        assert len(result["data"]) == 2
        assert result["data"][0]["id"] == "llama3:8b"
        assert result["data"][0]["object"] == "model"
        assert result["data"][0]["owned_by"] == "local"

    def test_empty_models(self):
        result = _translate_ollama_models_to_openai({"models": []})
        assert result["data"] == []

    def test_missing_models_key(self):
        result = _translate_ollama_models_to_openai({})
        assert result["data"] == []


# ---------------------------------------------------------------------------
# HTTP Response Building Tests
# ---------------------------------------------------------------------------

class TestHttpResponse:
    """Test HTTP response building."""

    def test_200_response(self):
        resp = _write_http_response(200, b'{"ok": true}')
        resp_str = resp.decode("utf-8")
        assert "HTTP/1.1 200 OK" in resp_str
        assert "Content-Length:" in resp_str
        assert b'{"ok": true}' in resp

    def test_error_response(self):
        resp = _json_error(502, "Backend down")
        assert b"HTTP/1.1 502 Bad Gateway" in resp
        body_start = resp.index(b"\r\n\r\n") + 4
        body = json.loads(resp[body_start:])
        assert body["error"]["message"] == "Backend down"
        assert body["error"]["code"] == 502

    def test_cors_headers(self):
        resp = _write_http_response(200, b"{}")
        resp_str = resp.decode("utf-8")
        assert "Access-Control-Allow-Origin: *" in resp_str
        assert "Access-Control-Allow-Methods: GET, POST, OPTIONS" in resp_str


# ---------------------------------------------------------------------------
# ProxyConfig Tests
# ---------------------------------------------------------------------------

class TestProxyConfig:
    """Test ProxyConfig defaults and URL normalization."""

    def test_defaults(self):
        config = ProxyConfig()
        assert config.host == DEFAULT_PROXY_HOST
        assert config.port == DEFAULT_PROXY_PORT
        assert config.backend_name == ""
        assert config.backend_url == ""

    def test_url_trailing_slash(self):
        config = ProxyConfig(backend_url="http://localhost:11434/")
        assert config.backend_url == "http://localhost:11434"

    def test_custom_config(self):
        config = ProxyConfig(
            host="0.0.0.0", port=9999,
            backend_name="ollama",
            backend_url="http://localhost:11434",
        )
        assert config.host == "0.0.0.0"
        assert config.port == 9999
        assert config.backend_name == "ollama"


# ---------------------------------------------------------------------------
# BackendRouter Tests (unit, mocked HTTP)
# Uses asyncio.run() for Python 3.9 compat (no pytest-asyncio needed)
# ---------------------------------------------------------------------------

def _make_mock_writer():
    """Create a mock asyncio.StreamWriter for testing."""
    writer = MagicMock()
    writer.write = MagicMock()

    async def _noop():
        pass

    writer.drain = _noop
    writer.close = MagicMock()
    writer.wait_closed = _noop
    return writer


def _make_request(method="GET", path="/", body=b""):
    req = HttpRequest()
    req.method = method
    req.path = path
    req.body = body
    return req


class TestBackendRouter:
    """Test router dispatch logic."""

    def test_health_endpoint(self):
        async def _test():
            config = ProxyConfig(backend_name="ollama", backend_url="http://localhost:11434")
            router = BackendRouter(config)
            writer = _make_mock_writer()
            req = _make_request("GET", "/health")
            await router.handle_request(req, writer)
            written = writer.write.call_args[0][0]
            assert b"200 OK" in written
            body_start = written.index(b"\r\n\r\n") + 4
            body = json.loads(written[body_start:])
            assert body["status"] == "ok"
            assert body["backend"] == "ollama"

        asyncio.run(_test())

    def test_root_returns_health(self):
        async def _test():
            config = ProxyConfig(backend_name="ollama", backend_url="http://localhost:11434")
            router = BackendRouter(config)
            writer = _make_mock_writer()
            req = _make_request("GET", "/")
            await router.handle_request(req, writer)
            written = writer.write.call_args[0][0]
            assert b"200 OK" in written

        asyncio.run(_test())

    def test_unknown_path_returns_404(self):
        async def _test():
            config = ProxyConfig(backend_name="ollama", backend_url="http://localhost:11434")
            router = BackendRouter(config)
            writer = _make_mock_writer()
            req = _make_request("GET", "/v1/unknown")
            await router.handle_request(req, writer)
            written = writer.write.call_args[0][0]
            assert b"404" in written

        asyncio.run(_test())

    def test_options_returns_200(self):
        async def _test():
            config = ProxyConfig(backend_name="ollama", backend_url="http://localhost:11434")
            router = BackendRouter(config)
            writer = _make_mock_writer()
            req = _make_request("OPTIONS", "/v1/chat/completions")
            await router.handle_request(req, writer)
            written = writer.write.call_args[0][0]
            assert b"200 OK" in written

        asyncio.run(_test())

    def test_no_backend_returns_503(self):
        async def _test():
            config = ProxyConfig(backend_name="", backend_url="")
            router = BackendRouter(config)
            writer = _make_mock_writer()
            req = _make_request("POST", "/v1/chat/completions",
                                b'{"model":"x","messages":[]}')
            await router.handle_request(req, writer)
            written = writer.write.call_args[0][0]
            assert b"503" in written

        asyncio.run(_test())

    def test_chat_completions_bad_json(self):
        async def _test():
            config = ProxyConfig(backend_name="ollama", backend_url="http://localhost:11434")
            router = BackendRouter(config)
            writer = _make_mock_writer()
            req = _make_request("POST", "/v1/chat/completions", b"not json")
            await router.handle_request(req, writer)
            written = writer.write.call_args[0][0]
            assert b"400" in written

        asyncio.run(_test())

    def test_chat_completions_wrong_method(self):
        async def _test():
            config = ProxyConfig(backend_name="ollama", backend_url="http://localhost:11434")
            router = BackendRouter(config)
            writer = _make_mock_writer()
            req = _make_request("GET", "/v1/chat/completions")
            await router.handle_request(req, writer)
            written = writer.write.call_args[0][0]
            assert b"400" in written

        asyncio.run(_test())

    def test_models_no_backend_returns_503(self):
        async def _test():
            config = ProxyConfig(backend_name="", backend_url="")
            router = BackendRouter(config)
            writer = _make_mock_writer()
            req = _make_request("GET", "/v1/models")
            await router.handle_request(req, writer)
            written = writer.write.call_args[0][0]
            assert b"503" in written

        asyncio.run(_test())


# ---------------------------------------------------------------------------
# Server Lifecycle Test
# ---------------------------------------------------------------------------

class TestServerLifecycle:
    """Test that the proxy server starts and accepts connections."""

    def test_proxy_starts_and_responds_to_health(self):
        """Start proxy, connect, verify health response, then shutdown."""

        async def _test():
            config = ProxyConfig(
                host="127.0.0.1", port=0,  # OS-assigned
                backend_name="ollama",
                backend_url="http://localhost:11434",
            )
            router = BackendRouter(config)

            async def _handler(r, w):
                from propagul.mesh.proxy import _read_http_request, _json_error
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

            server = await asyncio.start_server(_handler, "127.0.0.1", 0)
            port = server.sockets[0].getsockname()[1]

            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.write(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
                await writer.drain()

                data = await asyncio.wait_for(reader.read(4096), timeout=2.0)
                assert b"200 OK" in data
                assert b"propagul-mesh" in data

                writer.close()
                await writer.wait_closed()
            finally:
                server.close()
                await server.wait_closed()

        asyncio.run(_test())


# ---------------------------------------------------------------------------
# Backend Auth Forwarding Tests
# ---------------------------------------------------------------------------

class TestBackendAuthConfig:
    """Test ProxyConfig.backend_auth storage and validation."""

    def test_default_no_auth(self):
        config = ProxyConfig()
        assert config.backend_auth == ""

    def test_auth_stored(self):
        config = ProxyConfig(
            backend_url="http://localhost:1234",
            backend_auth="Bearer sk-lm-test123",
        )
        assert config.backend_auth == "Bearer sk-lm-test123"

    def test_ssrf_rejected_with_auth(self):
        """SSRF validation runs BEFORE auth is stored — auth doesn't bypass SSRF."""
        import pytest
        with pytest.raises(ValueError, match="SSRF blocked"):
            ProxyConfig(
                backend_url="http://evil.com:8080",
                backend_auth="Bearer sk-lm-test",
            )

    def test_router_backend_auth_propagates(self):
        """BackendRouter._backend_auth() returns the config value."""
        config = ProxyConfig(
            backend_url="http://localhost:11434",
            backend_auth="Bearer test-key",
        )
        router = BackendRouter(config)
        assert router._backend_auth() == "Bearer test-key"

    def test_router_backend_auth_empty_default(self):
        config = ProxyConfig()
        router = BackendRouter(config)
        assert router._backend_auth() == ""


class TestAuthHeaderInjection:
    """Test that auth headers are injected into HTTP helper functions."""

    def test_http_get_sync_adds_auth(self):
        """_http_get_sync builds a request with Authorization header when provided."""
        import urllib.request
        from unittest.mock import patch, MagicMock
        from propagul.mesh.proxy import _http_get_sync

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b'{"ok": true}'

        captured_req = []

        def mock_urlopen(req, timeout=None):
            captured_req.append(req)
            return mock_resp

        with patch.object(urllib.request, "urlopen", mock_urlopen):
            _http_get_sync("http://localhost:1234/v1/models", auth_header="Bearer sk-test")

        assert len(captured_req) == 1
        assert captured_req[0].get_header("Authorization") == "Bearer sk-test"

    def test_http_get_sync_no_auth_when_empty(self):
        """_http_get_sync does NOT add Authorization header when auth_header is empty."""
        import urllib.request
        from unittest.mock import patch, MagicMock
        from propagul.mesh.proxy import _http_get_sync

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b'{"ok": true}'

        captured_req = []

        def mock_urlopen(req, timeout=None):
            captured_req.append(req)
            return mock_resp

        with patch.object(urllib.request, "urlopen", mock_urlopen):
            _http_get_sync("http://localhost:1234/v1/models", auth_header="")

        assert len(captured_req) == 1
        assert captured_req[0].get_header("Authorization") is None

    def test_http_post_sync_adds_auth(self):
        """_http_post_sync builds a request with Authorization header when provided."""
        import urllib.request
        from unittest.mock import patch, MagicMock
        from propagul.mesh.proxy import _http_post_sync

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b'{"result": "ok"}'

        captured_req = []

        def mock_urlopen(req, timeout=None):
            captured_req.append(req)
            return mock_resp

        with patch.object(urllib.request, "urlopen", mock_urlopen):
            _http_post_sync(
                "http://localhost:1234/v1/chat/completions",
                b'{"model":"test"}',
                auth_header="Bearer sk-post-test",
            )

        assert len(captured_req) == 1
        assert captured_req[0].get_header("Authorization") == "Bearer sk-post-test"
        assert captured_req[0].get_header("Content-type") == "application/json"


# ---------------------------------------------------------------------------
# Stream Architecture Tests (Single-Thread-Per-Stream Pattern)
# ---------------------------------------------------------------------------

class TestStreamArchitecture:
    """Verify the streaming refactoring: single reader thread, asyncio.Queue bridge."""

    def test_stream_eof_is_singleton(self):
        """_STREAM_EOF must be a unique sentinel (identity-checked, not equality)."""
        assert _STREAM_EOF is not None
        assert _STREAM_EOF is not False
        assert _STREAM_EOF is not 0
        # Identity check: two imports of the same module yield the same object
        from propagul.mesh.proxy import _STREAM_EOF as eof2
        assert _STREAM_EOF is eof2

    def test_stream_queue_maxsize_is_bounded(self):
        """Queue must have a bounded maxsize to enforce backpressure."""
        assert isinstance(_STREAM_QUEUE_MAXSIZE, int)
        assert _STREAM_QUEUE_MAXSIZE > 0
        assert _STREAM_QUEUE_MAXSIZE <= 1024  # sanity: not absurdly large

    def test_queue_roundtrip_with_eof(self):
        """Prove the reader->queue->consumer pattern works with proper EOF."""

        async def _test():
            loop = asyncio.get_running_loop()
            q = asyncio.Queue(maxsize=_STREAM_QUEUE_MAXSIZE)

            lines = [b"line1\n", b"line2\n", b"line3\n"]
            received = []

            def _reader():
                for line in lines:
                    fut = asyncio.run_coroutine_threadsafe(q.put(line), loop)
                    fut.result()
                fut = asyncio.run_coroutine_threadsafe(q.put(_STREAM_EOF), loop)
                fut.result()

            loop.run_in_executor(None, _reader)

            while True:
                item = await q.get()
                if item is _STREAM_EOF:
                    break
                received.append(item)

            assert received == lines

        asyncio.run(_test())

    def test_queue_propagates_exception(self):
        """Reader thread exceptions must propagate to the async consumer."""

        async def _test():
            loop = asyncio.get_running_loop()
            q = asyncio.Queue(maxsize=_STREAM_QUEUE_MAXSIZE)

            def _reader():
                exc = ConnectionError("backend died")
                fut = asyncio.run_coroutine_threadsafe(q.put(exc), loop)
                fut.result()
                fut = asyncio.run_coroutine_threadsafe(q.put(_STREAM_EOF), loop)
                fut.result()

            loop.run_in_executor(None, _reader)

            item = await q.get()
            assert isinstance(item, ConnectionError)
            assert str(item) == "backend died"

        asyncio.run(_test())

    def test_queue_backpressure_blocks_reader(self):
        """With maxsize=1, reader thread should block until consumer reads."""

        async def _test():
            loop = asyncio.get_running_loop()
            q = asyncio.Queue(maxsize=1)

            def _reader():
                # First put: immediate
                fut = asyncio.run_coroutine_threadsafe(q.put("a"), loop)
                fut.result()
                # Second put: should block until consumer reads
                fut = asyncio.run_coroutine_threadsafe(q.put("b"), loop)
                fut.result()
                fut = asyncio.run_coroutine_threadsafe(q.put(_STREAM_EOF), loop)
                fut.result()

            loop.run_in_executor(None, _reader)

            # Small delay to let reader put "a"
            await asyncio.sleep(0.05)
            item1 = await q.get()
            assert item1 == "a"

            await asyncio.sleep(0.05)
            item2 = await q.get()
            assert item2 == "b"

            eof = await q.get()
            assert eof is _STREAM_EOF

        asyncio.run(_test())


class TestThinkingBudgetInflation:
    """Tests for thinking model budget inflation logic.

    Verifies that:
    - Known thinking models get inflated num_predict
    - Unknown models with low max_tokens omit num_predict (safety net)
    - Unknown models with high max_tokens pass through 1:1
    - Response-driven thinking model registration works
    - Streaming chunks also trigger registration
    - ProxyConfig.thinking_budget is configurable
    """

    def setup_method(self):
        """Reset module-level thinking model cache before each test."""
        _thinking_models.clear()

    def test_compute_num_predict_known_thinking_model(self):
        """Known thinking model: max_tokens + budget."""
        _thinking_models.add("qwen3:30b")
        result = _compute_num_predict("qwen3:30b", 512, 4096)
        assert result == 512 + 4096

    def test_compute_num_predict_unknown_low_tokens(self):
        """Unknown model + low max_tokens → None (safety net)."""
        result = _compute_num_predict("mystery-model", 50, 4096)
        assert result is None

    def test_compute_num_predict_unknown_high_tokens(self):
        """Unknown model + high max_tokens → passthrough 1:1."""
        result = _compute_num_predict("llama3:8b", 2048, 4096)
        assert result == 2048

    def test_compute_num_predict_threshold_boundary(self):
        """Exact threshold value should still omit (< check, not <=)."""
        # max_tokens == threshold-1 → omit
        result = _compute_num_predict("x", _LOW_MAX_TOKENS_THRESHOLD - 1, 4096)
        assert result is None
        # max_tokens == threshold → passthrough
        result = _compute_num_predict("x", _LOW_MAX_TOKENS_THRESHOLD, 4096)
        assert result == _LOW_MAX_TOKENS_THRESHOLD

    def test_register_thinking_model_caches(self):
        """_register_thinking_model adds to module-level set."""
        assert "deepseek-r1:70b" not in _thinking_models
        _register_thinking_model("deepseek-r1:70b")
        assert "deepseek-r1:70b" in _thinking_models

    def test_register_thinking_model_idempotent(self):
        """Double registration doesn't error or duplicate."""
        _register_thinking_model("qwq:32b")
        _register_thinking_model("qwq:32b")
        assert "qwq:32b" in _thinking_models

    def test_register_thinking_model_empty_noop(self):
        """Empty model name is ignored."""
        _register_thinking_model("")
        assert "" not in _thinking_models

    def test_response_registers_thinking_model(self):
        """_translate_ollama_response_to_openai registers model when thinking field present."""
        ollama_resp = {
            "message": {
                "role": "assistant",
                "content": "The answer is 42.",
                "thinking": "Let me reason step by step...",
            },
            "done": True,
            "eval_count": 100,
            "prompt_eval_count": 50,
        }
        result = _translate_ollama_response_to_openai(ollama_resp, "qwen3:30b")
        # Model should now be registered
        assert "qwen3:30b" in _thinking_models
        # Thinking field should be forwarded
        assert result["choices"][0]["message"]["thinking"] == "Let me reason step by step..."

    def test_stream_chunk_registers_thinking_model(self):
        """Streaming chunk with thinking tokens registers model."""
        chunk = {
            "message": {"role": "assistant", "content": "", "thinking": "Reasoning..."},
            "done": False,
        }
        _translate_ollama_stream_chunk_to_openai(chunk, "deepseek-r1:14b")
        assert "deepseek-r1:14b" in _thinking_models

    def test_budget_inflation_in_translation(self):
        """Full integration: registered model gets inflated num_predict."""
        _register_thinking_model("qwen3:30b")
        openai_body = {
            "model": "qwen3:30b", "messages": [],
            "max_tokens": 100,
        }
        result = _translate_openai_to_ollama_chat(openai_body, thinking_budget=4096)
        # Should inflate even though max_tokens < LOW_THRESHOLD (model is KNOWN)
        assert result["options"]["num_predict"] == 100 + 4096

    def test_custom_thinking_budget(self):
        """Configurable thinking_budget in ProxyConfig."""
        cfg = ProxyConfig(thinking_budget=8192)
        assert cfg.thinking_budget == 8192

    def test_zero_thinking_budget_disables_inflation(self):
        """thinking_budget=0 disables inflation for known models."""
        _register_thinking_model("qwen3:30b")
        result = _compute_num_predict("qwen3:30b", 512, 0)
        # 512 + 0 = 512 — no inflation
        assert result == 512

    def test_non_thinking_response_no_registration(self):
        """Response without thinking field does not register model."""
        ollama_resp = {
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True,
            "eval_count": 10,
            "prompt_eval_count": 5,
        }
        _translate_ollama_response_to_openai(ollama_resp, "llama3:8b")
        assert "llama3:8b" not in _thinking_models
