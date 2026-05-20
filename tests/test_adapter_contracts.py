"""test_adapter_contracts — Golden-file contract tests for backend adapters.

Tests that each backend adapter correctly parses REAL API responses.
Uses recorded fixtures (tests/fixtures/) to mock _http_get() and verify
field extraction, bounds, and schema compliance.

This catches silent failures when upstream backends change their API format.

Coverage:
    - vLLM: /v1/models + /metrics (Prometheus)
    - TGI: /info + /metrics (Prometheus)
    - llama.cpp: /health + /props + /slots + /v1/models
    - LM Studio: /v1/models

Edge cases:
    - Empty model lists
    - Missing fields
    - Malformed JSON
    - Unexpected types
"""

import json
import pathlib
from unittest.mock import patch

import pytest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> bytes:
    """Load a fixture file as bytes."""
    return (FIXTURES / name).read_bytes()


def _load_fixture_text(name: str) -> bytes:
    """Load a text fixture (Prometheus metrics) as bytes."""
    return (FIXTURES / name).read_bytes()


# ═══════════════════════════════════════════════════════════════════
# vLLM Contract Tests
# ═══════════════════════════════════════════════════════════════════

class TestVllmContracts:
    """Verify vLLM adapter parses real API responses correctly."""

    def test_get_models_happy_path(self):
        """Parse a real vLLM /v1/models response."""
        from propagul.mesh.backends.vllm import get_models
        fixture = _load_fixture("vllm_v1_models.json")

        with patch("propagul.mesh.backends.vllm._http_get", return_value=fixture):
            models = get_models()

        assert len(models) == 1
        assert models[0].id == "meta-llama/Llama-3-8B-Instruct"
        assert models[0].object == "model"
        assert models[0].max_model_len == 8192
        assert models[0].owned_by == "vllm"

    def test_get_models_empty_data(self):
        """Handle /v1/models with empty data array."""
        from propagul.mesh.backends.vllm import get_models
        fixture = json.dumps({"object": "list", "data": []}).encode()

        with patch("propagul.mesh.backends.vllm._http_get", return_value=fixture):
            models = get_models()

        assert models == []

    def test_get_models_missing_fields(self):
        """Handle model entries with missing optional fields."""
        from propagul.mesh.backends.vllm import get_models
        fixture = json.dumps({
            "object": "list",
            "data": [{"id": "test-model"}]
        }).encode()

        with patch("propagul.mesh.backends.vllm._http_get", return_value=fixture):
            models = get_models()

        assert len(models) == 1
        assert models[0].id == "test-model"
        assert models[0].max_model_len == 0  # Default
        assert models[0].owned_by == ""  # Default

    def test_get_metrics_happy_path(self):
        """Parse real vLLM Prometheus metrics."""
        from propagul.mesh.backends.vllm import get_metrics
        fixture = _load_fixture_text("vllm_metrics.txt")

        with patch("propagul.mesh.backends.vllm._http_get", return_value=fixture):
            metrics = get_metrics()

        assert metrics is not None
        assert metrics.num_requests_running == 3
        assert metrics.num_requests_waiting == 1
        assert abs(metrics.gpu_cache_usage_perc - 0.42) < 0.001
        assert abs(metrics.avg_prompt_throughput - 1250.5) < 0.1
        assert abs(metrics.avg_generation_throughput - 85.3) < 0.1
        assert metrics.cpu_cache_usage_perc == 0.0

    def test_get_metrics_empty_prometheus(self):
        """Handle empty Prometheus response (server up, no metrics yet)."""
        from propagul.mesh.backends.vllm import get_metrics
        fixture = b"# No metrics available yet\n"

        with patch("propagul.mesh.backends.vllm._http_get", return_value=fixture):
            metrics = get_metrics()

        assert metrics is not None
        assert metrics.num_requests_running == 0  # All defaults

    def test_collect_telemetry_happy_path(self):
        """Full telemetry collection with mocked responses."""
        from propagul.mesh.backends.vllm import collect_telemetry
        models_fixture = _load_fixture("vllm_v1_models.json")
        metrics_fixture = _load_fixture_text("vllm_metrics.txt")

        def mock_get(url, timeout=5.0):
            if "/v1/models" in url:
                return models_fixture
            elif "/metrics" in url:
                return metrics_fixture
            elif "/health" in url:
                return b""  # 200 OK (body doesn't matter for health)
            return None

        with patch("propagul.mesh.backends.vllm._http_get", side_effect=mock_get):
            result = collect_telemetry()

        assert result["backend"] == "vllm"
        assert result["healthy"] is True
        assert result["model_count"] == 1
        assert result["running_count"] == 1
        assert len(result["models"]) == 1
        assert result["models"][0]["name"] == "meta-llama/Llama-3-8B-Instruct"
        assert result["models"][0]["backend"] == "vllm"
        # Metrics should be enriched
        assert "metrics" in result
        assert result["metrics"]["requests_running"] == 3
        assert result["metrics"]["gpu_cache_usage"] == 42.0

    def test_collect_telemetry_schema_keys(self):
        """Verify all required heartbeat schema keys are present."""
        from propagul.mesh.backends.vllm import collect_telemetry
        models_fixture = _load_fixture("vllm_v1_models.json")

        def mock_get(url, timeout=5.0):
            if "/v1/models" in url:
                return models_fixture
            elif "/health" in url:
                return b""
            return None

        with patch("propagul.mesh.backends.vllm._http_get", side_effect=mock_get):
            result = collect_telemetry()

        required_keys = {"backend", "url", "healthy", "models", "model_count", "running_count"}
        assert required_keys.issubset(set(result.keys())), (
            f"Missing: {required_keys - set(result.keys())}"
        )

    def test_get_models_malformed_json(self):
        """Handle malformed JSON without crashing."""
        from propagul.mesh.backends.vllm import get_models
        with patch("propagul.mesh.backends.vllm._http_get", return_value=b"not json{"):
            models = get_models()
        assert models == []


# ═══════════════════════════════════════════════════════════════════
# TGI Contract Tests
# ═══════════════════════════════════════════════════════════════════

class TestTgiContracts:
    """Verify TGI adapter parses real API responses correctly."""

    def test_get_info_happy_path(self):
        """Parse a real TGI /info response."""
        from propagul.mesh.backends.tgi import get_info
        fixture = _load_fixture("tgi_info.json")

        with patch("propagul.mesh.backends.tgi._http_get", return_value=fixture):
            info = get_info()

        assert info is not None
        assert info.model_id == "HuggingFaceH4/zephyr-7b-beta"
        assert info.model_dtype == "torch.float16"
        assert info.model_device_type == "cuda"
        assert info.max_total_tokens == 2048
        assert info.max_input_length == 1024
        assert info.version == "2.4.1"
        assert info.sha == "abc123def456"

    def test_get_info_missing_model_id(self):
        """Non-TGI server: /info without model_id returns None."""
        from propagul.mesh.backends.tgi import get_info
        fixture = json.dumps({"version": "1.0", "some_field": True}).encode()

        with patch("propagul.mesh.backends.tgi._http_get", return_value=fixture):
            info = get_info()

        assert info is None  # Not TGI

    def test_get_metrics_happy_path(self):
        """Parse real TGI Prometheus metrics."""
        from propagul.mesh.backends.tgi import get_metrics
        fixture = _load_fixture_text("tgi_metrics.txt")

        with patch("propagul.mesh.backends.tgi._http_get", return_value=fixture):
            metrics = get_metrics()

        assert metrics is not None
        assert metrics.queue_size == 5
        assert metrics.batch_current_size == 8
        assert abs(metrics.inference_duration_sum - 142.75) < 0.01
        assert metrics.total_tokens_generated == 98500

    def test_collect_telemetry_happy_path(self):
        """Full TGI telemetry with mocked responses."""
        from propagul.mesh.backends.tgi import collect_telemetry
        info_fixture = _load_fixture("tgi_info.json")
        metrics_fixture = _load_fixture_text("tgi_metrics.txt")

        def mock_get(url, timeout=5.0):
            if "/info" in url:
                return info_fixture
            elif "/health" in url:
                return b""  # 200 OK
            elif "/metrics" in url:
                return metrics_fixture
            return None

        with patch("propagul.mesh.backends.tgi._http_get", side_effect=mock_get):
            result = collect_telemetry()

        assert result["backend"] == "tgi"
        assert result["healthy"] is True
        assert result["model_count"] == 1
        assert result["models"][0]["name"] == "HuggingFaceH4/zephyr-7b-beta"
        assert result["models"][0]["quantization"] == "torch.float16"
        assert result["model_info"]["version"] == "2.4.1"
        assert result["model_info"]["device"] == "cuda"
        assert result["metrics"]["queue_size"] == 5
        assert result["metrics"]["tokens_generated"] == 98500

    def test_collect_telemetry_schema_keys(self):
        """Verify all required heartbeat schema keys."""
        from propagul.mesh.backends.tgi import collect_telemetry

        with patch("propagul.mesh.backends.tgi._http_get", return_value=None):
            result = collect_telemetry()

        required_keys = {"backend", "url", "healthy", "models", "model_count", "running_count"}
        assert required_keys.issubset(set(result.keys()))

    def test_is_tgi_with_valid_info(self):
        """is_tgi returns True for valid TGI /info response."""
        from propagul.mesh.backends.tgi import is_tgi
        fixture = _load_fixture("tgi_info.json")

        with patch("propagul.mesh.backends.tgi._http_get", return_value=fixture):
            assert is_tgi() is True

    def test_is_tgi_without_model_id(self):
        """is_tgi returns False when /info lacks model_id."""
        from propagul.mesh.backends.tgi import is_tgi
        fixture = json.dumps({"not_a_model": True}).encode()

        with patch("propagul.mesh.backends.tgi._http_get", return_value=fixture):
            assert is_tgi() is False


# ═══════════════════════════════════════════════════════════════════
# llama.cpp Contract Tests
# ═══════════════════════════════════════════════════════════════════

class TestLlamaCppContracts:
    """Verify llama.cpp adapter parses real API responses correctly."""

    def test_check_health_ok(self):
        """Parse /health → status ok."""
        from propagul.mesh.backends.llamacpp import check_health
        fixture = _load_fixture("llamacpp_health.json")

        with patch("propagul.mesh.backends.llamacpp._http_get", return_value=fixture):
            result = check_health()

        assert result["healthy"] is True
        assert result["status"] == "ok"

    def test_check_health_loading(self):
        """Parse /health → loading model."""
        from propagul.mesh.backends.llamacpp import check_health
        fixture = json.dumps({"status": "loading model"}).encode()

        with patch("propagul.mesh.backends.llamacpp._http_get", return_value=fixture):
            result = check_health()

        assert result["healthy"] is False
        assert result["status"] == "loading model"

    def test_get_props_happy_path(self):
        """Parse /props with model path and context size."""
        from propagul.mesh.backends.llamacpp import get_props
        fixture = _load_fixture("llamacpp_props.json")

        with patch("propagul.mesh.backends.llamacpp._http_get", return_value=fixture):
            props = get_props()

        assert "default_generation_settings" in props
        settings = props["default_generation_settings"]
        assert settings["n_ctx"] == 4096
        assert "llama-3.1-8b" in settings["model"]

    def test_get_slots_happy_path(self):
        """Parse /slots with active and idle slots."""
        from propagul.mesh.backends.llamacpp import get_slots
        fixture = _load_fixture("llamacpp_slots.json")

        with patch("propagul.mesh.backends.llamacpp._http_get", return_value=fixture):
            slots = get_slots()

        assert len(slots) == 4
        active = sum(1 for s in slots if s.get("is_processing"))
        assert active == 2  # 2 active, 2 idle

    def test_get_models_v1_happy_path(self):
        """Parse /v1/models OpenAI-compatible response."""
        from propagul.mesh.backends.llamacpp import get_models_v1
        fixture = _load_fixture("llamacpp_v1_models.json")

        with patch("propagul.mesh.backends.llamacpp._http_get", return_value=fixture):
            models = get_models_v1()

        assert len(models) == 1
        assert models[0]["id"] == "llama-3.1-8b-instruct-q4_k_m"

    def test_extract_model_name_from_path(self):
        """_extract_model_name strips path and .gguf extension."""
        from propagul.mesh.backends.llamacpp import _extract_model_name

        props = {
            "default_generation_settings": {
                "model": "/models/llama-3.1-8b-instruct-q4_k_m.gguf"
            }
        }
        assert _extract_model_name(props) == "llama-3.1-8b-instruct-q4_k_m"

    def test_extract_model_name_no_extension(self):
        """Model path without .gguf extension."""
        from propagul.mesh.backends.llamacpp import _extract_model_name

        props = {"default_generation_settings": {"model": "/models/some-model"}}
        assert _extract_model_name(props) == "some-model"

    def test_extract_model_name_empty(self):
        """Empty model path returns 'unknown'."""
        from propagul.mesh.backends.llamacpp import _extract_model_name

        assert _extract_model_name({}) == "unknown"
        assert _extract_model_name({"default_generation_settings": {}}) == "unknown"
        assert _extract_model_name({"default_generation_settings": {"model": ""}}) == "unknown"

    def test_collect_telemetry_with_v1_models(self):
        """Full telemetry using /v1/models endpoint (preferred path)."""
        from propagul.mesh.backends.llamacpp import collect_telemetry
        health_fix = _load_fixture("llamacpp_health.json")
        models_fix = _load_fixture("llamacpp_v1_models.json")
        props_fix = _load_fixture("llamacpp_props.json")
        slots_fix = _load_fixture("llamacpp_slots.json")

        def mock_get(url, timeout=5.0):
            if "/health" in url and "/v1" not in url:
                return health_fix
            elif "/v1/models" in url:
                return models_fix
            elif "/props" in url:
                return props_fix
            elif "/slots" in url:
                return slots_fix
            return None

        with patch("propagul.mesh.backends.llamacpp._http_get", side_effect=mock_get):
            result = collect_telemetry()

        assert result["backend"] == "llama_cpp"
        assert result["healthy"] is True
        assert result["model_count"] == 1
        assert result["running_count"] == 1
        assert result["models"][0]["backend"] == "llama_cpp"
        # Slot metrics should be present
        assert result["slots"]["total"] == 4
        assert result["slots"]["active"] == 2
        assert result["slots"]["idle"] == 2

    def test_collect_telemetry_fallback_to_props(self):
        """Fallback to /props when /v1/models is unavailable."""
        from propagul.mesh.backends.llamacpp import collect_telemetry
        health_fix = _load_fixture("llamacpp_health.json")
        props_fix = _load_fixture("llamacpp_props.json")

        def mock_get(url, timeout=5.0):
            if "/health" in url and "/v1" not in url:
                return health_fix
            elif "/v1/models" in url:
                return None  # Not available (older build)
            elif "/props" in url:
                return props_fix
            elif "/slots" in url:
                return None
            return None

        with patch("propagul.mesh.backends.llamacpp._http_get", side_effect=mock_get):
            result = collect_telemetry()

        assert result["backend"] == "llama_cpp"
        assert result["healthy"] is True
        assert result["model_count"] == 1
        assert result["models"][0]["name"] == "llama-3.1-8b-instruct-q4_k_m"
        assert result["models"][0].get("context_size") == 4096

    def test_collect_telemetry_unreachable(self):
        """Unreachable server returns correct schema."""
        from propagul.mesh.backends.llamacpp import collect_telemetry

        with patch("propagul.mesh.backends.llamacpp._http_get", return_value=None):
            result = collect_telemetry()

        assert result["backend"] == "llama_cpp"
        assert result["healthy"] is False
        assert result["models"] == []
        assert result["model_count"] == 0
        assert result["running_count"] == 0

    def test_collect_telemetry_schema_keys(self):
        """Verify all required heartbeat schema keys."""
        from propagul.mesh.backends.llamacpp import collect_telemetry

        with patch("propagul.mesh.backends.llamacpp._http_get", return_value=None):
            result = collect_telemetry()

        required_keys = {"backend", "url", "healthy", "models", "model_count", "running_count"}
        assert required_keys.issubset(set(result.keys()))


# ═══════════════════════════════════════════════════════════════════
# LM Studio Contract Tests
# ═══════════════════════════════════════════════════════════════════

class TestLmStudioContracts:
    """Verify LM Studio adapter parses real API responses correctly."""

    def test_get_models_happy_path(self):
        """Parse a real LM Studio /v1/models response with multiple models."""
        from propagul.mesh.backends.lm_studio import get_models
        fixture = _load_fixture("lm_studio_v1_models.json")

        with patch("propagul.mesh.backends.lm_studio._http_get", return_value=fixture):
            models = get_models()

        assert len(models) == 2
        assert "Meta-Llama-3" in models[0].id
        assert "Mistral-7B" in models[1].id

    def test_get_models_empty(self):
        """Handle /v1/models with empty data."""
        from propagul.mesh.backends.lm_studio import get_models
        fixture = json.dumps({"object": "list", "data": []}).encode()

        with patch("propagul.mesh.backends.lm_studio._http_get", return_value=fixture):
            models = get_models()

        assert models == []

    def test_collect_telemetry_happy_path(self):
        """Full LM Studio telemetry with mocked response."""
        from propagul.mesh.backends.lm_studio import collect_telemetry
        models_fixture = _load_fixture("lm_studio_v1_models.json")

        def mock_get(url, timeout=5.0):
            if "/v1/models" in url:
                return models_fixture
            return None

        with patch("propagul.mesh.backends.lm_studio._http_get", side_effect=mock_get):
            result = collect_telemetry()

        assert result["backend"] == "lm_studio"
        assert result["healthy"] is True
        assert result["model_count"] == 2
        assert result["running_count"] == 2
        assert len(result["models"]) == 2
        # Check both models have correct backend tag
        for m in result["models"]:
            assert m["backend"] == "lm_studio"
            assert m["name"]  # Not empty

    def test_collect_telemetry_schema_compliance(self):
        """Verify heartbeat schema keys match Ollama adapter contract."""
        from propagul.mesh.backends.lm_studio import collect_telemetry

        with patch("propagul.mesh.backends.lm_studio._http_get", return_value=None):
            result = collect_telemetry()

        required_keys = {"backend", "url", "healthy", "models", "model_count", "running_count"}
        assert required_keys.issubset(set(result.keys()))


# ═══════════════════════════════════════════════════════════════════
# Cross-Adapter Schema Consistency
# ═══════════════════════════════════════════════════════════════════

class TestCrossAdapterSchema:
    """Verify all adapters return compatible telemetry schemas."""

    @pytest.mark.parametrize("adapter_module,module_path", [
        ("propagul.mesh.backends.vllm", "propagul.mesh.backends.vllm._http_get"),
        ("propagul.mesh.backends.tgi", "propagul.mesh.backends.tgi._http_get"),
        ("propagul.mesh.backends.llamacpp", "propagul.mesh.backends.llamacpp._http_get"),
        ("propagul.mesh.backends.lm_studio", "propagul.mesh.backends.lm_studio._http_get"),
    ])
    def test_unreachable_returns_consistent_schema(self, adapter_module, module_path):
        """All adapters must return same base schema when unreachable."""
        import importlib
        mod = importlib.import_module(adapter_module)

        with patch(module_path, return_value=None):
            result = mod.collect_telemetry(base_url="http://127.0.0.1:19999", timeout=0.1)

        # Required keys for heartbeat payload
        assert isinstance(result, dict)
        assert isinstance(result["backend"], str)
        assert len(result["backend"]) > 0
        assert isinstance(result["url"], str)
        assert result["healthy"] is False
        assert isinstance(result["models"], list)
        assert result["models"] == []
        assert result["model_count"] == 0
        assert isinstance(result["running_count"], int)
        assert result["running_count"] >= 0

    @pytest.mark.parametrize("adapter_module,module_path", [
        ("propagul.mesh.backends.vllm", "propagul.mesh.backends.vllm._http_get"),
        ("propagul.mesh.backends.tgi", "propagul.mesh.backends.tgi._http_get"),
        ("propagul.mesh.backends.llamacpp", "propagul.mesh.backends.llamacpp._http_get"),
        ("propagul.mesh.backends.lm_studio", "propagul.mesh.backends.lm_studio._http_get"),
    ])
    def test_model_list_items_have_required_fields(self, adapter_module, module_path):
        """Model items in telemetry must have name + backend fields."""
        import importlib
        mod = importlib.import_module(adapter_module)

        # Build a mock that returns a single-model response for each backend
        fixtures_map = {
            "vllm": _load_fixture("vllm_v1_models.json"),
            "tgi": _load_fixture("tgi_info.json"),
            "llamacpp": _load_fixture("llamacpp_v1_models.json"),
            "lm_studio": _load_fixture("lm_studio_v1_models.json"),
        }
        backend_name = adapter_module.rsplit(".", 1)[-1]
        # Map module name to fixture key
        fixture_key = {
            "vllm": "vllm", "tgi": "tgi",
            "llamacpp": "llamacpp", "lm_studio": "lm_studio",
        }[backend_name]

        def mock_get(url, timeout=5.0):
            # Return appropriate fixture based on URL
            return fixtures_map[fixture_key]

        with patch(module_path, side_effect=mock_get):
            result = mod.collect_telemetry()

        assert result["model_count"] > 0, f"{backend_name} should have at least 1 model"
        for model in result["models"]:
            assert "name" in model, f"{backend_name} model missing 'name'"
            assert len(model["name"]) > 0, f"{backend_name} model name is empty"
            assert "backend" in model, f"{backend_name} model missing 'backend'"
            assert model["backend"] == backend_name.replace("llamacpp", "llama_cpp")
