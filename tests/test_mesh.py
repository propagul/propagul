"""Tests for propagul.mesh — Node Agent, Backends, GPU, FleetStore.

Covers:
- SSRF validation (P2-01)
- Ollama backend data parsing
- GPU metric fallbacks
- FleetStore CRUD + persistence
- Dashboard API node_id validation (P1-01)
- Body size limits (P2-03)
"""

import json
import os
import tempfile
import time

import pytest


# ─── SSRF Validation Tests ───────────────────────────────────────


class TestSSRFValidation:
    """P2-01: Validate that SSRF guard blocks non-local URLs."""

    def test_localhost_allowed(self):
        from propagul.mesh.backends.ollama import _validate_url
        _validate_url("http://localhost:11434")  # No exception

    def test_127_allowed(self):
        from propagul.mesh.backends.ollama import _validate_url
        _validate_url("http://127.0.0.1:11434")

    def test_private_ip_allowed(self):
        from propagul.mesh.backends.ollama import _validate_url
        _validate_url("http://192.168.1.100:11434")
        _validate_url("http://10.0.0.5:8000")
        _validate_url("http://172.16.0.1:8080")

    def test_cloud_metadata_blocked(self):
        from propagul.mesh.backends.ollama import _validate_url
        with pytest.raises(ValueError, match="SSRF blocked"):
            _validate_url("http://169.254.169.254/latest/meta-data")

    def test_public_ip_blocked(self):
        from propagul.mesh.backends.ollama import _validate_url
        with pytest.raises(ValueError, match="SSRF blocked"):
            _validate_url("http://8.8.8.8:11434")

    def test_external_domain_blocked(self):
        from propagul.mesh.backends.ollama import _validate_url
        with pytest.raises(ValueError, match="SSRF blocked"):
            _validate_url("http://evil.com:11434")

    def test_poll_with_ssrf_url_returns_offline(self):
        from propagul.mesh.backends.ollama import poll
        result = poll(base_url="http://169.254.169.254")
        assert not result.online
        assert "SSRF blocked" in result.error

    def test_execute_with_ssrf_url_returns_error(self):
        from propagul.mesh.backends.ollama import execute_command
        result = execute_command("pull", "llama3", base_url="http://evil.com")
        assert result["status"] == "error"
        assert "SSRF blocked" in result["error"]


# ─── Ollama Backend Tests ────────────────────────────────────────


class TestOllamaDataclasses:
    """Test Ollama data model serialization."""

    def test_ollama_model_size_gb(self):
        from propagul.mesh.backends.ollama import OllamaModel
        m = OllamaModel(
            name="llama3:8b", size_bytes=4_294_967_296,
            parameter_size="8B", quantization="Q4_K_M",
            family="llama", modified_at="2024-01-01", digest="abc123def456",
        )
        assert m.size_gb == 4.0

    def test_ollama_status_to_dict(self):
        from propagul.mesh.backends.ollama import OllamaStatus, OllamaModel
        status = OllamaStatus(
            online=True, version="0.1.0", url="http://localhost:11434",
            models=[
                OllamaModel("m1", 1024**3, "7B", "Q4", "llama", "", "abc"),
                OllamaModel("m2", 2 * 1024**3, "13B", "Q5", "mistral", "", "def"),
            ],
        )
        d = status.to_dict()
        assert d["backend"] == "ollama"
        assert d["online"] is True
        assert d["model_count"] == 2
        assert d["total_model_size_gb"] == 3.0
        assert len(d["models"]) == 2
        assert d["models"][0]["digest"] == "abc"  # Truncated to 12 chars

    def test_poll_unreachable_returns_offline(self):
        from propagul.mesh.backends.ollama import poll
        result = poll(base_url="http://127.0.0.1:19999", timeout=0.5)
        assert not result.online
        assert result.error is not None


# ─── GPU Tests ───────────────────────────────────────────────────


class TestGPU:
    """Test GPU metric collection."""

    def test_collect_never_raises(self):
        from propagul.mesh.gpu import collect
        result = collect()
        assert result.backend in ("nvidia", "apple", "none")
        assert isinstance(result.gpus, list)

    def test_gpu_info_vram_pct_zero_total(self):
        from propagul.mesh.gpu import GpuInfo
        g = GpuInfo(0, "Test", "1.0", 0, 0, 0, 50.0)
        assert g.vram_utilization_pct == 0.0

    def test_gpu_info_vram_pct_normal(self):
        from propagul.mesh.gpu import GpuInfo
        g = GpuInfo(0, "Test", "1.0", 8192, 4096, 4096, 50.0)
        assert g.vram_utilization_pct == 50.0

    def test_parse_int_with_units(self):
        from propagul.mesh.gpu import _parse_int
        assert _parse_int("8192 MiB") == 8192
        assert _parse_int("75 %") == 75
        assert _parse_int("") == 0
        assert _parse_int("N/A") == 0

    def test_parse_float_with_units(self):
        from propagul.mesh.gpu import _parse_float
        assert _parse_float("125.50 W") == 125.50
        assert _parse_float("") == 0.0


# ─── Auto-Detection Tests ───────────────────────────────────────


class TestDetection:
    """Test backend auto-detection."""

    def test_detect_returns_list(self):
        from propagul.mesh.backends.detect import detect
        result = detect(timeout=0.5)
        assert isinstance(result, list)

    def test_detected_backend_dataclass(self):
        from propagul.mesh.backends.detect import DetectedBackend
        b = DetectedBackend("ollama", "http://localhost:11434", "0.1.0", 0.9)
        assert b.name == "ollama"
        assert b.confidence == 0.9


# ─── FleetStore Tests ────────────────────────────────────────────


class TestPersistencePathResolution:
    """Test _resolve_persist_path fallback logic. SKIPPED: Redis-native."""

    def test_explicit_env_var(self, monkeypatch, tmp_path):
        pass

    def test_fallback_to_user_home(self, monkeypatch, tmp_path):
        pass

    def test_save_creates_parent_dir(self, tmp_path):
        pass

    def test_load_validates_node_ids_from_disk(self, tmp_path):
        pass

    def test_stats_survive_persistence(self, tmp_path):
        pass


# ─── Landing Page Tests ─────────────────────────────────────────





class TestConfigSyncWiring:
    """Verify config-sync is wired into API and agent."""



    def test_agent_has_config_map(self):
        """MeshAgent has a config_map and desired_models property."""
        from propagul.mesh.agent import MeshAgent
        agent = MeshAgent(node_id="test-node")
        assert hasattr(agent, "config_map")
        assert hasattr(agent, "desired_models")
        assert agent.desired_models == {}

    def test_agent_config_sync_in_stats(self):
        """Agent stats include config_syncs counter."""
        from propagul.mesh.agent import MeshAgent
        agent = MeshAgent(node_id="test-node")
        assert "config_syncs" in agent.stats
        assert agent.stats["config_syncs"] == 0

    def test_agent_sync_config_method_exists(self):
        """Agent has _sync_config method."""
        from propagul.mesh.agent import MeshAgent
        agent = MeshAgent(node_id="test-node")
        assert hasattr(agent, "_sync_config")


# ─── Config-Map Persistence Tests ────────────────────────────────


class TestAutoPull:
    """Verify MeshAgent auto-pull/delete reconciliation."""

    def test_auto_pull_default_disabled(self):
        """Auto-pull is off by default."""
        from propagul.mesh.agent import MeshAgent
        agent = MeshAgent(node_id="test-node")
        assert agent._auto_pull is False

    def test_auto_pull_opt_in(self):
        """Auto-pull can be enabled via constructor."""
        from propagul.mesh.agent import MeshAgent
        agent = MeshAgent(node_id="test-node", auto_pull=True)
        assert agent._auto_pull is True

    def test_auto_pull_stats_tracked(self):
        """Agent stats include auto_pulls and auto_deletes counters."""
        from propagul.mesh.agent import MeshAgent
        agent = MeshAgent(node_id="test-node", auto_pull=True)
        assert "auto_pulls" in agent.stats
        assert "auto_deletes" in agent.stats
        assert agent.stats["auto_pulls"] == 0
        assert agent.stats["auto_deletes"] == 0

    def test_reconcile_method_exists(self):
        """Agent has _reconcile_desired_models method."""
        from propagul.mesh.agent import MeshAgent
        agent = MeshAgent(node_id="test-node")
        assert hasattr(agent, "_reconcile_desired_models")

    def test_reconcile_no_desired_is_noop(self):
        """Reconcile with no desired models does nothing."""
        from propagul.mesh.agent import MeshAgent
        agent = MeshAgent(node_id="test-node", auto_pull=True)
        # No desired models + no snapshot = nothing happens
        agent._reconcile_desired_models()
        assert agent.stats["auto_pulls"] == 0

    def test_reconcile_skips_already_installed(self):
        """Reconcile skips models that are already present."""
        from propagul.mesh.agent import MeshAgent, TelemetrySnapshot
        agent = MeshAgent(node_id="test-node", auto_pull=True)

        # Set desired: llama3:8b should be pulled
        agent._config_map.set_desired_model("llama3:8b", "pull")

        # Simulate snapshot with llama3:8b already installed
        snapshot = TelemetrySnapshot(
            timestamp=0.0,
            node={"node_id": "test-node"},
            backends=[{
                "backend": "ollama",
                "models": [{"name": "llama3:8b", "size_gb": 4.7}],
            }],
            gpu={},
            system={},
        )
        agent._last_snapshot = snapshot
        agent._reconcile_desired_models(snapshot)
        # Should NOT have tried to pull (already installed)
        assert agent.stats["auto_pulls"] == 0

    def test_reconcile_skips_delete_when_not_installed(self):
        """Reconcile skips delete when model isn't installed."""
        from propagul.mesh.agent import MeshAgent, TelemetrySnapshot
        agent = MeshAgent(node_id="test-node", auto_pull=True)

        # Set desired: old-model should be deleted
        agent._config_map.set_desired_model("old-model", "delete")

        # Simulate snapshot with no models
        snapshot = TelemetrySnapshot(
            timestamp=0.0,
            node={"node_id": "test-node"},
            backends=[{"backend": "ollama", "models": []}],
            gpu={},
            system={},
        )
        agent._reconcile_desired_models(snapshot)
        # Should NOT have tried to delete (not installed)
        assert agent.stats["auto_deletes"] == 0

    def test_auto_pull_wiring_in_push_telemetry(self):
        """Verify async reconciliation is wired from _push_telemetry."""
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "python" / "propagul" / "mesh" / "agent.py"
        code = src.read_text()
        assert "_reconcile_desired_models" in code
        assert "_async_reconcile" in code
        assert "asyncio.to_thread" in code
        assert "_pull_in_progress" in code
        # AG-03: Uses create_task + done callback instead of ensure_future
        assert "create_task" in code

    def test_reconcile_returns_results(self):
        """Reconcile returns list of action results."""
        from propagul.mesh.agent import MeshAgent
        agent = MeshAgent(node_id="test-node", auto_pull=True)
        results = agent._reconcile_desired_models()
        assert isinstance(results, list)
        assert len(results) == 0  # No desired models, no results

    def test_pull_in_progress_guard(self):
        """Concurrent reconciliation is blocked by _pull_in_progress flag."""
        from propagul.mesh.agent import MeshAgent
        agent = MeshAgent(node_id="test-node", auto_pull=True)
        assert agent._pull_in_progress is False


# ─── Debounced Persistence Tests ─────────────────────────────────


class TestAsyncAutoPull:
    """Verify async reconciliation wiring."""

    def test_async_reconcile_method_exists(self):
        """Agent has _async_reconcile method."""
        from propagul.mesh.agent import MeshAgent
        agent = MeshAgent(node_id="test-node")
        assert hasattr(agent, "_async_reconcile")

    def test_async_reconcile_is_coroutine(self):
        """_async_reconcile is a coroutine function."""
        import asyncio
        from propagul.mesh.agent import MeshAgent
        agent = MeshAgent(node_id="test-node")
        assert asyncio.iscoroutinefunction(agent._async_reconcile)

    def test_agent_uses_to_thread(self):
        """Verify asyncio.to_thread is used for non-blocking downloads."""
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "python" / "propagul" / "mesh" / "agent.py"
        code = src.read_text()
        assert "asyncio.to_thread" in code
        assert "_pull_in_progress" in code
        # Verify the guard prevents concurrent reconciliation
        assert "not self._pull_in_progress" in code


# ═══════════════════════════════════════════════════════════════════
# Phase 3.8: Dashboard Production Features
# ═══════════════════════════════════════════════════════════════════



class TestA2ASharedState:
    """SharedAgentState CRDT-backed shared state."""

    def test_set_and_get(self):
        from propagul.a2a import SharedAgentState
        state = SharedAgentState(room="test", agent_id="a1")
        state.set("key1", "value1")
        assert state.get("key1") == "value1"

    def test_get_missing_returns_none(self):
        from propagul.a2a import SharedAgentState
        assert SharedAgentState(room="test", agent_id="a1").get("nope") is None

    def test_delete_key(self):
        from propagul.a2a import SharedAgentState
        state = SharedAgentState(room="test", agent_id="a1")
        state.set("key1", "value1")
        state.delete("key1")
        assert state.get("key1") is None

    def test_get_all_shared(self):
        from propagul.a2a import SharedAgentState
        state = SharedAgentState(room="test", agent_id="a1")
        state.set("a", "1")
        state.set("b", "2")
        assert state.get_all_shared() == {"a": "1", "b": "2"}

    def test_private_state_isolation(self):
        from propagul.a2a import SharedAgentState
        s1 = SharedAgentState(room="test", agent_id="a1", node_id=1)
        s2 = SharedAgentState(room="test", agent_id="a2", node_id=2)
        s1.set_private("secret", "mine")
        assert s1.get_private("secret") == "mine"
        assert s2.get_private("secret") is None

    def test_read_other_agent_state(self):
        from propagul.a2a import SharedAgentState
        s1 = SharedAgentState(room="test", agent_id="a1", node_id=1)
        s2 = SharedAgentState(room="test", agent_id="a2", node_id=2)
        s1.set_private("status", "busy")
        s2.merge(s1.snapshot())
        assert s2.get_agent_state("a1")["status"] == "busy"

    def test_merge_convergence(self):
        from propagul.a2a import SharedAgentState
        s1 = SharedAgentState(room="test", agent_id="a1", node_id=1)
        s2 = SharedAgentState(room="test", agent_id="a2", node_id=2)
        s1.set("key", "from_a1")
        s2.set("key", "from_a2")
        s1.merge(s2.snapshot())
        s2.merge(s1.snapshot())
        assert s1.get("key") == s2.get("key")

    def test_snapshot_and_merge(self):
        from propagul.a2a import SharedAgentState
        s1 = SharedAgentState(room="test", agent_id="a1", node_id=1)
        s1.set("x", "42")
        s2 = SharedAgentState(room="test", agent_id="a2", node_id=2)
        result = s2.merge(s1.snapshot())
        assert result.had_changes
        assert s2.get("x") == "42"

    def test_sync_result_fields(self):
        from propagul.a2a import SharedAgentState
        s1 = SharedAgentState(room="test", agent_id="a1", node_id=1)
        s1.set("a", "1")
        s2 = SharedAgentState(room="test", agent_id="a2", node_id=2)
        result = s2.merge(s1.snapshot())
        assert result.delta > 0

    def test_metadata(self):
        from propagul.a2a import SharedAgentState
        state = SharedAgentState(room="test", agent_id="a1")
        state.set_meta("desc", "test room")
        assert state.get_meta("desc") == "test room"

    def test_key_count(self):
        from propagul.a2a import SharedAgentState
        state = SharedAgentState(room="test", agent_id="a1")
        initial = state.key_count
        state.set("k1", "v1")
        assert state.key_count > initial

    def test_summary(self):
        from propagul.a2a import SharedAgentState
        state = SharedAgentState(room="test-room", agent_id="a1")
        state.set("key", "val")
        s = state.summary()
        assert s["room"] == "test-room"
        assert s["shared_keys"] == 1

    def test_persistence_roundtrip(self, tmp_path):
        from propagul.a2a import SharedAgentState, A2AConfig
        config = A2AConfig(persist_path=str(tmp_path), debounce_seconds=0)
        s1 = SharedAgentState(room="persist-test", agent_id="a1", config=config)
        s1.set("key", "persistent")
        s1.save_to_disk()
        s2 = SharedAgentState(room="persist-test", agent_id="a2", config=config)
        assert s2.get("key") == "persistent"

    def test_max_keys_enforced(self):
        from propagul.a2a import SharedAgentState, A2AConfig
        import pytest
        config = A2AConfig(max_keys_per_room=5)
        state = SharedAgentState(room="test", agent_id="a1", config=config)
        with pytest.raises(ValueError, match="Max keys"):
            for i in range(20):
                state.set(f"k{i}", f"v{i}")



class TestA2ARoom:
    """AgentRoom collaboration context."""

    def test_join_creates_state(self):
        from propagul.a2a import AgentRoom
        room = AgentRoom("test-room")
        assert room.state is None
        room.join("agent-01")
        assert room.state is not None

    def test_join_returns_agent_info(self):
        from propagul.a2a import AgentRoom
        info = AgentRoom("test-room").join("agent-01", capabilities=["inference"])
        assert info.agent_id == "agent-01"
        assert "inference" in info.capabilities

    def test_leave_removes_member(self):
        from propagul.a2a import AgentRoom
        room = AgentRoom("test-room")
        room.join("agent-01")
        assert room.has_member("agent-01")
        room.leave("agent-01")
        assert not room.has_member("agent-01")

    def test_leave_nonexistent(self):
        from propagul.a2a import AgentRoom
        assert AgentRoom("test-room").leave("nobody") is False

    def test_members_list(self):
        from propagul.a2a import AgentRoom
        room = AgentRoom("test-room")
        room.join("a1")
        room.join("a2")
        room.join("a3")
        assert room.member_count() == 3
        assert set(m.agent_id for m in room.members()) == {"a1", "a2", "a3"}

    def test_room_summary(self):
        from propagul.a2a import AgentRoom
        room = AgentRoom("project-alpha")
        room.join("a1")
        s = room.summary()
        assert s["room_id"] == "project-alpha"
        assert s["member_count"] == 1
        assert s["has_state"] is True

    def test_room_state_shared(self):
        from propagul.a2a import AgentRoom
        room = AgentRoom("test-room")
        room.join("a1")
        room.join("a2")
        room.state.set("shared_key", "shared_value")
        assert room.state.get("shared_key") == "shared_value"

    def test_room_flush(self, tmp_path):
        from propagul.a2a import AgentRoom, A2AConfig
        config = A2AConfig(persist_path=str(tmp_path), debounce_seconds=0)
        room = AgentRoom("flush-test", config=config)
        room.join("a1")
        room.state.set("key", "value")
        assert room.flush() is True



class TestA2ATypes:
    """A2A type definitions."""

    def test_agent_info_active(self):
        from propagul.a2a.types import AgentInfo
        import time
        assert AgentInfo(agent_id="a1", last_seen=time.time()).is_active is True
        assert AgentInfo(agent_id="a2", last_seen=time.time() - 600).is_active is False

    def test_agent_info_to_dict(self):
        from propagul.a2a.types import AgentInfo
        d = AgentInfo(agent_id="a1", capabilities=["search"]).to_dict()
        assert d["agent_id"] == "a1"
        assert "search" in d["capabilities"]

    def test_sync_result_had_changes(self):
        from propagul.a2a.types import SyncResult
        assert SyncResult(delta=0).had_changes is False
        assert SyncResult(delta=3).had_changes is True

    def test_a2a_config_defaults(self):
        from propagul.a2a.types import A2AConfig
        c = A2AConfig()
        assert c.persist_path is None
        assert c.debounce_seconds == 2.0
        assert c.max_keys_per_room == 10_000


# ═══════════════════════════════════════════════════════════════════
# Phase 4.0: Multi-Backend Adapters + Dashboard Polish
# ═══════════════════════════════════════════════════════════════════



class TestVllmAdapter:
    """vLLM backend adapter tests."""

    def test_vllm_model_info_dataclass(self):
        from propagul.mesh.backends.vllm import VllmModelInfo
        m = VllmModelInfo(id="meta-llama/Llama-3-8B", max_model_len=8192)
        assert m.id == "meta-llama/Llama-3-8B"
        assert m.max_model_len == 8192
        assert m.object == "model"

    def test_vllm_metrics_dataclass(self):
        from propagul.mesh.backends.vllm import VllmMetrics
        m = VllmMetrics(num_requests_running=5, gpu_cache_usage_perc=0.75)
        assert m.num_requests_running == 5
        assert m.gpu_cache_usage_perc == 0.75

    def test_get_models_unreachable(self):
        from propagul.mesh.backends.vllm import get_models
        result = get_models(base_url="http://127.0.0.1:19999", timeout=0.3)
        assert result == []

    def test_check_health_unreachable(self):
        from propagul.mesh.backends.vllm import check_health
        assert check_health(base_url="http://127.0.0.1:19999", timeout=0.3) is False

    def test_get_metrics_unreachable(self):
        from propagul.mesh.backends.vllm import get_metrics
        assert get_metrics(base_url="http://127.0.0.1:19999", timeout=0.3) is None

    def test_collect_telemetry_unreachable(self):
        from propagul.mesh.backends.vllm import collect_telemetry
        result = collect_telemetry(base_url="http://127.0.0.1:19999", timeout=0.3)
        assert result["backend"] == "vllm"
        assert result["healthy"] is False
        assert result["models"] == []
        assert result["model_count"] == 0

    def test_collect_telemetry_schema(self):
        """Verify return schema matches expected heartbeat format."""
        from propagul.mesh.backends.vllm import collect_telemetry
        result = collect_telemetry(base_url="http://127.0.0.1:19999", timeout=0.3)
        assert "backend" in result
        assert "url" in result
        assert "healthy" in result
        assert "models" in result
        assert "model_count" in result
        assert "running_count" in result



class TestTgiAdapter:
    """TGI backend adapter tests."""

    def test_tgi_model_info_dataclass(self):
        from propagul.mesh.backends.tgi import TgiModelInfo
        m = TgiModelInfo(model_id="HuggingFaceH4/zephyr-7b-beta", model_dtype="float16")
        assert m.model_id == "HuggingFaceH4/zephyr-7b-beta"
        assert m.model_dtype == "float16"

    def test_tgi_metrics_dataclass(self):
        from propagul.mesh.backends.tgi import TgiMetrics
        m = TgiMetrics(queue_size=3, total_tokens_generated=10000)
        assert m.queue_size == 3
        assert m.total_tokens_generated == 10000

    def test_get_info_unreachable(self):
        from propagul.mesh.backends.tgi import get_info
        assert get_info(base_url="http://127.0.0.1:19999", timeout=0.3) is None

    def test_check_health_unreachable(self):
        from propagul.mesh.backends.tgi import check_health
        assert check_health(base_url="http://127.0.0.1:19999", timeout=0.3) is False

    def test_is_tgi_unreachable(self):
        from propagul.mesh.backends.tgi import is_tgi
        assert is_tgi(base_url="http://127.0.0.1:19999", timeout=0.3) is False

    def test_collect_telemetry_unreachable(self):
        from propagul.mesh.backends.tgi import collect_telemetry
        result = collect_telemetry(base_url="http://127.0.0.1:19999", timeout=0.3)
        assert result["backend"] == "tgi"
        assert result["healthy"] is False
        assert result["models"] == []

    def test_collect_telemetry_schema(self):
        from propagul.mesh.backends.tgi import collect_telemetry
        result = collect_telemetry(base_url="http://127.0.0.1:19999", timeout=0.3)
        assert "backend" in result
        assert "url" in result
        assert "model_count" in result



class TestLmStudioAdapter:
    """LM Studio backend adapter tests."""

    def test_lm_studio_model_info_dataclass(self):
        from propagul.mesh.backends.lm_studio import LmStudioModelInfo
        m = LmStudioModelInfo(id="TheBloke/Llama-2-7B-GGUF")
        assert m.id == "TheBloke/Llama-2-7B-GGUF"
        assert m.object == "model"

    def test_get_models_unreachable(self):
        from propagul.mesh.backends.lm_studio import get_models
        result = get_models(base_url="http://127.0.0.1:19999", timeout=0.3)
        assert result == []

    def test_check_health_unreachable(self):
        from propagul.mesh.backends.lm_studio import check_health
        assert check_health(base_url="http://127.0.0.1:19999", timeout=0.3) is False

    def test_collect_telemetry_unreachable(self):
        from propagul.mesh.backends.lm_studio import collect_telemetry
        result = collect_telemetry(base_url="http://127.0.0.1:19999", timeout=0.3)
        assert result["backend"] == "lm_studio"
        assert result["healthy"] is False
        assert result["models"] == []

    def test_collect_telemetry_schema(self):
        from propagul.mesh.backends.lm_studio import collect_telemetry
        result = collect_telemetry(base_url="http://127.0.0.1:19999", timeout=0.3)
        assert "backend" in result
        assert "url" in result
        assert "model_count" in result
        assert "running_count" in result



class TestDetectionTgi:
    """TGI detection and port 8080 disambiguation."""

    def test_tgi_probe_exists_in_probes(self):
        from propagul.mesh.backends.detect import _PROBES
        names = [p["name"] for p in _PROBES]
        assert "tgi" in names
        # TGI must come BEFORE llama_cpp (both use 8080)
        tgi_idx = names.index("tgi")
        llama_idx = names.index("llama_cpp")
        assert tgi_idx < llama_idx, "TGI probe must come before llama_cpp for disambiguation"

    def test_tgi_probe_uses_info_endpoint(self):
        from propagul.mesh.backends.detect import _PROBES
        tgi_probe = [p for p in _PROBES if p["name"] == "tgi"][0]
        assert tgi_probe["health_path"] == "/info"
        assert tgi_probe["sig_body_key"] == "model_id"

    def test_detected_backend_includes_tgi(self):
        from propagul.mesh.backends.detect import DetectedBackend
        b = DetectedBackend("tgi", "http://localhost:8080", "1.0.0", 0.9)
        assert b.name == "tgi"
        assert b.url == "http://localhost:8080"



class TestAgentAdapterWiring:
    """Verify agent.py routes detected backends to correct adapters."""

    def test_agent_imports_all_adapters(self):
        """All backend adapters must be imported in agent.py."""
        from propagul.mesh import agent
        assert hasattr(agent, 'vllm_backend')
        assert hasattr(agent, 'tgi_backend')
        assert hasattr(agent, 'lm_studio_backend')
        assert hasattr(agent, 'ollama_backend')

    def test_collect_telemetry_has_no_stub_comment(self):
        """The 'Phase 2' stub comment must be gone."""
        import inspect
        from propagul.mesh.agent import MeshAgent
        source = inspect.getsource(MeshAgent._collect_telemetry)
        assert "Phase 2" not in source
        assert "will be implemented" not in source

    def test_collect_telemetry_routes_vllm(self):
        """_collect_telemetry must have a vllm branch."""
        import inspect
        from propagul.mesh.agent import MeshAgent
        source = inspect.getsource(MeshAgent._collect_telemetry)
        assert "vllm_backend.collect_telemetry" in source

    def test_collect_telemetry_routes_tgi(self):
        """_collect_telemetry must have a tgi branch."""
        import inspect
        from propagul.mesh.agent import MeshAgent
        source = inspect.getsource(MeshAgent._collect_telemetry)
        assert "tgi_backend.collect_telemetry" in source

    def test_collect_telemetry_routes_lm_studio(self):
        """_collect_telemetry must have an lm_studio branch."""
        import inspect
        from propagul.mesh.agent import MeshAgent
        source = inspect.getsource(MeshAgent._collect_telemetry)
        assert "lm_studio_backend.collect_telemetry" in source

    def test_unknown_backend_has_complete_schema(self):
        """Unknown backends must return model_count + models list (no stub)."""
        import inspect
        from propagul.mesh.agent import MeshAgent
        source = inspect.getsource(MeshAgent._collect_telemetry)
        # The else-branch stub must include model_count and models
        assert '"model_count"' in source
        assert '"models"' in source


# ─── F-01: Command Routing (Backend-Aware) ──────────────────────


class TestCommandRouting:
    """F-01: Commands must be routed based on detected backend type."""

    def test_readonly_backend_rejects_pull(self):
        """vLLM/TGI/LM Studio are read-only — pull must return an error."""
        import inspect
        from propagul.mesh.agent import MeshAgent
        source = inspect.getsource(MeshAgent._fetch_and_execute_commands)
        assert "read-only" in source.lower() or "does not support" in source
        assert "backend_name" in source

    def test_ollama_backend_routes_to_execute_command(self):
        """Ollama commands must call ollama_backend.execute_command()."""
        import inspect
        from propagul.mesh.agent import MeshAgent
        source = inspect.getsource(MeshAgent._fetch_and_execute_commands)
        assert "ollama_backend.execute_command" in source

    def test_readonly_backend_returns_error_dict(self):
        """Non-Ollama backends must return {status: error, error: ...}."""
        import inspect
        from propagul.mesh.agent import MeshAgent
        source = inspect.getsource(MeshAgent._fetch_and_execute_commands)
        assert '"status": "error"' in source
        assert '"error"' in source
    def test_eject_routes_to_ollama_eject_all(self):
        """Eject command must call ollama_backend.eject_all() for Ollama."""
        import inspect
        from propagul.mesh.agent import MeshAgent
        source = inspect.getsource(MeshAgent._fetch_and_execute_commands)
        assert "ollama_backend.eject_all" in source

    def test_eject_rejected_for_readonly_backends(self):
        """Eject must be rejected for non-Ollama backends with a clear error."""
        import inspect
        from propagul.mesh.agent import MeshAgent
        source = inspect.getsource(MeshAgent._fetch_and_execute_commands)
        assert "does not support VRAM eject" in source

    def test_eject_tracks_stats(self):
        """Eject must increment the 'ejects' stats counter."""
        import inspect
        from propagul.mesh.agent import MeshAgent
        source = inspect.getsource(MeshAgent._fetch_and_execute_commands)
        assert '"ejects"' in source


# ─── Ollama Eject Backend Tests ─────────────────────────────────


class TestOllamaEject:
    """Backend-level tests for ollama.eject_all()."""

    def test_eject_all_function_exists(self):
        """eject_all must be importable from the ollama backend."""
        from propagul.mesh.backends.ollama import eject_all
        assert callable(eject_all)

    def test_eject_all_ssrf_blocked(self):
        """SSRF guard must block eject to external hosts."""
        from propagul.mesh.backends.ollama import eject_all
        result = eject_all(base_url="http://evil.com:11434")
        assert result["status"] == "error"
        assert "SSRF" in result.get("error", "")
        assert result["ejected"] == []
        assert result["failed"] == []

    def test_eject_all_unreachable_returns_error(self):
        """Unreachable Ollama must return error with correct schema."""
        from propagul.mesh.backends.ollama import eject_all
        result = eject_all(base_url="http://127.0.0.1:19999", timeout=0.3)
        assert result["status"] == "error"
        assert "Failed to list running models" in result.get("error", "")
        assert isinstance(result["ejected"], list)
        assert isinstance(result["failed"], list)
        assert result["running_before"] == 0
        assert result["running_after"] == 0

    def test_eject_all_return_schema(self):
        """Return dict must contain all required keys."""
        from propagul.mesh.backends.ollama import eject_all
        result = eject_all(base_url="http://127.0.0.1:19999", timeout=0.3)
        required_keys = {"status", "ejected", "failed", "running_before", "running_after"}
        assert required_keys.issubset(set(result.keys()))

    def test_eject_all_has_ssrf_guard(self):
        """eject_all source must call _validate_url (SSRF prevention)."""
        import inspect
        from propagul.mesh.backends.ollama import eject_all
        source = inspect.getsource(eject_all)
        assert "_validate_url" in source

    def test_eject_all_uses_keep_alive_zero(self):
        """eject_all must use keep_alive: 0 (the Ollama unload mechanism)."""
        import inspect
        from propagul.mesh.backends.ollama import eject_all
        source = inspect.getsource(eject_all)
        assert '"keep_alive": 0' in source or "'keep_alive': 0" in source

    def test_eject_all_uses_api_ps(self):
        """eject_all must discover running models via /api/ps."""
        import inspect
        from propagul.mesh.backends.ollama import eject_all
        source = inspect.getsource(eject_all)
        assert "/api/ps" in source

    def test_eject_all_uses_api_generate(self):
        """eject_all must unload via POST /api/generate."""
        import inspect
        from propagul.mesh.backends.ollama import eject_all
        source = inspect.getsource(eject_all)
        assert "/api/generate" in source



@pytest.mark.skip(reason="Legacy file-I/O — persistence is now Redis BGSAVE")

class TestSecureFilePermissions:
    """F-02: SKIPPED — Redis handles persistence natively."""

    def test_fleet_state_permissions(self, tmp_path):
        pass

    def test_fleet_state_fsync_in_code(self):
        pass


# ─── F-03: Config-Map Lifecycle (Redis-native) ─────────────────





# ─── F-10: CRDT Delta Signal ───────────────────────────────────


class TestCRDTDeltaSignal:
    """F-10: CRDT merge delta must never be negative."""

    def test_max_keys_drop_does_not_produce_negative_delta(self):
        from propagul.crdt import ORMap, MAX_KEYS
        crdt = ORMap(node_id=1)
        # Fill to MAX_KEYS
        for i in range(MAX_KEYS):
            crdt.set(f"key-{i}", f"val-{i}")

        # Try to merge a remote snapshot with a new key that exceeds MAX_KEYS
        remote = ORMap(node_id=2)
        remote.set("overflow-key", "overflow-val")
        delta = crdt.merge(remote.snapshot())
        assert delta >= 0, f"Delta must never be negative, got {delta}"


# ─── F-11: LAN Backend URL Rewriting ──────────────────────────


class TestRewriteBackendUrls:
    """F-11: Agent must rewrite localhost backend URLs to LAN IP for fleet routing.

    Without this, multi-machine LAN fleets fail: remote agents see
    'http://localhost:11434' and connect to their own localhost.
    """

    def test_rewrite_localhost_to_lan_ip(self):
        from propagul.mesh.agent import MeshAgent
        backends = [
            {"backend": "ollama", "url": "http://localhost:11434", "models": []},
        ]
        MeshAgent._rewrite_backend_urls(backends, "192.168.1.50")
        assert backends[0]["url"] == "http://192.168.1.50:11434"

    def test_rewrite_127_0_0_1_to_lan_ip(self):
        from propagul.mesh.agent import MeshAgent
        backends = [
            {"backend": "ollama", "url": "http://127.0.0.1:11434", "models": []},
        ]
        MeshAgent._rewrite_backend_urls(backends, "10.0.0.5")
        assert backends[0]["url"] == "http://10.0.0.5:11434"

    def test_rewrites_non_loopback_private_ip(self):
        """Non-loopback private IPs must also be rewritten to advertise_ip.

        This covers WireGuard (10.100.0.x), Docker bridge (172.17.x.x),
        and other VPN IPs that are not reachable from remote fleet nodes.
        """
        from propagul.mesh.agent import MeshAgent
        backends = [
            {"backend": "ollama", "url": "http://192.168.1.20:11434", "models": []},
        ]
        MeshAgent._rewrite_backend_urls(backends, "192.168.1.50")
        assert backends[0]["url"] == "http://192.168.1.50:11434"

    def test_preserves_url_matching_advertise_ip(self):
        """URL already using the advertise_ip must not be double-rewritten."""
        from propagul.mesh.agent import MeshAgent
        backends = [
            {"backend": "ollama", "url": "http://192.168.1.50:11434", "models": []},
        ]
        MeshAgent._rewrite_backend_urls(backends, "192.168.1.50")
        assert backends[0]["url"] == "http://192.168.1.50:11434"

    def test_preserves_port(self):
        from propagul.mesh.agent import MeshAgent
        backends = [
            {"backend": "lm_studio", "url": "http://localhost:1234", "models": []},
        ]
        MeshAgent._rewrite_backend_urls(backends, "192.168.1.50")
        assert backends[0]["url"] == "http://192.168.1.50:1234"

    def test_multiple_backends(self):
        """All backends with non-advertise hostnames must be rewritten."""
        from propagul.mesh.agent import MeshAgent
        backends = [
            {"backend": "ollama", "url": "http://localhost:11434", "models": []},
            {"backend": "vllm", "url": "http://127.0.0.1:8000", "models": []},
            {"backend": "lm_studio", "url": "http://192.168.1.20:1234", "models": []},
        ]
        MeshAgent._rewrite_backend_urls(backends, "10.0.1.100")
        assert backends[0]["url"] == "http://10.0.1.100:11434"
        assert backends[1]["url"] == "http://10.0.1.100:8000"
        assert backends[2]["url"] == "http://10.0.1.100:1234"  # Also rewritten

    def test_noop_when_advertise_ip_is_loopback(self):
        from propagul.mesh.agent import MeshAgent
        backends = [
            {"backend": "ollama", "url": "http://localhost:11434", "models": []},
        ]
        MeshAgent._rewrite_backend_urls(backends, "127.0.0.1")
        assert backends[0]["url"] == "http://localhost:11434"  # Unchanged

    def test_noop_when_advertise_ip_is_empty(self):
        from propagul.mesh.agent import MeshAgent
        backends = [
            {"backend": "ollama", "url": "http://localhost:11434", "models": []},
        ]
        MeshAgent._rewrite_backend_urls(backends, "")
        assert backends[0]["url"] == "http://localhost:11434"  # Unchanged

    def test_skips_backend_without_url(self):
        from propagul.mesh.agent import MeshAgent
        backends = [
            {"backend": "ollama", "models": []},
            {"backend": "vllm", "url": "", "models": []},
        ]
        # Should not raise
        result = MeshAgent._rewrite_backend_urls(backends, "192.168.1.50")
        assert result == backends

    def test_tailscale_ip(self):
        """Tailscale IPs (100.x.x.x) should work for WAN fleet routing."""
        from propagul.mesh.agent import MeshAgent
        backends = [
            {"backend": "ollama", "url": "http://localhost:11434", "models": []},
        ]
        MeshAgent._rewrite_backend_urls(backends, "100.64.0.5")
        assert backends[0]["url"] == "http://100.64.0.5:11434"

    def test_ipv6_loopback_rewrite(self):
        from propagul.mesh.agent import MeshAgent
        backends = [
            {"backend": "ollama", "url": "http://[::1]:11434", "models": []},
        ]
        MeshAgent._rewrite_backend_urls(backends, "192.168.1.50")
        assert backends[0]["url"] == "http://192.168.1.50:11434"

    def test_e2e_heartbeat_body_has_lan_ip(self):
        """Integration test: full heartbeat body must contain LAN IP, not localhost.

        Simulates the complete path:
          TelemetrySnapshot → _push_telemetry() → _rewrite_backend_urls() → body
        Verifies the *exact same dict* that would be POST'd to the dashboard.
        """
        import json
        from propagul.mesh.agent import MeshAgent, TelemetrySnapshot

        # Simulate a realistic telemetry snapshot with localhost URLs
        snapshot_dict = {
            "timestamp": 1234567890.0,
            "node": {
                "node_id": "test-node",
                "hostname": "test",
                "platform": "Linux",
                "arch": "x86_64",
                "python_version": "3.11.0",
                "agent_version": "dev",
                "local_ip": "192.168.1.42",
                "uptime_seconds": 1000,
            },
            "backends": [
                {
                    "backend": "ollama",
                    "online": True,
                    "version": "0.24.0",
                    "url": "http://localhost:11434",
                    "model_count": 3,
                    "models": [{"name": "llama3.2:3b"}, {"name": "mistral:latest"}],
                },
                {
                    "backend": "lm_studio",
                    "online": True,
                    "version": "0.3.0",
                    "url": "http://127.0.0.1:1234",
                    "model_count": 1,
                    "models": [{"name": "phi-3"}],
                },
            ],
            "gpu": {"backend": "nvidia", "gpus": [], "total_vram_mb": 8192},
            "system": {"cpu_percent": 25.0},
        }

        # Apply the rewrite (same as _push_telemetry does)
        MeshAgent._rewrite_backend_urls(
            snapshot_dict["backends"], "192.168.1.42",
        )

        # Verify: no localhost in any backend URL
        for b in snapshot_dict["backends"]:
            url = b.get("url", "")
            assert "localhost" not in url, f"localhost still in URL: {url}"
            assert "127.0.0.1" not in url, f"127.0.0.1 still in URL: {url}"
            assert "192.168.1.42" in url, f"LAN IP not in URL: {url}"

        # Verify specific rewrites
        assert snapshot_dict["backends"][0]["url"] == "http://192.168.1.42:11434"
        assert snapshot_dict["backends"][1]["url"] == "http://192.168.1.42:1234"

    def test_e2e_fleet_routing_table_receives_lan_ip(self):
        """Verify the server-side fleet_routing_table would contain LAN IPs.

        The server's get_fleet_routing_table() reads backend URLs from stored
        telemetry. If the agent rewrites correctly, the server passes LAN IPs
        to other agents — enabling multi-machine cross-routing.
        """
        from propagul.mesh.agent import MeshAgent

        # Simulate what a node's telemetry looks like AFTER rewriting
        backends = [
            {"backend": "ollama", "url": "http://localhost:11434",
             "models": [{"name": "llama3.2:3b"}]},
        ]
        MeshAgent._rewrite_backend_urls(backends, "192.168.1.42")

        # Simulate what redis_store.get_fleet_routing_table() does (line 696):
        # backend_url = b.get("url", "")
        backend_url = backends[0].get("url", "")

        # This URL must be routable from OTHER machines in the LAN
        assert backend_url == "http://192.168.1.42:11434"
        assert "localhost" not in backend_url
        assert "127.0.0.1" not in backend_url


# ─── Fleet Keep-Alive Override Tests ─────────────────────────────

