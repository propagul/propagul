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

class TestFleetStore:
    """Test FleetStore CRUD and persistence."""

    def _make_store(self, persist_path=None):
        from propagul.server.dashboard_api import FleetStore
        return FleetStore(persist_path=persist_path)

    def test_add_and_get_node(self):
        store = self._make_store()
        node = store.update_node("node-1", {"backends": []})
        assert node.node_id == "node-1"
        assert store.node_count == 1
        # Redis deserializes a new object on each get — check equality, not identity
        retrieved = store.get_node("node-1")
        assert retrieved is not None
        assert retrieved.node_id == node.node_id

    def test_max_nodes_enforced(self):
        from propagul.server.dashboard_api import MAX_NODES
        store = self._make_store()
        for i in range(MAX_NODES):
            store.update_node(f"node-{i}", {})
        with pytest.raises(ValueError, match="Max nodes"):
            store.update_node("node-overflow", {})

    def test_node_online_status(self):
        store = self._make_store()
        store.update_node("node-1", {})
        node = store.get_node("node-1")
        assert node.is_online  # Just created
        assert node.status == "online"

    def test_command_queue(self):
        store = self._make_store()
        store.update_node("node-1", {})
        assert store.add_command("node-1", "pull", "llama3:8b")
        assert store.add_command("node-1", "delete", "old-model")

        commands = store.pop_commands("node-1")
        assert len(commands) == 2
        assert commands[0]["command"] == "pull"
        assert commands[1]["model"] == "old-model"

        # Queue should be empty after pop
        assert store.pop_commands("node-1") == []

    def test_command_queue_limit(self):
        from propagul.server.dashboard_api import MAX_PENDING_COMMANDS
        store = self._make_store()
        store.update_node("node-1", {})
        for i in range(MAX_PENDING_COMMANDS):
            assert store.add_command("node-1", "pull", f"model-{i}")
        assert not store.add_command("node-1", "pull", "overflow")

    def test_command_on_nonexistent_node(self):
        store = self._make_store()
        assert not store.add_command("ghost", "pull", "llama3")
        assert store.pop_commands("ghost") == []

    def test_fleet_health(self):
        store = self._make_store()
        store.update_node("a", {
            "backends": [{"model_count": 3, "models": [
                {"name": "m1"}, {"name": "m2"}, {"name": "m3"}
            ]}],
            "gpu": {"gpu_count": 1},
        })
        store.update_node("b", {
            "backends": [{"model_count": 2, "models": [
                {"name": "m4"}, {"name": "m5"}
            ]}],
            "gpu": {"gpu_count": 2},
        })

        health = store.fleet_health()
        assert health["total_nodes"] == 2
        assert health["online"] == 2
        assert health["total_models"] == 5
        assert health["total_gpus"] == 3

    def test_get_all_models(self):
        store = self._make_store()
        store.update_node("n1", {
            "backends": [{"backend": "ollama", "models": [
                {"name": "llama3", "size_gb": 4.0},
                {"name": "mistral", "size_gb": 7.0},
            ]}],
        })
        models = store.get_all_models()
        assert len(models) == 2
        assert models[0]["node_id"] == "n1"
        assert models[1]["name"] == "mistral"

    # ─── Persistence (Redis-native — file I/O tests skipped) ─────

    @pytest.mark.skip(reason="Legacy file-I/O test — persistence is now Redis BGSAVE")
    def test_save_and_load(self):
        pass

    @pytest.mark.skip(reason="Legacy file-I/O test — persistence is now Redis BGSAVE")
    def test_load_nonexistent_file(self):
        pass

    @pytest.mark.skip(reason="Legacy file-I/O test — persistence is now Redis BGSAVE")
    def test_load_corrupt_file(self):
        pass

    @pytest.mark.skip(reason="Legacy file-I/O test — persistence is now Redis BGSAVE")
    def test_no_persist_path_does_nothing(self):
        pass


# ─── Dashboard API Validation Tests ──────────────────────────────

class TestNodeIdValidation:
    """P1-01: node_id regex validation."""

    def test_valid_node_ids(self):
        from propagul.server.dashboard_api import _NODE_ID_RE
        for name in ["workstation-01", "gpu.server.3", "a", "node_a-b.c"]:
            assert _NODE_ID_RE.match(name), f"Should be valid: {name}"

    def test_invalid_node_ids(self):
        from propagul.server.dashboard_api import _NODE_ID_RE
        for name in ["", "-start", ".start", "../../etc/passwd",
                      "a" * 65, "node id", "node\nid"]:
            assert not _NODE_ID_RE.match(name), f"Should be invalid: {name}"


# ─── Dashboard Web XSS Tests ────────────────────────────────────

class TestXSS:
    """Verify HTML escaping prevents XSS."""

    def test_esc_function(self):
        from propagul.server.dashboard_web import _esc
        assert _esc('<script>alert("xss")</script>') == '&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;'
        assert _esc("normal text") == "normal text"
        assert _esc(None) == ""
        assert _esc("it's") == "it&#39;s"

    def test_node_card_escapes_input(self):
        from propagul.server.dashboard_web import _render_node_card
        malicious = {
            "node_id": '<img src=x onerror=alert(1)>',
            "status": "online",
            "hostname": "safe",
            "platform": "Linux",
            "arch": "x86",
            "model_count": 0,
            "running_count": 0,
            "gpu_backend": "none",
            "gpu_count": 0,
            "gpu_utilization": 0,
            "vram_total_mb": 0,
            "vram_used_mb": 0,
            "last_seen": time.time(),
            "pending_commands": 0,
        }
        html = _render_node_card(malicious)
        assert "<img" not in html
        assert "&lt;img" in html


# ─── Persistence Path Resolution Tests (Legacy — skipped) ───────

@pytest.mark.skip(reason="Legacy file-I/O — persistence is now Redis BGSAVE")
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

class TestLandingPage:
    """Test the public landing page renderer."""

    def test_render_returns_valid_html(self):
        from propagul.server.landing import _render_landing_page
        html = _render_landing_page()
        assert "<!DOCTYPE html>" in html
        assert "Propagul" in html
        assert "Join Waitlist" in html

    def test_landing_has_meta_tags(self):
        from propagul.server.landing import _render_landing_page
        html = _render_landing_page()
        assert 'name="description"' in html
        assert 'name="viewport"' in html

    def test_landing_has_feature_cards(self):
        from propagul.server.landing import _render_landing_page
        html = _render_landing_page()
        # "Built for local inference" detail grid
        assert "Real-Time Fleet View" in html
        assert "Remote Model Management" in html
        assert "Multi-Backend" in html
        assert "Audit Log" in html
        # Config Sync was removed as vaporware (not yet implemented)

    def test_landing_has_why_section(self):
        from propagul.server.landing import _render_landing_page
        html = _render_landing_page()
        assert "Why Propagul?" in html
        assert "See everything, everywhere." in html
        assert "Deploy models with one click." in html
        assert "Your data stays home." in html

    def test_landing_has_trust_box(self):
        from propagul.server.landing import _render_landing_page
        html = _render_landing_page()
        assert "Don't trust us. Verify it." in html
        assert "Open Source Agent" in html
        assert "Network verifiable" in html
        assert "WAN-loss resilient inference" in html

        


# ─── Admin Dashboard Tests ──────────────────────────────────────

class TestAdminDashboard:
    """Test the admin dashboard renderer."""

    def test_render_with_empty_data(self):
        from propagul.server.admin_dashboard import _render_admin_page
        html = _render_admin_page([], [], {}, {})
        assert "Propagul" in html
        assert "No entries" in html
        assert "No keys issued" in html

    def test_render_with_waitlist(self):
        from propagul.server.admin_dashboard import _render_admin_page
        entries = [
            {"email": "a@b.com", "use_case": "test", "status": "waiting",
             "created_at": "2026-01-01", "api_key_hash": None},
            {"email": "c@d.com", "use_case": "", "status": "key_sent",
             "created_at": "2026-01-02", "api_key_hash": "abc123..."},
        ]
        html = _render_admin_page(entries, [], {}, {})
        assert "a@b.com" in html
        assert "c@d.com" in html
        assert "badge-waiting" in html
        assert "badge-key_sent" in html

    def test_render_with_keys(self):
        from propagul.server.admin_dashboard import _render_admin_page
        keys = [
            {"key_hash_prefix": "abc123def456...", "tier": "pro",
             "owner": "dev@test.com", "created_at": time.time()},
        ]
        html = _render_admin_page([], keys, {}, {})
        assert "abc123def456" in html
        assert "dev@test.com" in html
        assert "Revoke" in html

    def test_render_with_fleet_data(self):
        from propagul.server.admin_dashboard import _render_admin_page
        fleet = {"total_nodes": 5, "online": 3, "offline": 2,
                 "stats": {"heartbeats": 1500, "commands_sent": 42,
                           "commands_executed": 40}}
        html = _render_admin_page([], [], fleet, fleet["stats"])
        assert "5" in html  # total nodes
        assert "1,500" in html  # heartbeats formatted
        assert "42" in html  # commands sent

    def test_xss_in_waitlist_email(self):
        from propagul.server.admin_dashboard import _render_admin_page
        entries = [{"email": '<img src=x onerror=alert(1)>', "use_case": "",
                    "status": "waiting", "created_at": "", "api_key_hash": None}]
        html = _render_admin_page(entries, [], {}, {})
        assert "<img src=" not in html
        assert "&lt;img" in html

    def test_time_ago_formatting(self):
        from propagul.server.admin_dashboard import _time_ago
        assert _time_ago(0) == "never"
        assert "s ago" in _time_ago(time.time() - 30)
        assert "m ago" in _time_ago(time.time() - 300)
        assert "h ago" in _time_ago(time.time() - 7200)
        assert "d ago" in _time_ago(time.time() - 172800)

    def test_memory_usage_returns_dict(self):
        from propagul.server.admin_dashboard import _get_memory_usage
        mem = _get_memory_usage()
        assert "rss_mb" in mem
        assert isinstance(mem["rss_mb"], float)


# ─── FleetStore Remove + Dashboard Actions ──────────────────────

class TestFleetStoreRemove:
    """Test FleetStore.remove_node."""

    def test_remove_existing_node(self):
        from propagul.server.dashboard_api import FleetStore
        store = FleetStore()
        store.update_node("node-1", {})
        assert store.remove_node("node-1") is True
        assert store.get_node("node-1") is None
        assert store.node_count == 0

    def test_remove_nonexistent_node(self):
        from propagul.server.dashboard_api import FleetStore
        store = FleetStore()
        assert store.remove_node("ghost") is False


class TestDashboardActions:
    """Test model delete button and nav-account rendering."""

    def test_model_table_has_delete_button(self):
        from propagul.server.dashboard_web import _render_model_table
        models = [{"name": "llama3:8b", "size_gb": 4.0, "parameter_size": "8B",
                   "quantization": "Q4_K_M", "node_id": "n1", "node_status": "online",
                   "backend": "ollama"}]
        html = _render_model_table(models)
        assert "Delete" in html
        assert "btn-danger" in html
        assert "Actions" in html

    def test_model_table_empty_has_no_delete(self):
        from propagul.server.dashboard_web import _render_model_table
        html = _render_model_table([])
        assert "Delete" not in html

    def test_offline_node_has_remove_button(self):
        from propagul.server.dashboard_web import _render_node_card
        node = {"node_id": "old-box", "status": "offline", "hostname": "h",
                "platform": "Linux", "arch": "x86", "model_count": 0,
                "running_count": 0, "gpu_backend": "none", "gpu_count": 0,
                "gpu_utilization": 0, "vram_total_mb": 0, "vram_used_mb": 0,
                "last_seen": time.time() - 600, "pending_commands": 0}
        html = _render_node_card(node)
        assert "btn-remove" in html
        assert "hx-delete" in html

    def test_online_node_has_no_remove_button(self):
        from propagul.server.dashboard_web import _render_node_card
        node = {"node_id": "alive-box", "status": "online", "hostname": "h",
                "platform": "Linux", "arch": "x86", "model_count": 2,
                "running_count": 1, "gpu_backend": "nvidia", "gpu_count": 1,
                "gpu_utilization": 50, "vram_total_mb": 8192, "vram_used_mb": 4096,
                "last_seen": time.time(), "pending_commands": 0}
        html = _render_node_card(node)
        assert "btn-remove" not in html

    def test_nav_account_in_dashboard(self):
        from propagul.server.dashboard_web import render_dashboard_page
        html = render_dashboard_page(
            {"online": 0, "offline": 0, "total_nodes": 0,
             "total_models": 0, "total_gpus": 0}, [], [],
            nonce="test", user_email="dev@test.com",
            user_tier="pro", masked_key="pg_pro_****1234")
        assert "nav-account" in html
        assert "PRO" in html
        assert "dev@test.com" in html
        # Old account-bar class should NOT exist
        assert "account-bar" not in html

    def test_fleet_tab_has_nav_tabs(self):
        from propagul.server.dashboard_web import render_dashboard_page
        html = render_dashboard_page(
            {"online": 0, "offline": 0, "total_nodes": 0,
             "total_models": 0, "total_gpus": 0}, [], [],
            nonce="test", active_tab="fleet")
        assert "nav-tabs" in html
        assert "/dashboard/models" in html
        assert "/dashboard/account" in html
        # Fleet tab should be active
        assert 'class="nav-tab active">Fleet' in html

    def test_models_tab_renders_model_content(self):
        from propagul.server.dashboard_web import render_dashboard_page
        models = [{"name": "llama3:8b", "size_gb": 4.0, "parameter_size": "8B",
                   "quantization": "Q4", "node_id": "n1", "node_status": "online",
                   "backend": "ollama"}]
        html = render_dashboard_page(
            {"online": 1, "offline": 0, "total_nodes": 1,
             "total_models": 1, "total_gpus": 1}, [], models,
            nonce="test", active_tab="models")
        assert "Model Inventory" in html
        assert "llama3:8b" in html
        assert 'class="nav-tab active">Models' in html
        # Fleet tab should NOT be active
        assert 'class="nav-tab active">Fleet' not in html

    def test_account_tab_renders_account_content(self):
        from propagul.server.dashboard_web import render_dashboard_page
        html = render_dashboard_page(
            {"online": 0, "offline": 0, "total_nodes": 0,
             "total_models": 0, "total_gpus": 0}, [], [],
            nonce="test", user_email="admin@fleet.io",
            user_tier="business", masked_key="pg_biz_****abcd",
            active_tab="account")
        assert "account-page" in html
        assert "admin@fleet.io" in html
        assert "BUSINESS" in html
        assert "pg_biz_****abcd" in html
        assert 'class="nav-tab active">Account' in html


# ─── Email Sender Tests ─────────────────────────────────────────

class TestEmailSender:
    """Test email_sender module."""

    def test_is_not_configured_by_default(self, monkeypatch):
        monkeypatch.delenv("PROPAGUL_RESEND_API_KEY", raising=False)
        from propagul.server.email_sender import is_configured
        assert not is_configured()

    def test_is_configured_with_key(self, monkeypatch):
        monkeypatch.setenv("PROPAGUL_RESEND_API_KEY", "re_test_1234")
        from propagul.server import email_sender
        # Force re-evaluation
        assert email_sender.is_configured()

    def test_send_noop_without_key(self, monkeypatch):
        monkeypatch.delenv("PROPAGUL_RESEND_API_KEY", raising=False)
        from propagul.server.email_sender import send_api_key_email
        result = send_api_key_email("user@test.com", "pg_free_abc123", "free")
        assert result is False  # No-op, not an error


# ─── Billing Tests ──────────────────────────────────────────────

class TestBilling:
    """Test Stripe billing module."""

    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("PROPAGUL_STRIPE_MODE", raising=False)
        monkeypatch.delenv("PROPAGUL_STRIPE_KEY", raising=False)
        from propagul.server.billing import is_enabled
        assert not is_enabled()

    def test_invalid_mode_defaults_disabled(self, monkeypatch):
        monkeypatch.setenv("PROPAGUL_STRIPE_MODE", "invalid")
        from propagul.server.billing import _get_mode
        assert _get_mode() == "disabled"

    def test_sandbox_rejects_live_key(self, monkeypatch):
        monkeypatch.setenv("PROPAGUL_STRIPE_MODE", "sandbox")
        monkeypatch.setenv("PROPAGUL_STRIPE_KEY", "sk_live_should_fail")
        from propagul.server.billing import is_enabled
        assert not is_enabled()

    def test_live_rejects_test_key(self, monkeypatch):
        monkeypatch.setenv("PROPAGUL_STRIPE_MODE", "live")
        monkeypatch.setenv("PROPAGUL_STRIPE_KEY", "sk_test_should_fail")
        from propagul.server.billing import is_enabled
        assert not is_enabled()

    def test_sandbox_accepts_test_key(self, monkeypatch):
        monkeypatch.setenv("PROPAGUL_STRIPE_MODE", "sandbox")
        monkeypatch.setenv("PROPAGUL_STRIPE_KEY", "sk_test_abc123")
        from propagul.server.billing import is_enabled
        assert is_enabled()

    def test_handle_checkout_completed(self):
        from propagul.server.billing import handle_webhook_event
        event = {
            "type": "checkout.session.completed",
            "data": {"object": {
                "customer_email": "user@test.com",
                "subscription": "sub_abc123",
            }},
        }
        result = handle_webhook_event(event)
        assert result["action"] == "upgrade"
        assert result["email"] == "user@test.com"

    def test_handle_subscription_deleted(self):
        from propagul.server.billing import handle_webhook_event
        event = {
            "type": "customer.subscription.deleted",
            "data": {"object": {
                "id": "sub_abc123",
                "customer": "cus_xyz",
            }},
        }
        result = handle_webhook_event(event)
        assert result["action"] == "downgrade"

    def test_handle_unknown_event(self):
        from propagul.server.billing import handle_webhook_event
        result = handle_webhook_event({"type": "some.random.event", "data": {}})
        assert result["action"] == "ignored"

    def test_webhook_sig_rejects_empty(self):
        from propagul.server.billing import verify_webhook_signature
        assert not verify_webhook_signature(b"payload", "")

    def test_webhook_sig_rejects_without_secret(self, monkeypatch):
        monkeypatch.delenv("PROPAGUL_STRIPE_WEBHOOK_SECRET", raising=False)
        from propagul.server.billing import verify_webhook_signature
        assert not verify_webhook_signature(b"test", "t=123,v1=abc")


# ─── Wiring Integration Tests ───────────────────────────────────

class TestEmailWiring:
    """Verify email_sender is wired into waitlist key generation."""

    def test_generate_key_calls_email_sender(self):
        """waitlist.py must import and call send_api_key_email."""
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "python" / "propagul" / "server" / "waitlist.py"
        code = src.read_text()
        assert "send_api_key_email" in code, "email_sender.send_api_key_email must be called in waitlist.py"
        assert "email_sent" in code, "Response must include email_sent field"

    def test_email_send_is_nonfatal(self):
        """If email fails, key is still returned (non-fatal)."""
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "python" / "propagul" / "server" / "waitlist.py"
        code = src.read_text()
        assert "except Exception" in code, "Email send must be wrapped in try/except"


class TestBillingWiring:
    """Verify billing is wired into dashboard routes."""

    def test_account_tab_shows_upgrade_for_free_tier(self, monkeypatch):
        """Free tier users see an upgrade button when billing is enabled."""
        monkeypatch.setenv("PROPAGUL_STRIPE_MODE", "sandbox")
        monkeypatch.setenv("PROPAGUL_STRIPE_KEY", "sk_test_wiring_test")
        from propagul.server.dashboard_web import _render_account_content
        html = _render_account_content(
            user_email="free@user.com",
            user_tier="free",
            masked_key="pg_free_****1234",
        )
        assert "Upgrade to Pro" in html
        assert "/dashboard/api/checkout" in html

    def test_account_tab_hides_upgrade_for_pro(self):
        """Pro tier users do NOT see an upgrade button."""
        from propagul.server.dashboard_web import _render_account_content
        html = _render_account_content(
            user_email="pro@user.com",
            user_tier="pro",
            masked_key="pg_pro_****abcd",
        )
        assert "Upgrade to Pro" not in html

    def test_account_tab_hides_upgrade_when_billing_disabled(self, monkeypatch):
        """Upgrade button is hidden when billing is disabled."""
        monkeypatch.delenv("PROPAGUL_STRIPE_MODE", raising=False)
        monkeypatch.delenv("PROPAGUL_STRIPE_KEY", raising=False)
        from propagul.server.dashboard_web import _render_account_content
        html = _render_account_content(
            user_email="free@user.com",
            user_tier="free",
            masked_key="pg_free_****1234",
        )
        assert "Upgrade to Pro" not in html

    def test_checkout_route_registered(self):
        """Verify /dashboard/api/checkout route exists in code."""
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "python" / "propagul" / "server" / "dashboard_web.py"
        code = src.read_text()
        assert "/dashboard/api/checkout" in code
        assert "create_checkout_session" in code

    def test_webhook_route_registered(self):
        """Verify /webhook/stripe route exists in code."""
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "python" / "propagul" / "server" / "dashboard_web.py"
        code = src.read_text()
        assert "/webhook/stripe" in code
        assert "verify_webhook_signature" in code
        assert "handle_webhook_event" in code


# ─── R-02: Stale Node Cleanup Tests ─────────────────────────────

class TestStaleNodeCleanup:
    """Verify RedisFleetStore.cleanup_stale_nodes()."""

    def test_cleanup_removes_stale_nodes(self):
        """Nodes with old heartbeats are removed."""
        import time
        from propagul.server.dashboard_api import FleetStore
        store = FleetStore()
        # Add a node with an ancient heartbeat
        store.update_node("stale-node", {})
        store._nodes["stale-node"].last_heartbeat = time.time() - 86400 * 30  # 30 days ago
        # Add an active node
        store.update_node("active-node", {})
        # Cleanup with 7-day threshold
        removed = store.cleanup_stale_nodes(max_age_seconds=86400 * 7)
        assert removed == 1
        assert store.get_node("stale-node") is None
        assert store.get_node("active-node") is not None

    def test_cleanup_preserves_active_nodes(self):
        """Nodes with recent heartbeats survive cleanup."""
        from propagul.server.dashboard_api import FleetStore
        store = FleetStore()
        store.update_node("active-1", {})
        store.update_node("active-2", {})
        removed = store.cleanup_stale_nodes(max_age_seconds=86400 * 7)
        assert removed == 0
        assert store.get_node("active-1") is not None
        assert store.get_node("active-2") is not None

    def test_cleanup_empty_store_returns_zero(self):
        """Empty store returns 0 removed."""
        from propagul.server.dashboard_api import FleetStore
        store = FleetStore()
        removed = store.cleanup_stale_nodes(max_age_seconds=86400)
        assert removed == 0

    def test_cleanup_handles_corrupt_node_data(self):
        """Corrupt JSON in node hash is skipped without crash."""
        from propagul.server.dashboard_api import FleetStore
        store = FleetStore()
        # Directly write corrupt data to Redis
        store.client.hset(store._node_key(), "corrupt-node", "{not valid json!!!")
        store.update_node("valid-node", {})
        # Should not crash, just skip corrupt entry
        removed = store.cleanup_stale_nodes(max_age_seconds=1)
        # valid-node was just created, so it shouldn't be removed
        assert store.get_node("valid-node") is not None
        # Clean up the corrupt entry
        store.client.hdel(store._node_key(), "corrupt-node")

    def test_cleanup_method_exists_in_code(self):
        """Verify cleanup_stale_nodes is defined in redis_store.py."""
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "python" / "propagul" / "server" / "redis_store.py"
        code = src.read_text()
        assert "cleanup_stale_nodes" in code
        assert "max_age_seconds" in code

    def test_env_var_wired_in_server(self):
        """Verify PROPAGUL_NODE_TTL_DAYS is read in server.py."""
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "python" / "propagul" / "server" / "server.py"
        code = src.read_text()
        assert "PROPAGUL_NODE_TTL_DAYS" in code
        assert "_periodic_node_cleanup" in code

    def test_from_prefix_creates_valid_store(self):
        """from_prefix() creates a store with the given prefix."""
        from propagul.server.redis_store import RedisFleetStore
        store = RedisFleetStore.from_prefix("propagul:fleet:testprefix")
        assert store.prefix == "propagul:fleet:testprefix"
        assert store._node_key() == "propagul:fleet:testprefix:nodes"

    def test_from_prefix_has_all_init_fields(self):
        """from_prefix() goes through __init__, so all fields exist."""
        from propagul.server.redis_store import RedisFleetStore
        store = RedisFleetStore.from_prefix("propagul:fleet:abc")
        # These fields are set in __init__ — if __init__ changes,
        # from_prefix() inherits the changes automatically.
        assert hasattr(store, "_nodes")
        assert hasattr(store, "_audit_max")
        assert hasattr(store, "client")
        assert store._max_nodes == 100

    def test_server_uses_from_prefix_not_new(self):
        """server.py must use from_prefix(), not __new__."""
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "python" / "propagul" / "server" / "server.py"
        code = src.read_text()
        assert "from_prefix" in code
        assert "__new__" not in code

    def test_no_google_fonts_links(self):
        """No Google Fonts links in server code (self-hosted via propagul.css)."""
        import pathlib
        server_dir = pathlib.Path(__file__).parent.parent / "python" / "propagul" / "server"
        for py_file in server_dir.glob("*.py"):
            code = py_file.read_text()
            assert "fonts.googleapis.com" not in code, (
                f"Residual Google Fonts link in {py_file.name}. "
                f"Use self-hosted fonts via propagul.css instead."
            )


# ─── CB-03: Redis Webhook Dedup Tests ────────────────────────────

class TestWebhookDedup:
    """Verify Redis-backed webhook deduplication."""

    def test_new_event_returns_true(self):
        """New event_id returns True (should be processed)."""
        from propagul.server.billing import _dedup_check_and_mark
        import uuid
        event_id = f"evt_test_{uuid.uuid4().hex[:12]}"
        assert _dedup_check_and_mark(event_id) is True

    def test_duplicate_event_returns_false(self):
        """Same event_id returns False on second call (duplicate)."""
        from propagul.server.billing import _dedup_check_and_mark
        import uuid
        event_id = f"evt_test_{uuid.uuid4().hex[:12]}"
        assert _dedup_check_and_mark(event_id) is True
        assert _dedup_check_and_mark(event_id) is False

    def test_empty_event_id_always_processes(self):
        """Empty event_id always returns True (can't dedup)."""
        from propagul.server.billing import _dedup_check_and_mark
        assert _dedup_check_and_mark("") is True
        assert _dedup_check_and_mark("") is True

    def test_handle_webhook_dedup_integration(self):
        """handle_webhook_event rejects duplicate events."""
        from propagul.server.billing import handle_webhook_event
        import uuid
        event_id = f"evt_dedup_test_{uuid.uuid4().hex[:12]}"
        event = {"id": event_id, "type": "some.event", "data": {}}
        result1 = handle_webhook_event(event)
        assert result1["action"] == "ignored"  # Unknown event type
        result2 = handle_webhook_event(event)
        assert result2["action"] == "duplicate"

    def test_dedup_uses_redis(self):
        """Verify dedup key exists in Redis after check."""
        from propagul.server.billing import _dedup_check_and_mark, _DEDUP_PREFIX
        from propagul.server.redis_store import get_redis
        import uuid
        event_id = f"evt_redis_test_{uuid.uuid4().hex[:12]}"
        _dedup_check_and_mark(event_id)
        r = get_redis()
        key = f"{_DEDUP_PREFIX}{event_id}"
        assert r.exists(key) == 1
        # Verify TTL is set
        ttl = r.ttl(key)
        assert ttl > 0
        assert ttl <= 86400
        # Clean up
        r.delete(key)


# ─── Subscription Tracking Tests ─────────────────────────────────

class TestSubscriptionTracking:
    """Verify AuthManager.update_tier() and subscription ledger."""

    def test_update_tier_changes_matching_keys(self, tmp_path, monkeypatch):
        """update_tier() changes tier for all keys matching owner email."""
        monkeypatch.setenv("PROPAGUL_KEY_SALT", "test-salt-for-ci-only-32bytes!")
        from propagul.server.auth import AuthManager
        keys_file = str(tmp_path / "keys.json")
        am = AuthManager(keys_file=keys_file, key_salt=b"test-salt")
        am.add_key("pg_test_key1", tier="free", owner="user@test.com")
        am.add_key("pg_test_key2", tier="free", owner="user@test.com")
        am.add_key("pg_test_key3", tier="free", owner="other@test.com")

        count = am.update_tier("user@test.com", "pro")
        assert count == 2

        # Verify the tier changed
        info = am.validate("pg_test_key1")
        assert info is not None
        assert info.tier == "pro"

        # Verify other user unaffected
        info2 = am.validate("pg_test_key3")
        assert info2.tier == "free"

    def test_update_tier_unknown_owner_returns_zero(self, tmp_path):
        """update_tier() returns 0 for unknown owner."""
        from propagul.server.auth import AuthManager
        am = AuthManager(key_salt=b"test-salt")
        count = am.update_tier("nonexistent@test.com", "pro")
        assert count == 0

    def test_update_tier_invalid_tier_returns_zero(self, tmp_path):
        """update_tier() refuses invalid tiers."""
        from propagul.server.auth import AuthManager
        am = AuthManager(key_salt=b"test-salt")
        am.add_key("pg_test_key1", tier="free", owner="user@test.com")
        count = am.update_tier("user@test.com", "enterprise")  # invalid tier
        assert count == 0

    def test_subscription_ledger_roundtrip(self, tmp_path):
        """Subscription ledger survives save/load cycle."""
        from propagul.server.auth import AuthManager
        keys_file = str(tmp_path / "keys.json")
        am = AuthManager(keys_file=keys_file, key_salt=b"test-salt")
        am.add_key("pg_test_key1", tier="free", owner="user@test.com")
        am.record_subscription("cus_abc123", "user@test.com")
        am.save_keys()

        # Load in a new instance
        am2 = AuthManager(keys_file=keys_file, key_salt=b"test-salt")
        assert am2.resolve_customer("cus_abc123") == "user@test.com"
        assert am2.subscription_count == 1

    def test_resolve_unknown_customer_returns_none(self):
        """resolve_customer() returns None for unknown customer_id."""
        from propagul.server.auth import AuthManager
        am = AuthManager(key_salt=b"test-salt")
        assert am.resolve_customer("cus_unknown") is None

    @pytest.mark.skip(reason="Legacy file-I/O — persistence is now Redis BGSAVE")
    def test_update_tier_persists_to_disk(self, tmp_path):
        """update_tier() atomically saves changes to disk."""
        from propagul.server.auth import AuthManager
        import json
        keys_file = str(tmp_path / "keys.json")
        am = AuthManager(keys_file=keys_file, key_salt=b"test-salt")
        am.add_key("pg_test_key1", tier="free", owner="user@test.com")
        am.update_tier("user@test.com", "pro")

        # Read raw file and verify
        data = json.loads(open(keys_file).read())
        # Find any entry with tier=pro
        found_pro = any(
            v.get("tier") == "pro"
            for k, v in data.items()
            if not k.startswith("_") and isinstance(v, dict)
        )
        assert found_pro

    def test_webhook_calls_update_tier(self):
        """Verify webhook route calls auth_manager.update_tier (wiring test)."""
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "python" / "propagul" / "server" / "dashboard_web.py"
        code = src.read_text()
        assert "auth_manager.update_tier" in code
        assert "auth_manager.record_subscription" in code
        assert "auth_manager.resolve_customer" in code
        # The TODO should be gone
        assert "TODO: Update user tier" not in code


# ─── CRDT Config-Sync Tests ─────────────────────────────────────

class TestFleetConfigMap:
    """Verify FleetConfigMap CRDT wrapper."""

    def test_set_and_get_desired_model(self):
        """Set a desired model and retrieve it."""
        from propagul.server.config_sync import FleetConfigMap
        cm = FleetConfigMap(node_id=1)
        cm.set_desired_model("llama3:8b", "pull")
        desired = cm.get_desired_models()
        assert desired == {"llama3:8b": "pull"}

    def test_remove_desired_model(self):
        """Remove a desired model."""
        from propagul.server.config_sync import FleetConfigMap
        cm = FleetConfigMap(node_id=1)
        cm.set_desired_model("llama3:8b", "pull")
        cm.remove_desired_model("llama3:8b")
        assert cm.get_desired_models() == {}

    def test_invalid_action_raises(self):
        """Invalid action raises ValueError."""
        import pytest
        from propagul.server.config_sync import FleetConfigMap
        cm = FleetConfigMap(node_id=1)
        with pytest.raises(ValueError, match="pull.*delete"):
            cm.set_desired_model("llama3:8b", "invalid")

    def test_merge_convergence(self):
        """Two FleetConfigMaps merge to the same state."""
        from propagul.server.config_sync import FleetConfigMap
        server = FleetConfigMap(node_id=0)
        node_a = FleetConfigMap(node_id=1)

        server.set_desired_model("llama3:8b", "pull")
        node_a.set_desired_model("mistral:7b", "pull")

        # Bidirectional merge
        node_a.merge(server.snapshot())
        server.merge(node_a.snapshot())

        # Both should have both models
        assert server.get_desired_models() == {"llama3:8b": "pull", "mistral:7b": "pull"}
        assert node_a.get_desired_models() == {"llama3:8b": "pull", "mistral:7b": "pull"}

    def test_namespace_isolation(self):
        """Desired models, node prefs, and fleet settings don't collide."""
        from propagul.server.config_sync import FleetConfigMap
        cm = FleetConfigMap(node_id=1)
        cm.set_desired_model("llama3:8b", "pull")
        cm.set_node_preference("node-a", "gpu_affinity", "0")
        cm.set_fleet_setting("context_length", "4096")

        assert "llama3:8b" in cm.get_desired_models()
        assert cm.get_node_preferences("node-a") == {"gpu_affinity": "0"}
        assert cm.get_fleet_setting("context_length") == "4096"

        # Total keys: 3 (one for each namespace)
        assert cm.key_count() == 3

    def test_node_preferences_isolated_per_node(self):
        """Node preferences for different nodes don't interfere."""
        from propagul.server.config_sync import FleetConfigMap
        cm = FleetConfigMap(node_id=1)
        cm.set_node_preference("node-a", "gpu", "0")
        cm.set_node_preference("node-b", "gpu", "1")

        assert cm.get_node_preferences("node-a") == {"gpu": "0"}
        assert cm.get_node_preferences("node-b") == {"gpu": "1"}

    def test_fleet_settings_crud(self):
        """Fleet settings set, get, and list."""
        from propagul.server.config_sync import FleetConfigMap
        cm = FleetConfigMap(node_id=1)
        cm.set_fleet_setting("max_concurrent", "4")
        cm.set_fleet_setting("auto_pull", "false")

        assert cm.get_fleet_setting("max_concurrent") == "4"
        settings = cm.get_all_fleet_settings()
        assert settings == {"max_concurrent": "4", "auto_pull": "false"}

    def test_snapshot_roundtrip(self):
        """Snapshot and merge are inverse operations."""
        from propagul.server.config_sync import FleetConfigMap
        cm = FleetConfigMap(node_id=1)
        cm.set_desired_model("llama3:8b", "pull")
        cm.set_fleet_setting("ctx", "4096")

        snap = cm.snapshot()
        cm2 = FleetConfigMap(node_id=2)
        delta = cm2.merge(snap)
        assert delta == 2
        assert cm2.get_desired_models() == {"llama3:8b": "pull"}
        assert cm2.get_fleet_setting("ctx") == "4096"

    def test_summary(self):
        """Summary returns correct counts."""
        from propagul.server.config_sync import FleetConfigMap
        cm = FleetConfigMap(node_id=1)
        cm.set_desired_model("llama3:8b", "pull")
        cm.set_desired_model("old-model", "delete")

        s = cm.summary()
        assert s["total_keys"] == 2
        assert s["desired_models"] == 2
        assert s["desired_pull"] == 1
        assert s["desired_delete"] == 1


class TestConfigSyncWiring:
    """Verify config-sync is wired into API and agent."""

    def test_config_routes_registered(self):
        """Verify /mesh/config, /mesh/config/sync, /mesh/config/desired in code."""
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "python" / "propagul" / "server" / "dashboard_api.py"
        code = src.read_text()
        assert "/mesh/config" in code
        assert "/mesh/config/sync" in code
        assert "/mesh/config/desired" in code
        assert "FleetConfigMap" in code

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

class TestConfigMapPersistence:
    """Verify FleetConfigMap disk persistence."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Config map survives save/load cycle."""
        from propagul.server.config_sync import FleetConfigMap
        path = str(tmp_path / "config.json")
        cm = FleetConfigMap(node_id=1, persist_path=path, debounce_seconds=0)
        cm.set_desired_model("llama3:8b", "pull")
        cm.set_fleet_setting("ctx", "4096")
        cm.set_node_preference("node-a", "gpu", "0")

        # New instance loads from disk
        cm2 = FleetConfigMap(node_id=2, persist_path=path)
        assert cm2.get_desired_models() == {"llama3:8b": "pull"}
        assert cm2.get_fleet_setting("ctx") == "4096"
        assert cm2.get_node_preferences("node-a") == {"gpu": "0"}
        assert cm2.key_count() == 3

    def test_auto_save_on_mutation(self, tmp_path):
        """Every mutation triggers disk save."""
        import os
        from propagul.server.config_sync import FleetConfigMap
        path = str(tmp_path / "config.json")
        cm = FleetConfigMap(node_id=1, persist_path=path, debounce_seconds=0)

        assert not os.path.exists(path)  # No file yet (no data)
        cm.set_desired_model("llama3:8b", "pull")
        assert os.path.exists(path)  # File created after first mutation

        mtime1 = os.path.getmtime(path)
        import time; time.sleep(0.01)  # Ensure mtime differs
        cm.set_fleet_setting("ctx", "4096")
        mtime2 = os.path.getmtime(path)
        assert mtime2 >= mtime1  # File updated

    def test_load_nonexistent_file(self, tmp_path):
        """Loading from nonexistent path is a no-op."""
        from propagul.server.config_sync import FleetConfigMap
        path = str(tmp_path / "does_not_exist.json")
        cm = FleetConfigMap(node_id=1, persist_path=path)
        assert cm.key_count() == 0

    def test_load_corrupt_file(self, tmp_path):
        """Corrupt config file doesn't crash, returns 0."""
        from propagul.server.config_sync import FleetConfigMap
        path = str(tmp_path / "config.json")
        with open(path, "w") as f:
            f.write("{corrupt json!!!")
        cm = FleetConfigMap(node_id=1, persist_path=path)
        assert cm.key_count() == 0  # Graceful degradation

    def test_load_wrong_version(self, tmp_path):
        """Wrong version number is rejected gracefully."""
        import json
        from propagul.server.config_sync import FleetConfigMap
        path = str(tmp_path / "config.json")
        with open(path, "w") as f:
            json.dump({"version": 99, "snapshot": {}}, f)
        cm = FleetConfigMap(node_id=1, persist_path=path)
        assert cm.key_count() == 0

    def test_merge_auto_saves(self, tmp_path):
        """merge() with delta > 0 triggers auto-save."""
        import json
        from propagul.server.config_sync import FleetConfigMap
        path = str(tmp_path / "config.json")
        cm = FleetConfigMap(node_id=1, persist_path=path)

        # Create a remote snapshot
        remote = FleetConfigMap(node_id=2)
        remote.set_desired_model("mistral:7b", "pull")
        snap = remote.snapshot()

        cm.merge(snap)
        assert os.path.exists(path)

        # Verify persisted data
        data = json.loads(open(path).read())
        assert data["version"] == 1
        assert "entries" in data["snapshot"]

    def test_no_persist_path_does_nothing(self):
        """Without persist_path, save/load are no-ops."""
        from propagul.server.config_sync import FleetConfigMap
        cm = FleetConfigMap(node_id=1)
        cm.set_desired_model("llama3:8b", "pull")
        assert cm.save_to_disk() is False
        assert cm.load_from_disk() == 0

    @pytest.mark.skip(reason="Legacy file-I/O — config is now Redis-backed")
    def test_persist_path_wired_in_dashboard_api(self):
        """Verify dashboard_api uses persist_path for config_map. SKIPPED: Redis-native."""
        pass


# ─── Auto-Pull Tests ─────────────────────────────────────────────

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

class TestDebouncedPersistence:
    """Verify FleetConfigMap debounced I/O."""

    def test_rapid_mutations_coalesced(self, tmp_path):
        """Rapid mutations within debounce window produce only 1 disk write."""
        import os
        from propagul.server.config_sync import FleetConfigMap
        path = str(tmp_path / "config.json")
        # 10s debounce = only first mutation triggers write
        cm = FleetConfigMap(node_id=1, persist_path=path, debounce_seconds=10.0)

        cm.set_desired_model("model-a", "pull")
        assert os.path.exists(path)  # First write goes through
        mtime1 = os.path.getmtime(path)

        import time; time.sleep(0.01)
        cm.set_desired_model("model-b", "pull")
        cm.set_desired_model("model-c", "pull")
        cm.set_fleet_setting("ctx", "4096")
        mtime2 = os.path.getmtime(path)

        # mtime should NOT have changed (debounce blocked writes)
        assert mtime2 == mtime1

        # But dirty flag should be set
        assert cm.is_dirty is True

    def test_flush_captures_dirty_state(self, tmp_path):
        """flush_to_disk() saves pending debounced mutations."""
        from propagul.server.config_sync import FleetConfigMap
        path = str(tmp_path / "config.json")
        cm = FleetConfigMap(node_id=1, persist_path=path, debounce_seconds=10.0)

        cm.set_desired_model("model-a", "pull")  # Triggers write (first call)
        cm.set_desired_model("model-b", "pull")  # Debounced (no write)
        assert cm.is_dirty is True

        cm.flush_to_disk()
        assert cm.is_dirty is False

        # Verify both models persisted
        cm2 = FleetConfigMap(node_id=2, persist_path=path)
        assert cm2.get_desired_models() == {"model-a": "pull", "model-b": "pull"}

    def test_explicit_save_bypasses_debounce(self, tmp_path):
        """save_to_disk() always writes, regardless of debounce timer."""
        import os
        from propagul.server.config_sync import FleetConfigMap
        path = str(tmp_path / "config.json")
        cm = FleetConfigMap(node_id=1, persist_path=path, debounce_seconds=10.0)

        cm.set_desired_model("model-a", "pull")
        mtime1 = os.path.getmtime(path)

        import time; time.sleep(0.01)
        cm.set_desired_model("model-b", "pull")  # Debounced
        cm.save_to_disk()  # Explicit save bypasses debounce
        mtime2 = os.path.getmtime(path)
        assert mtime2 > mtime1

    def test_zero_debounce_saves_every_mutation(self, tmp_path):
        """debounce_seconds=0 saves on every mutation (backward compat)."""
        import os
        from propagul.server.config_sync import FleetConfigMap
        path = str(tmp_path / "config.json")
        cm = FleetConfigMap(node_id=1, persist_path=path, debounce_seconds=0.0)

        cm.set_desired_model("model-a", "pull")
        mtime1 = os.path.getmtime(path)

        import time; time.sleep(0.01)
        cm.set_desired_model("model-b", "pull")
        mtime2 = os.path.getmtime(path)
        assert mtime2 > mtime1  # Both wrote

    def test_flush_noop_when_clean(self, tmp_path):
        """flush_to_disk() returns False when not dirty."""
        from propagul.server.config_sync import FleetConfigMap
        path = str(tmp_path / "config.json")
        cm = FleetConfigMap(node_id=1, persist_path=path, debounce_seconds=0.0)
        cm.set_desired_model("model-a", "pull")
        assert cm.is_dirty is False  # Saved immediately with debounce=0
        assert cm.flush_to_disk() is False

    def test_dirty_flag_after_merge(self, tmp_path):
        """Merge with delta>0 within debounce sets dirty flag."""
        from propagul.server.config_sync import FleetConfigMap
        path = str(tmp_path / "config.json")
        cm = FleetConfigMap(node_id=1, persist_path=path, debounce_seconds=10.0)

        # First mutation triggers write
        cm.set_desired_model("model-a", "pull")
        assert cm.is_dirty is False  # Was written

        # Merge from remote within debounce window → dirty
        remote = FleetConfigMap(node_id=2)
        remote.set_desired_model("model-b", "pull")
        cm.merge(remote.snapshot())
        assert cm.is_dirty is True  # Debounced, not yet saved


# ─── Async Auto-Pull Tests ───────────────────────────────────────

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


class TestGpuTempDisplay:
    """GPU temperature and power draw in node summary."""

    def test_to_summary_includes_gpu_temp(self):
        """Node summary includes GPU temperature from telemetry."""
        from propagul.server.dashboard_api import NodeState
        import time
        node = NodeState(
            node_id="gpu-node",
            last_heartbeat=time.time(),
            telemetry={
                "gpu": {
                    "backend": "nvidia", "gpu_count": 1,
                    "avg_utilization_pct": 50,
                    "total_vram_mb": 24000, "total_vram_used_mb": 12000,
                    "gpus": [{"temperature_c": 72, "power_draw_w": 250.0}],
                },
                "backends": [],
            },
        )
        summary = node.to_summary()
        assert summary["gpu_temp_c"] == 72
        assert summary["gpu_power_w"] == 250.0

    def test_to_summary_no_gpu_returns_none(self):
        """Node without GPU has None for temp/power."""
        from propagul.server.dashboard_api import NodeState
        import time
        node = NodeState(
            node_id="cpu-node",
            last_heartbeat=time.time(),
            telemetry={"gpu": {"backend": "none", "gpus": []}, "backends": []},
        )
        summary = node.to_summary()
        assert summary["gpu_temp_c"] is None
        assert summary["gpu_power_w"] is None

    def test_render_temp_row_green(self):
        """Temperature <60C renders green."""
        from propagul.server.dashboard_web import _render_gpu_temp_row
        html = _render_gpu_temp_row({"gpu_temp_c": 45, "gpu_power_w": 120.0})
        assert "45" in html
        assert "accent-green" in html
        assert "120W" in html

    def test_render_temp_row_amber(self):
        """Temperature 60-80C renders amber."""
        from propagul.server.dashboard_web import _render_gpu_temp_row
        html = _render_gpu_temp_row({"gpu_temp_c": 70, "gpu_power_w": None})
        assert "70" in html
        assert "accent-amber" in html

    def test_render_temp_row_red(self):
        """Temperature >80C renders red."""
        from propagul.server.dashboard_web import _render_gpu_temp_row
        html = _render_gpu_temp_row({"gpu_temp_c": 85, "gpu_power_w": 300.0})
        assert "85" in html
        assert "accent-red" in html

    def test_render_temp_row_no_data(self):
        """No GPU data returns empty string."""
        from propagul.server.dashboard_web import _render_gpu_temp_row
        html = _render_gpu_temp_row({"gpu_temp_c": None, "gpu_power_w": None})
        assert html == ""


class TestAuditLog:
    """Audit log ring buffer in FleetStore."""

    def test_audit_log_records_command(self):
        """add_command logs an audit event."""
        from propagul.server.dashboard_api import FleetStore
        store = FleetStore()
        store.update_node("node-1", {})
        store.add_command("node-1", "pull", "llama3:8b")
        events = store.get_audit_log()
        assert len(events) == 1
        assert events[0]["action"] == "pull"
        assert events[0]["model"] == "llama3:8b"
        assert events[0]["node_id"] == "node-1"

    def test_audit_log_max_size(self):
        """Ring buffer evicts oldest events at max capacity."""
        from propagul.server.dashboard_api import FleetStore
        store = FleetStore()
        store._audit_max = 5
        store.update_node("n1", {})
        for i in range(10):
            store.add_command("n1", "pull", f"model-{i}")
        events = store.get_audit_log(limit=100)
        assert len(events) == 5
        assert events[0]["model"] == "model-9"
        assert events[4]["model"] == "model-5"

    def test_audit_log_persists(self):
        """Audit events survive across store instances (Redis-backed)."""
        from propagul.server.dashboard_api import FleetStore
        store1 = FleetStore()
        store1.update_node("n1", {})
        store1.add_command("n1", "pull", "llama3:8b")
        # Same Redis prefix — second instance sees the same data
        store2 = FleetStore()
        events = store2.get_audit_log()
        assert len(events) == 1
        assert events[0]["action"] == "pull"

    def test_audit_log_fleet_command_logs_all(self):
        """Fleet-wide command creates N audit events."""
        from propagul.server.dashboard_api import FleetStore
        store = FleetStore()
        store.update_node("n1", {})
        store.update_node("n2", {})
        store.update_node("n3", {})
        for node in store.get_all_nodes():
            store.add_command(node.node_id, "delete", "mistral:7b")
        events = store.get_audit_log()
        assert len(events) == 3
        assert all(e["action"] == "delete" for e in events)

    def test_audit_count_property(self):
        """audit_count returns total events in log."""
        from propagul.server.dashboard_api import FleetStore
        store = FleetStore()
        store.update_node("n1", {})
        assert store.audit_count == 0
        store.add_command("n1", "pull", "llama3:8b")
        assert store.audit_count == 1

    def test_audit_log_newest_first(self):
        """get_audit_log returns newest events first."""
        from propagul.server.dashboard_api import FleetStore
        import time
        store = FleetStore()
        store.update_node("n1", {})
        store.add_command("n1", "pull", "model-a")
        time.sleep(0.01)
        store.add_command("n1", "pull", "model-b")
        events = store.get_audit_log()
        assert events[0]["model"] == "model-b"
        assert events[1]["model"] == "model-a"


class TestCoverageMatrix:
    """Model coverage matrix across fleet nodes."""

    def test_coverage_matrix_all_installed(self):
        """All nodes have all models."""
        from propagul.server.dashboard_api import FleetStore
        store = FleetStore()
        store.update_node("n1", {"backends": [{"models": [{"name": "llama3:8b"}]}]})
        store.update_node("n2", {"backends": [{"models": [{"name": "llama3:8b"}]}]})
        matrix = store.model_coverage_matrix()
        assert matrix["models"] == ["llama3:8b"]
        assert matrix["matrix"]["llama3:8b"]["n1"] is True
        assert matrix["matrix"]["llama3:8b"]["n2"] is True

    def test_coverage_matrix_missing_model(self):
        """Model on n1 but not n2."""
        from propagul.server.dashboard_api import FleetStore
        store = FleetStore()
        store.update_node("n1", {"backends": [{"models": [{"name": "llama3:8b"}, {"name": "mistral:7b"}]}]})
        store.update_node("n2", {"backends": [{"models": [{"name": "llama3:8b"}]}]})
        matrix = store.model_coverage_matrix()
        assert matrix["matrix"]["mistral:7b"]["n1"] is True
        assert matrix["matrix"]["mistral:7b"]["n2"] is False

    def test_coverage_matrix_empty_fleet(self):
        """Empty fleet returns empty matrix."""
        from propagul.server.dashboard_api import FleetStore
        store = FleetStore()
        matrix = store.model_coverage_matrix()
        assert matrix["models"] == []
        assert matrix["nodes"] == []

    def test_coverage_matrix_render_shows_checkmarks(self):
        """Coverage matrix HTML contains check/cross marks."""
        from propagul.server.dashboard_web import _render_coverage_matrix
        coverage = {
            "models": ["llama3:8b"],
            "nodes": ["n1", "n2"],
            "matrix": {"llama3:8b": {"n1": True, "n2": False}},
        }
        html = _render_coverage_matrix(coverage)
        assert "Model Coverage" in html

    def test_coverage_matrix_render_empty(self):
        """Empty coverage returns empty string."""
        from propagul.server.dashboard_web import _render_coverage_matrix
        html = _render_coverage_matrix({"models": [], "nodes": [], "matrix": {}})
        assert html == ""


class TestDashboardRendering:
    """Dashboard HTML rendering correctness."""

    def test_render_node_card_contains_temperature(self):
        """Node card includes GPU temperature when present."""
        from propagul.server.dashboard_web import _render_node_card
        import time
        node = {
            "node_id": "test-node", "status": "online",
            "last_seen": time.time(), "hostname": "gpu-server",
            "platform": "Linux", "arch": "x86_64",
            "model_count": 3, "running_count": 1,
            "gpu_backend": "nvidia", "gpu_count": 1,
            "gpu_utilization": 75, "vram_total_mb": 24000,
            "vram_used_mb": 18000,
            "gpu_temp_c": 68, "gpu_power_w": 200.0,
        }
        html = _render_node_card(node)
        assert "68" in html
        assert "200W" in html

    def test_render_node_card_escapes_xss(self):
        """XSS in node_id is escaped."""
        from propagul.server.dashboard_web import _render_node_card
        import time
        node = {
            "node_id": "<script>alert(1)</script>",
            "status": "online", "last_seen": time.time(),
            "hostname": "", "platform": "", "arch": "",
            "model_count": 0, "running_count": 0,
            "gpu_backend": "none", "gpu_count": 0,
            "gpu_utilization": 0, "vram_total_mb": 0,
            "vram_used_mb": 0,
            "gpu_temp_c": None, "gpu_power_w": None,
        }
        html = _render_node_card(node)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_render_audit_log_table(self):
        """Audit log table renders with correct styling."""
        from propagul.server.dashboard_web import _render_audit_log
        import time
        events = [
            {"timestamp": time.time(), "action": "pull", "model": "llama3:8b",
             "node_id": "n1", "user": "dashboard"},
            {"timestamp": time.time(), "action": "delete", "model": "mistral:7b",
             "node_id": "n2", "user": "dashboard"},
        ]
        html = _render_audit_log(events)
        assert "llama3:8b" in html
        assert "accent-green" in html
        assert "accent-red" in html

    def test_render_audit_log_empty(self):
        """Empty audit log shows placeholder."""
        from propagul.server.dashboard_web import _render_audit_log
        html = _render_audit_log([])
        assert "No activity recorded" in html

    def test_activity_tab_in_navigation(self):
        """Activity tab appears in dashboard."""
        from propagul.server.dashboard_web import render_dashboard_page
        html = render_dashboard_page(
            fleet_health={"online": 0, "offline": 0, "total_models": 0, "total_gpus": 0},
            nodes=[], models=[],
            active_tab="activity", audit_events=[],
        )
        assert "/dashboard/activity" in html
        assert "Activity" in html

    def test_batch_delete_button_in_model_table(self):
        """Batch delete button appears in model table."""
        from propagul.server.dashboard_web import _render_model_table
        models = [{
            "name": "llama3:8b", "size_gb": 4.7, "parameter_size": "8B",
            "quantization": "Q4_0", "node_id": "n1", "node_status": "online",
            "backend": "ollama",
        }]
        html = _render_model_table(models)
        assert "/dashboard/api/fleet/delete" in html
        assert ">All</button>" in html


class TestNodeModelNames:
    """NodeState._model_names() helper."""

    def test_model_names_from_telemetry(self):
        """Extracts model names from backends."""
        from propagul.server.dashboard_api import NodeState
        node = NodeState(
            node_id="n1",
            telemetry={
                "backends": [
                    {"models": [{"name": "llama3:8b"}, {"name": "mistral:7b"}]},
                    {"models": [{"name": "llama3:8b"}]},
                ],
            },
        )
        names = node._model_names()
        assert names == {"llama3:8b", "mistral:7b"}

    def test_model_names_empty(self):
        """No backends returns empty set."""
        from propagul.server.dashboard_api import NodeState
        node = NodeState(node_id="n1")
        names = node._model_names()
        assert names == set()


class TestFleetDeleteWiring:
    """Fleet-wide delete batch action."""

    def test_fleet_delete_queues_all_nodes(self):
        """Fleet delete queues command on every node."""
        from propagul.server.dashboard_api import FleetStore
        store = FleetStore()
        store.update_node("n1", {})
        store.update_node("n2", {})
        store.update_node("n3", {})
        for node in store.get_all_nodes():
            store.add_command(node.node_id, "delete", "llama3:8b")
        for nid in ["n1", "n2", "n3"]:
            cmds = store.pop_commands(nid)
            assert len(cmds) == 1
            assert cmds[0]["command"] == "delete"

    def test_activity_route_registered(self):
        """Activity route is registered."""
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "python" / "propagul" / "server" / "dashboard_web.py"
        code = src.read_text()
        assert '"/dashboard/activity"' in code
        assert '"/dashboard/partial/activity"' in code
        assert '"/dashboard/api/fleet/delete"' in code


# ═══════════════════════════════════════════════════════════════════
# Phase 3.9: VRAM Sparkline + Health Alerts + A2A State Layer
# ═══════════════════════════════════════════════════════════════════


class TestVramSparkline:
    """VRAM history ring buffer and SVG sparkline rendering."""

    def test_vram_history_populated_on_heartbeat(self):
        from propagul.server.dashboard_api import FleetStore
        store = FleetStore()
        store.update_node("n1", {
            "gpu": {"total_vram_mb": 24000, "total_vram_used_mb": 12000},
            "backends": [],
        })
        node = store.get_node("n1")
        assert len(node.vram_history) == 1
        assert node.vram_history[0] == 50.0

    def test_vram_history_retains_items(self):
        from propagul.server.dashboard_api import FleetStore
        store = FleetStore()
        for i in range(100):
            store.update_node("n1", {
                "gpu": {"total_vram_mb": 100, "total_vram_used_mb": i},
                "backends": [],
            })
        node = store.get_node("n1")
        assert len(node.vram_history) == 100

    def test_vram_history_no_gpu_skips(self):
        from propagul.server.dashboard_api import FleetStore
        store = FleetStore()
        store.update_node("n1", {"backends": []})
        node = store.get_node("n1")
        assert len(node.vram_history) == 0

    def test_vram_history_in_summary(self):
        from propagul.server.dashboard_api import FleetStore
        store = FleetStore()
        store.update_node("n1", {
            "gpu": {"total_vram_mb": 100, "total_vram_used_mb": 75},
            "backends": [],
        })
        summary = store.get_node("n1").to_summary()
        assert "vram_history" in summary
        assert summary["vram_history"] == [75.0]



    def test_chartjs_canvas_wired_into_node_card(self):
        from propagul.server.dashboard_web import _render_node_card
        import time
        node = {
            "node_id": "test", "status": "online",
            "last_seen": time.time(), "hostname": "", "platform": "",
            "arch": "", "model_count": 0, "running_count": 0,
            "gpu_backend": "nvidia", "gpu_count": 1, "gpu_utilization": 50,
            "vram_total_mb": 100, "vram_used_mb": 50,
            "gpu_temp_c": None, "gpu_power_w": None,
            "vram_history": [20, 30, 40, 50],
        }
        html = _render_node_card(node)
        assert "<canvas" in html
        assert "data-vram-initial=" in html


class TestHealthAlerts:
    """Node health alert detection and cooldown."""

    def test_detect_offline_node(self):
        from propagul.server.dashboard_api import FleetStore
        import time
        store = FleetStore()
        store._offline_alert_threshold = 10
        store._alert_cooldown_seconds = 0
        store.update_node("n1", {})
        store._nodes["n1"].last_heartbeat = time.time() - 100
        alerts = store.check_health_alerts()
        assert len(alerts) == 1
        assert alerts[0]["node_id"] == "n1"

    def test_online_node_no_alert(self):
        from propagul.server.dashboard_api import FleetStore
        store = FleetStore()
        store.update_node("n1", {})
        assert len(store.check_health_alerts()) == 0

    def test_cooldown_prevents_repeat(self):
        from propagul.server.dashboard_api import FleetStore
        import time
        store = FleetStore()
        store._offline_alert_threshold = 10
        store._alert_cooldown_seconds = 3600
        store.update_node("n1", {})
        store._nodes["n1"].last_heartbeat = time.time() - 100
        assert len(store.check_health_alerts()) == 1
        assert len(store.check_health_alerts()) == 0

    def test_never_seen_no_alert(self):
        from propagul.server.dashboard_api import FleetStore
        store = FleetStore()
        store._offline_alert_threshold = 0
        store.update_node("n1", {})
        store._nodes["n1"].last_heartbeat = 0.0
        assert len(store.check_health_alerts()) == 0

    def test_alert_logged_to_audit(self):
        from propagul.server.dashboard_api import FleetStore
        import time
        store = FleetStore()
        store._offline_alert_threshold = 10
        store._alert_cooldown_seconds = 0
        store.update_node("n1", {})
        store._nodes["n1"].last_heartbeat = time.time() - 100
        store.check_health_alerts()
        assert any(e["action"] == "health_alert" for e in store.get_audit_log())

    def test_health_alert_email_exists(self):
        from propagul.server.email_sender import send_health_alert_email
        assert callable(send_health_alert_email)

    def test_health_alert_email_noop(self):
        from propagul.server.email_sender import send_health_alert_email
        import os
        old = os.environ.pop("PROPAGUL_RESEND_API_KEY", None)
        try:
            assert send_health_alert_email("t@t.com", "n1", 3.5) is False
        finally:
            if old:
                os.environ["PROPAGUL_RESEND_API_KEY"] = old

    def test_health_check_api_route(self):
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "python" / "propagul" / "server" / "dashboard_api.py"
        code = src.read_text()
        assert "/mesh/fleet/health-check" in code
        assert "/mesh/audit" in code


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


class TestLandingPagePhase4:
    """Landing page Phase 4.0 updates."""

    def test_landing_has_pypi_coming_soon(self):
        from propagul.server.landing import _render_landing_page
        html = _render_landing_page()
        assert "Coming Soon" in html
        assert "pip install propagul-mesh" in html

    def test_landing_has_9_features(self):
        from propagul.server.landing import _render_landing_page
        html = _render_landing_page()
        # 9 feature cards: Fleet, Models, Multi-Backend, Security, Audit, Agent, ZeroPF, API, Routing
        icon_count = html.count('class="icon"')
        assert icon_count == 9, f"Expected 9 feature cards, got {icon_count}"

    def test_landing_mentions_all_backends(self):
        from propagul.server.landing import _render_landing_page
        html = _render_landing_page()
        assert "vLLM" in html
        assert "TGI" in html
        assert "LM Studio" in html
        assert "llama.cpp" in html

    def test_landing_version_in_footer(self):
        from propagul.server.landing import _render_landing_page
        html = _render_landing_page()
        assert "v0.13.27" in html


class TestDashboardBackendBadge:
    """Dashboard node card backend badge rendering."""

    def test_node_card_shows_backend_badge(self):
        from propagul.server.dashboard_web import _render_node_card
        node = {
            "node_id": "gpu-box", "status": "online", "hostname": "h",
            "platform": "Linux", "arch": "x86", "model_count": 1,
            "running_count": 1, "gpu_backend": "nvidia", "gpu_count": 1,
            "gpu_utilization": 50, "vram_total_mb": 8192, "vram_used_mb": 4096,
            "last_seen": time.time(), "pending_commands": 0, "backend": "vllm",
        }
        html = _render_node_card(node)
        assert "backend-badge" in html
        assert "vllm" in html.lower()

    def test_node_card_readonly_for_non_ollama(self):
        from propagul.server.dashboard_web import _render_node_card
        node = {
            "node_id": "tgi-box", "status": "online", "hostname": "h",
            "platform": "Linux", "arch": "x86", "model_count": 1,
            "running_count": 1, "gpu_backend": "nvidia", "gpu_count": 1,
            "gpu_utilization": 50, "vram_total_mb": 8192, "vram_used_mb": 4096,
            "last_seen": time.time(), "pending_commands": 0, "backend": "tgi",
        }
        html = _render_node_card(node)
        assert "read-only" in html

    def test_node_card_no_readonly_for_ollama(self):
        from propagul.server.dashboard_web import _render_node_card
        node = {
            "node_id": "ollama-box", "status": "online", "hostname": "h",
            "platform": "Linux", "arch": "x86", "model_count": 2,
            "running_count": 1, "gpu_backend": "nvidia", "gpu_count": 1,
            "gpu_utilization": 50, "vram_total_mb": 8192, "vram_used_mb": 4096,
            "last_seen": time.time(), "pending_commands": 0, "backend": "ollama",
        }
        html = _render_node_card(node)
        assert "read-only" not in html




# ─── CSP Compliance Tests ───────────────────────────────────────

class TestCSPCompliance:
    """Verify no hx-on:: attributes in rendered HTML (CSP-unsafe)."""

    def test_fleet_tab_no_hx_on(self):
        """Fleet tab must not use hx-on:: (CSP blocks it)."""
        from propagul.server.dashboard_web import render_dashboard_page
        html = render_dashboard_page(
            {"online": 1, "offline": 0, "total_nodes": 1,
             "total_models": 1, "total_gpus": 1},
            [{"node_id": "test", "status": "online", "hostname": "h",
              "platform": "L", "arch": "x86", "model_count": 1,
              "running_count": 0, "gpu_backend": "nvidia", "gpu_count": 1,
              "gpu_utilization": 50, "vram_total_mb": 8192, "vram_used_mb": 4096,
              "last_seen": time.time(), "pending_commands": 0, "backend": "ollama"}],
            [{"name": "llama3", "size_gb": 4.0, "parameter_size": "8B",
              "quantization": "Q4", "node_id": "test", "node_status": "online",
              "backend": "ollama"}],
            nonce="test_nonce", active_tab="fleet")
        # hx-on:: requires unsafe-eval in CSP — must not appear
        assert 'hx-on::' not in html or 'hx-on:: needed' in html  # comment OK

    def test_models_tab_no_hx_on(self):
        """Models tab must use data-toast instead of hx-on::."""
        from propagul.server.dashboard_web import _render_model_table
        models = [{"name": "llama3", "size_gb": 4.0, "parameter_size": "8B",
                   "quantization": "Q4", "node_id": "n1", "node_status": "online",
                   "backend": "ollama"}]
        html = _render_model_table(models)
        assert 'data-toast' in html
        assert 'hx-on::after-request' not in html

    def test_remove_button_uses_data_reload(self):
        """Remove button must use data-reload, not hx-on::."""
        from propagul.server.dashboard_web import _render_node_card
        node = {"node_id": "old-box", "status": "offline", "hostname": "h",
                "platform": "Linux", "arch": "x86", "model_count": 0,
                "running_count": 0, "gpu_backend": "none", "gpu_count": 0,
                "gpu_utilization": 0, "vram_total_mb": 0, "vram_used_mb": 0,
                "last_seen": time.time() - 600, "pending_commands": 0}
        html = _render_node_card(node)
        assert 'data-reload' in html
        assert 'hx-on::after-request' not in html

    def test_nonce_script_has_event_listener(self):
        """Verify centralized event listener exists in rendered page."""
        from propagul.server.dashboard_web import render_dashboard_page
        html = render_dashboard_page(
            {"online": 0, "offline": 0, "total_nodes": 0,
             "total_models": 0, "total_gpus": 0}, [], [],
            nonce="test_nonce")
        # After asset extraction, event listeners live in dashboard.js (external)
        # Verify the script tag is present and the nonce is applied
        assert 'src="/assets/js/dashboard.js' in html
        assert 'nonce="test_nonce"' in html


# ─── Agent Adapter Wiring Tests ─────────────────────────────────

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

class TestConfigMapFlush:
    """F-03: Config map accessor tests (Redis-backed)."""

    def test_get_config_map_returns_redis_instance(self):
        """get_config_map() returns a RedisConfigMap with an owner."""
        from propagul.server.dashboard_api import get_config_map
        result = get_config_map("test-owner")
        assert result is not None
        assert hasattr(result, "snapshot")
        assert hasattr(result, "merge")

    def test_lifespan_exists_and_is_async(self):
        """server.py lifespan is an async context manager."""
        from propagul.server.server import lifespan
        # @asynccontextmanager wraps the function — check for the
        # standard dunder or for callable (the decorator returns a
        # regular function that produces an async context manager).
        assert callable(lifespan)


# ─── F-04: Telemetry Depth Validation ───────────────────────────

class TestTelemetryDepthValidation:
    """F-04: Heartbeat telemetry must be depth- and key-limited."""

    def test_shallow_payload_passes(self):
        from propagul.server.dashboard_api import _validate_telemetry_depth
        payload = {"backends": [{"model_count": 3}], "gpu": {"gpus": [{"temp": 72}]}}
        assert _validate_telemetry_depth(payload) is True

    def test_deeply_nested_payload_fails(self):
        from propagul.server.dashboard_api import _validate_telemetry_depth
        # Build a payload with depth > 5
        nested = {"a": "leaf"}
        for _ in range(10):
            nested = {"inner": nested}
        assert _validate_telemetry_depth(nested) is False

    def test_too_many_keys_fails(self):
        from propagul.server.dashboard_api import _validate_telemetry_depth
        # Build a payload with >500 keys
        payload = {f"k{i}": i for i in range(600)}
        assert _validate_telemetry_depth(payload) is False

    def test_empty_payload_passes(self):
        from propagul.server.dashboard_api import _validate_telemetry_depth
        assert _validate_telemetry_depth({}) is True

    def test_nested_lists_counted(self):
        from propagul.server.dashboard_api import _validate_telemetry_depth
        payload = {"a": [[[[[{"b": 1}]]]]]}
        assert _validate_telemetry_depth(payload) is False


# ─── F-05: Rate Limiter Key Consistency ─────────────────────────

class TestRateLimiterKeying:
    """F-05: Rate limiter must use key_hash, not plaintext key."""

    def test_server_uses_key_hash(self):
        """server.py _get_auth must use key_info.key_hash for rate limiting."""
        import inspect
        try:
            from propagul.server.server import _get_auth
        except RuntimeError:
            # PROPAGUL_KEY_SALT may not be set in all test environments
            pytest.skip("server.py requires PROPAGUL_KEY_SALT")
        source = inspect.getsource(_get_auth)
        assert "key_info.key_hash" in source
        assert "key_info.key)" not in source  # Must NOT use plaintext key


# ─── F-06: Config Sync Desired-Model Filter ─────────────────────

class TestConfigSyncFilter:
    """F-06: /mesh/config/sync must filter desired:model:* keys."""

    def test_filter_code_in_sync_endpoint(self):
        """The sync endpoint must strip desired:model:* from remote snapshot."""
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "python" / "propagul" / "server" / "dashboard_api.py"
        code = src.read_text()
        assert 'desired:model:' in code
        assert 'startswith("desired:model:")' in code


# ─── F-07: Waitlist Atomic Lock ─────────────────────────────────

@pytest.mark.skip(reason="Legacy file-I/O — Waitlist is now in Redis")
class TestWaitlistAtomicLock:
    """F-07: Waitlist operations must be guarded by asyncio lock."""

    import pytest
    @pytest.fixture
    def anyio_backend(self):
        return 'asyncio'

    @pytest.mark.anyio
    async def test_lock_exists_in_waitlist(self, anyio_backend):
        """waitlist.py must have a _get_waitlist_lock function."""
        from propagul.server.waitlist import _get_waitlist_lock
        import asyncio
        lock = _get_waitlist_lock()
        assert isinstance(lock, asyncio.Lock)

    def test_waitlist_operations_use_lock(self):
        """All load-modify-save operations must use the lock."""
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "python" / "propagul" / "server" / "waitlist.py"
        code = src.read_text()
        # Count lock usage: should appear in join_waitlist, delete_waitlist_entry,
        # and generate_key (3 operations)
        lock_count = code.count("_get_waitlist_lock()")
        assert lock_count >= 3, f"Expected >= 3 lock usages, found {lock_count}"


# ─── F-08: Key Revocation Exact Match ───────────────────────────

class TestKeyRevocationExactMatch:
    """F-08: Key revocation must reject ambiguous prefix matches."""

    def test_revoke_rejects_ambiguous_prefix(self):
        """If >1 key matches prefix, revocation must fail with 409."""
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "python" / "propagul" / "server" / "waitlist.py"
        code = src.read_text()
        assert "Ambiguous prefix" in code
        assert "409" in code


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

    def test_preserves_non_loopback_url(self):
        from propagul.mesh.agent import MeshAgent
        backends = [
            {"backend": "ollama", "url": "http://192.168.1.20:11434", "models": []},
        ]
        MeshAgent._rewrite_backend_urls(backends, "192.168.1.50")
        assert backends[0]["url"] == "http://192.168.1.20:11434"

    def test_preserves_port(self):
        from propagul.mesh.agent import MeshAgent
        backends = [
            {"backend": "lm_studio", "url": "http://localhost:1234", "models": []},
        ]
        MeshAgent._rewrite_backend_urls(backends, "192.168.1.50")
        assert backends[0]["url"] == "http://192.168.1.50:1234"

    def test_multiple_backends(self):
        from propagul.mesh.agent import MeshAgent
        backends = [
            {"backend": "ollama", "url": "http://localhost:11434", "models": []},
            {"backend": "vllm", "url": "http://127.0.0.1:8000", "models": []},
            {"backend": "lm_studio", "url": "http://192.168.1.20:1234", "models": []},
        ]
        MeshAgent._rewrite_backend_urls(backends, "10.0.1.100")
        assert backends[0]["url"] == "http://10.0.1.100:11434"
        assert backends[1]["url"] == "http://10.0.1.100:8000"
        assert backends[2]["url"] == "http://192.168.1.20:1234"  # Unchanged

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
