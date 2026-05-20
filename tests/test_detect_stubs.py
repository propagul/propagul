"""test_detect_stubs — E2E auto-detection test using lightweight HTTP stubs.

Spins up 5 minimal HTTP servers that mimic the probe responses of:
- Ollama    (port 21434, GET /api/version → {"version": "0.4.8"})
- vLLM      (port 28000, GET /v1/models → {"data": [...]})
- TGI       (port 28080, GET /info → {"model_id": "...", "version": "2.4.1"})
- llama.cpp (port 28081, GET /health → {"status": "ok"})
- LM Studio (port 21234, GET /v1/models → {"data": [...]})

Then patches detect._PROBES to use these ports and runs detect.detect().
Asserts all 5 backends are detected with correct names.

Port range 21xxx/28xxx avoids collisions with real services.
"""

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from unittest.mock import patch

import pytest

from propagul.mesh.backends.detect import detect, _PROBES


# ─── Stub responses per backend ─────────────────────────────────────
STUB_RESPONSES = {
    "ollama": {
        "port": 21434,
        "routes": {
            "/api/version": {"version": "0.4.8"},
        },
    },
    "vllm": {
        "port": 28000,
        "routes": {
            "/v1/models": {
                "object": "list",
                "data": [{"id": "meta-llama/Llama-3-8B", "object": "model"}],
            },
        },
    },
    "tgi": {
        "port": 28080,
        "routes": {
            "/info": {
                "model_id": "meta-llama/Llama-3-8B",
                "version": "2.4.1",
                "sha": "abc123",
            },
        },
    },
    "llama_cpp": {
        "port": 28081,
        "routes": {
            "/health": {"status": "ok"},
        },
    },
    "lm_studio": {
        "port": 21234,
        "routes": {
            "/v1/models": {
                "object": "list",
                "data": [{"id": "lmstudio-community/Meta-Llama-3-8B-Q4", "object": "model"}],
            },
        },
    },
}


class StubHandler(BaseHTTPRequestHandler):
    """Generic handler that serves pre-configured JSON responses."""

    # Set per-instance by the factory
    routes: dict = {}

    def do_GET(self):
        if self.path in self.routes:
            body = json.dumps(self.routes[self.path]).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress noisy request logs during tests."""
        pass


def _make_handler(routes: dict):
    """Create a handler class with bound routes (closure)."""
    class BoundHandler(StubHandler):
        pass
    BoundHandler.routes = routes
    return BoundHandler


def _start_stub(name: str, config: dict) -> tuple:
    """Start a stub HTTP server in a daemon thread. Returns (server, thread)."""
    handler = _make_handler(config["routes"])
    server = HTTPServer(("127.0.0.1", config["port"]), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name=f"stub-{name}")
    thread.start()
    return server, thread


def _patched_probes() -> list:
    """Return _PROBES with URLs rewritten to use stub ports."""
    port_map = {name: cfg["port"] for name, cfg in STUB_RESPONSES.items()}
    patched = []
    for probe in _PROBES:
        p = dict(probe)
        name = p["name"]
        if name in port_map:
            p["urls"] = [f"http://127.0.0.1:{port_map[name]}"]
        else:
            # LocalAI — not stubbed, use non-reachable port
            p["urls"] = ["http://127.0.0.1:29999"]
        patched.append(p)
    return patched


@pytest.fixture(scope="module")
def stub_servers():
    """Start all 5 stub servers, yield, then shut them down."""
    servers = []
    for name, config in STUB_RESPONSES.items():
        srv, thr = _start_stub(name, config)
        servers.append(srv)
    # Give servers time to bind
    time.sleep(0.3)
    yield
    for srv in servers:
        srv.shutdown()


class TestBackendDetection:
    """Test auto-detection against stub HTTP servers."""

    def test_all_backends_detected(self, stub_servers):
        """All 5 stubbed backends should be detected."""
        with patch("propagul.mesh.backends.detect._PROBES", _patched_probes()):
            results = detect(timeout=2.0)

        detected_names = {r.name for r in results}
        expected = {"ollama", "vllm", "tgi", "llama_cpp", "lm_studio"}

        assert detected_names == expected, (
            f"Missing: {expected - detected_names}, "
            f"Extra: {detected_names - expected}"
        )

    def test_ollama_version_extracted(self, stub_servers):
        """Ollama probe should extract the version string."""
        with patch("propagul.mesh.backends.detect._PROBES", _patched_probes()):
            results = detect(timeout=2.0)

        ollama = [r for r in results if r.name == "ollama"][0]
        assert ollama.version == "0.4.8"
        assert ollama.confidence == 0.9  # Version present → high confidence

    def test_tgi_version_extracted(self, stub_servers):
        """TGI probe should extract version from /info."""
        with patch("propagul.mesh.backends.detect._PROBES", _patched_probes()):
            results = detect(timeout=2.0)

        tgi = [r for r in results if r.name == "tgi"][0]
        assert tgi.version == "2.4.1"

    def test_vllm_no_version(self, stub_servers):
        """vLLM has no version key — confidence should be 0.7."""
        with patch("propagul.mesh.backends.detect._PROBES", _patched_probes()):
            results = detect(timeout=2.0)

        vllm = [r for r in results if r.name == "vllm"][0]
        assert vllm.version == ""
        assert vllm.confidence == 0.7

    def test_llama_cpp_detected_on_separate_port(self, stub_servers):
        """llama.cpp on a separate port from TGI should both be detected."""
        with patch("propagul.mesh.backends.detect._PROBES", _patched_probes()):
            results = detect(timeout=2.0)

        llama = [r for r in results if r.name == "llama_cpp"][0]
        assert llama.url == "http://127.0.0.1:28081"

    def test_no_backends_on_dead_ports(self):
        """Detection should return empty when no servers are running."""
        dead_probes = []
        for probe in _PROBES:
            p = dict(probe)
            p["urls"] = ["http://127.0.0.1:39999"]
            dead_probes.append(p)

        with patch("propagul.mesh.backends.detect._PROBES", dead_probes):
            results = detect(timeout=0.5)

        assert results == []

    def test_partial_detection(self, stub_servers):
        """Only some stubs reachable — should detect only those."""
        partial_probes = _patched_probes()
        # Kill vLLM and TGI from the probe list by pointing to dead port
        for p in partial_probes:
            if p["name"] in ("vllm", "tgi"):
                p["urls"] = ["http://127.0.0.1:39999"]

        with patch("propagul.mesh.backends.detect._PROBES", partial_probes):
            results = detect(timeout=1.0)

        detected_names = {r.name for r in results}
        assert "ollama" in detected_names
        assert "lm_studio" in detected_names
        assert "llama_cpp" in detected_names
        assert "vllm" not in detected_names
        assert "tgi" not in detected_names


class TestDetectionEdgeCases:
    """Edge cases for the detection probe logic."""

    def test_tgi_before_llama_cpp_on_same_port(self):
        """When TGI and llama.cpp share a port, TGI should win (probe order).

        In production, both probe port 8080. TGI checks /info for 'model_id'.
        If TGI is found, the URL is marked as seen and llama.cpp is skipped.
        """
        # Spin up a TGI-only stub on a shared port
        tgi_routes = {
            "/info": {"model_id": "test-model", "version": "1.0"},
            "/health": {"status": "ok"},  # llama.cpp probe would also match this
        }
        handler = _make_handler(tgi_routes)
        server = HTTPServer(("127.0.0.1", 28082), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.2)

        try:
            # Both TGI and llama.cpp point to same port
            shared_probes = [
                {
                    "name": "tgi",
                    "urls": ["http://127.0.0.1:28082"],
                    "health_path": "/info",
                    "version_key": "version",
                    "sig_header": None,
                    "sig_body_key": "model_id",
                },
                {
                    "name": "llama_cpp",
                    "urls": ["http://127.0.0.1:28082"],
                    "health_path": "/health",
                    "version_key": None,
                    "sig_header": None,
                    "sig_body_key": "status",
                },
            ]

            with patch("propagul.mesh.backends.detect._PROBES", shared_probes):
                results = detect(timeout=2.0)

            names = [r.name for r in results]
            assert names == ["tgi"], f"Expected only TGI, got {names}"
        finally:
            server.shutdown()
