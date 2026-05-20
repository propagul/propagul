"""conftest.py — pytest configuration for propagul tests.

Most test files in this directory are standalone scripts (runnable via
``python tests/test_foo.py``). They use sys.exit() at module level.
To run them under pytest, we need to suppress sys.exit during collection.

Only the files with proper ``test_*`` functions are collected by pytest.
Standalone scripts that call sys.exit() at import time are excluded via
the ``collect_ignore`` list below.
"""

import sys
import os

import pytest

# Add python/ to path for propagul imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

# ─── Environment Setup ──────────────────────────────────────────
# Auth tests need PROPAGUL_KEY_SALT set before import.
# Use a deterministic test-only salt (never reuse in production).
os.environ.setdefault("PROPAGUL_KEY_SALT", "test_salt_for_pytest_only_do_not_use_in_prod_0123456789abcdef")


# ─── Redis Test Isolation ────────────────────────────────────────
# Each test function gets a clean Redis DB to prevent data leaks.
# Uses DB 15 (highest standard DB) to prevent accidental production data loss.
# CRITICAL: DB 0 is the default production DB — never flush it in tests.
_REDIS_TEST_DB = int(os.environ.get("REDIS_TEST_DB", "15"))

@pytest.fixture(autouse=True)
def _redis_test_isolation(monkeypatch):
    """Flush Redis test DB before each test to ensure isolation.

    Uses DB 15 (not DB 0) to prevent accidental production data destruction.
    Patches redis_store.get_redis_client() to use the test DB so all
    server code under test operates on the isolated DB.
    """
    import redis as _redis
    r = _redis.Redis(host='localhost', port=6379, db=_REDIS_TEST_DB, decode_responses=True)
    try:
        r.flushdb()
    except Exception:
        pass  # Redis not available — skip flush, tests will fail naturally

    # Patch redis_store to use the test DB for all server code
    try:
        from propagul.server import redis_store

        def _test_get_redis():
            """Return a Redis client pointing at the test DB."""
            return _redis.Redis(
                host='localhost', port=6379, db=_REDIS_TEST_DB,
                decode_responses=True,
            )

        monkeypatch.setattr(redis_store, "get_redis", _test_get_redis)
        # Force pool re-init on next call by clearing cached state
        monkeypatch.setattr(redis_store, "_redis_pool", None)
        monkeypatch.setattr(redis_store, "_redis_url", None)
    except ImportError:
        pass  # redis_store not importable in this context

    yield

    # Post-test cleanup
    try:
        r.flushdb()
    except Exception:
        pass


# Standalone test scripts that use sys.exit() at module level or require
# TCP ports / asyncio.run() / Python 3.11+ deps that conflict with pytest.
# These must be run directly: python tests/test_foo.py
# Converted to pytest-compatible structure (uses assert, __name__ guard):
#   test_hardening.py (NOT in this list, runs with .venv)
#   test_e2e_frameworks.py (IN this list — needs .venv311 for CrewAI/LangGraph)
collect_ignore = [
    "test_real_integration.py",
    "test_p2p_checkpointer.py",
    "test_e2e_frameworks.py",  # Needs .venv311 (CrewAI/LangGraph not in 3.9)
    "test_integrations.py",
    "test_server.py",
    "test_http_api.py",
    "test_persistence_partition.py",
    "test_p2p_python.py",
    "test_gossip_e2e.py",  # TCP gossip tests hang under pytest's event loop
]

# Legacy Rust-dependent tests (require compiled entropy_state native module).
# Excluded via glob to cover the entire _legacy_rust/ subtree.
collect_ignore_glob = [
    "_legacy_rust/*",
]
