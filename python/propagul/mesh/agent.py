"""propagul.mesh.agent — Node Agent core.

The agent runs on each machine in the fleet. It:
1. Auto-detects the local inference engine (Ollama, vLLM, etc.)
2. Polls telemetry (models, GPU, system) every N seconds
3. Pushes telemetry to the Propagul Cloud dashboard
4. Listens for commands from the dashboard (pull model, delete, etc.)
5. Syncs fleet config via CRDT (desired models, node preferences)
6. Auto-pulls/deletes models to match desired state (opt-in via --auto-pull)

Zero external dependencies. Pure stdlib + propagul internals.
"""

import asyncio
import json
import logging
import os
import platform
import socket
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

from propagul.mesh.backends import detect as backend_detect
from propagul.mesh.backends import ollama as ollama_backend
from propagul.mesh.backends import vllm as vllm_backend
from propagul.mesh.backends import tgi as tgi_backend
from propagul.mesh.backends import lm_studio as lm_studio_backend
from propagul.mesh.backends import llamacpp as llamacpp_backend
from propagul.mesh import gpu as gpu_collector

logger = logging.getLogger("propagul.mesh.agent")

# Defaults
DEFAULT_POLL_INTERVAL = 10  # seconds
DEFAULT_HEARTBEAT_INTERVAL = 30  # seconds
DEFAULT_DASHBOARD_URL = "https://cloud.propagul.dev"


@dataclass
class NodeIdentity:
    """Unique identity of this node in the fleet."""
    node_id: str  # User-chosen name or auto-generated
    hostname: str = field(default_factory=socket.gethostname)
    platform: str = field(default_factory=lambda: platform.system())
    arch: str = field(default_factory=lambda: platform.machine())
    python_version: str = field(default_factory=lambda: platform.python_version())
    agent_version: str = field(default_factory=lambda: _get_agent_version())
    local_ip: str = field(default_factory=lambda: _get_local_ip())

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "platform": self.platform,
            "arch": self.arch,
            "python_version": self.python_version,
            "agent_version": self.agent_version,
            "local_ip": self.local_ip,
            "uptime_seconds": _get_uptime_seconds(),
        }


def _get_agent_version() -> str:
    """Get propagul-mesh package version. Falls back to 'dev'."""
    try:
        from importlib.metadata import version
        return version("propagul-mesh")
    except Exception:
        return "dev"


def _get_local_ip() -> str:
    """Get the local IP address via UDP connect trick (no traffic sent).

    Creates a UDP socket and connects to a public IP to determine
    which local interface would be used for outbound traffic.
    """
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        return ip
    except Exception:
        return "127.0.0.1"
    finally:
        if s is not None:
            try:
                s.close()
            except Exception:
                pass


def _get_uptime_seconds() -> float:
    """Get system uptime from /proc/uptime (Linux) or sysctl (macOS)."""
    try:
        if platform.system() == "Linux":
            with open("/proc/uptime", "r") as f:
                return float(f.read().split()[0])
        elif platform.system() == "Darwin":
            import subprocess
            out = subprocess.check_output(
                ["sysctl", "-n", "kern.boottime"], timeout=2,
            ).decode()
            # Format: { sec = 1700000000, usec = 0 }
            import re
            m = re.search(r"sec\s*=\s*(\d+)", out)
            if m:
                boot_ts = int(m.group(1))
                return time.time() - boot_ts
    except Exception:
        pass
    return 0.0


@dataclass
class TelemetrySnapshot:
    """Complete telemetry snapshot for a single poll cycle."""
    timestamp: float
    node: dict
    backends: list[dict]
    gpu: dict
    system: dict

    def to_json(self) -> str:
        return json.dumps({
            "timestamp": self.timestamp,
            "node": self.node,
            "backends": self.backends,
            "gpu": self.gpu,
            "system": self.system,
        })


def _read_cpu_percent() -> float:
    """Read CPU utilization % cross-platform.

    Linux: /proc/stat delta (two samples 100ms apart).
    Windows: wmic cpu get loadpercentage (one-shot).
    macOS: vm_stat + sysctl (not yet implemented, returns 0.0).

    Returns 0.0 on unsupported platform or on any error.
    """
    system = platform.system()

    if system == "Linux":
        try:
            def _read_stat():
                with open("/proc/stat", "r") as f:
                    for line in f:
                        if line.startswith("cpu "):
                            parts = line.split()
                            values = [int(x) for x in parts[1:8]]
                            idle = values[3]
                            total = sum(values)
                            return idle, total
                return 0, 0

            idle1, total1 = _read_stat()
            time.sleep(0.1)
            idle2, total2 = _read_stat()

            delta_total = total2 - total1
            delta_idle = idle2 - idle1
            if delta_total <= 0:
                return 0.0
            return round((1.0 - delta_idle / delta_total) * 100, 1)
        except Exception:
            return 0.0

    elif system == "Windows":
        # Strategy: try wmic first (fast), fall back to PowerShell (wmic deprecated in Win11 24H2)
        import subprocess
        try:
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "loadpercentage"],
                timeout=3, stderr=subprocess.DEVNULL,
            ).decode()
            for line in out.strip().splitlines():
                line = line.strip()
                if line.isdigit():
                    return float(line)
        except Exception:
            pass

        # Fallback: PowerShell Get-CimInstance (works on all modern Windows)
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).LoadPercentage"],
                timeout=5, stderr=subprocess.DEVNULL,
            ).decode().strip()
            if out.isdigit():
                return float(out)
        except Exception:
            pass
        return 0.0

    return 0.0


def _read_ram_info() -> dict:
    """Read system RAM cross-platform.

    Linux: /proc/meminfo (MemAvailable for accurate used-memory).
    Windows: kernel32.GlobalMemoryStatusEx via ctypes (zero deps).
    macOS: sysctl hw.memsize + vm_stat.

    Returns dict with total_mb, used_mb, available_mb, percent.
    """
    result = {"total_mb": 0, "used_mb": 0, "available_mb": 0, "percent": 0.0}
    system = platform.system()

    if system == "Linux":
        try:
            meminfo = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        meminfo[key] = int(parts[1])  # in kB

            total_kb = meminfo.get("MemTotal", 0)
            available_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
            used_kb = total_kb - available_kb

            result["total_mb"] = total_kb // 1024
            result["available_mb"] = available_kb // 1024
            result["used_mb"] = used_kb // 1024
            if total_kb > 0:
                result["percent"] = round(used_kb / total_kb * 100, 1)
        except Exception:
            pass

    elif system == "Windows":
        try:
            import ctypes
            import ctypes.wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.wintypes.DWORD),
                    ("dwMemoryLoad", ctypes.wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))

            result["total_mb"] = mem.ullTotalPhys // (1024 * 1024)
            result["available_mb"] = mem.ullAvailPhys // (1024 * 1024)
            result["used_mb"] = result["total_mb"] - result["available_mb"]
            result["percent"] = round(mem.dwMemoryLoad, 1)
        except Exception:
            pass

    elif system == "Darwin":
        try:
            import subprocess
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], timeout=2,
            ).decode().strip()
            total_bytes = int(out)
            result["total_mb"] = total_bytes // (1024 * 1024)
            vm_out = subprocess.check_output(
                ["vm_stat"], timeout=2,
            ).decode()
            page_size = 4096
            free_pages = 0
            for line in vm_out.splitlines():
                if "Pages free" in line:
                    free_pages += int(line.split(":")[1].strip().rstrip("."))
                elif "Pages inactive" in line:
                    free_pages += int(line.split(":")[1].strip().rstrip("."))
            avail_bytes = free_pages * page_size
            result["available_mb"] = avail_bytes // (1024 * 1024)
            result["used_mb"] = result["total_mb"] - result["available_mb"]
            if result["total_mb"] > 0:
                result["percent"] = round(
                    result["used_mb"] / result["total_mb"] * 100, 1,
                )
        except Exception:
            pass

    return result


def _read_disk_info(path: str = "/") -> dict:
    """Read disk usage via stdlib shutil.disk_usage.

    Returns dict with total_gb, used_gb, free_gb, percent.
    Uses the root filesystem by default — this is where models
    typically reside (/usr/share/ollama/.ollama/models or ~/.ollama).
    """
    import shutil
    result = {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0, "percent": 0.0}
    try:
        usage = shutil.disk_usage(path)
        result["total_gb"] = round(usage.total / (1024**3), 1)
        result["used_gb"] = round(usage.used / (1024**3), 1)
        result["free_gb"] = round(usage.free / (1024**3), 1)
        if usage.total > 0:
            result["percent"] = round(usage.used / usage.total * 100, 1)
    except Exception:
        pass
    return result


def _system_info() -> dict:
    """Collect comprehensive system metrics (CPU, RAM, Disk).

    Zero external dependencies — uses /proc (Linux), sysctl (macOS),
    and stdlib shutil.disk_usage. Every metric is fail-safe: if a
    source is unavailable, its value defaults to 0.
    """
    ram = _read_ram_info()
    disk = _read_disk_info()

    return {
        "cpu_count": os.cpu_count() or 1,
        "cpu_percent": _read_cpu_percent(),
        "load_avg": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
        "ram_total_mb": ram["total_mb"],
        "ram_used_mb": ram["used_mb"],
        "ram_available_mb": ram["available_mb"],
        "ram_percent": ram["percent"],
        "disk_total_gb": disk["total_gb"],
        "disk_used_gb": disk["used_gb"],
        "disk_free_gb": disk["free_gb"],
        "disk_percent": disk["percent"],
    }


class MeshAgent:
    """The mesh node agent.

    Usage:
        agent = MeshAgent(node_id="my-workstation", api_key="pg_mesh_...")
        await agent.start()
    """

    def __init__(
        self,
        node_id: str,
        api_key: str = "",
        dashboard_url: str = DEFAULT_DASHBOARD_URL,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        heartbeat_interval: int = DEFAULT_HEARTBEAT_INTERVAL,
        backend_url: Optional[str] = None,
        on_command: Optional[Callable] = None,
        auto_pull: bool = False,
        proxy_port: int = 0,
        proxy_backend_auth: str = "",
        advertise_ip: Optional[str] = None,
    ):
        self._identity = NodeIdentity(node_id=node_id)
        self._api_key = api_key
        self._dashboard_url = dashboard_url.rstrip("/")
        self._poll_interval = poll_interval
        self._heartbeat_interval = heartbeat_interval
        self._backend_url = backend_url  # Override auto-detection
        self._on_command = on_command
        self._auto_pull = auto_pull
        self._proxy_port = proxy_port  # 0 = disabled, >0 = start local proxy
        self._proxy_backend_auth = proxy_backend_auth  # Auth for backend requests
        self._running = False
        self._detected_backends: list = []
        self._last_snapshot: Optional[TelemetrySnapshot] = None
        self._stats = {
            "polls": 0, "pushes": 0, "errors": 0, "commands": 0,
            "config_syncs": 0, "auto_pulls": 0, "auto_deletes": 0,
        }
        self._pull_in_progress = False  # Guard against concurrent reconciliation
        self._cancel_pulls: set = set()  # Models to cancel pulling

        # LAN-routable IP for multi-machine fleet routing.
        # When this agent's backend URLs are sent to the dashboard, localhost
        # and 127.0.0.1 are replaced with this IP so other agents in the LAN
        # can reach this node's backends directly (prompt data stays in LAN).
        # Priority: --advertise-ip flag > auto-detected LAN IP > 127.0.0.1
        self._advertise_ip = advertise_ip or self._identity.local_ip
        if self._advertise_ip and self._advertise_ip != "127.0.0.1":
            logger.info(
                "Advertise IP: %s (other agents will route to this address)",
                self._advertise_ip,
            )

        # Fleet-aware routing: shared state between agent and proxy.
        # Updated from heartbeat response, read by proxy for each request.
        from propagul.mesh.router import FleetState, RoutingEventLog
        self._fleet_state = FleetState()
        self._routing_log = RoutingEventLog(maxlen=50)

        # CRDT config-sync: local config map for eventual consistency
        # Uses a hash of node_id as numeric CRDT node identifier
        self._config_node_id = abs(hash(node_id)) % (2**31)
        from propagul.mesh.config import FleetConfigMap
        self._config_map = FleetConfigMap(node_id=self._config_node_id)

    async def start(self) -> None:
        """Start the agent loop. Blocks until stop() is called."""
        logger.info("Starting mesh agent: node_id=%s", self._identity.node_id)

        # Auto-detect backends
        self._detected_backends = backend_detect.detect()
        if self._backend_url:
            logger.info("Backend URL override: %s", self._backend_url)

        if not self._detected_backends and not self._backend_url:
            logger.warning("No inference engines detected — will retry on each poll")

        self._running = True

        # Build task list: telemetry + commands + optional proxy
        tasks = [
            self._poll_loop(),
            self._command_loop(),
        ]

        # Start local reverse proxy if enabled
        if self._proxy_port > 0:
            tasks.append(self._start_proxy())

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Agent stopped")
        finally:
            self._running = False

    async def _start_proxy(self) -> None:
        """Start the local OpenAI-compatible reverse proxy.

        Routes localhost:<port>/v1/* to the detected local backend.
        This enables tools like Open Interpreter, LangChain, etc. to
        connect to a single stable endpoint regardless of which
        inference engine is running locally.
        """
        from propagul.mesh.proxy import start_proxy, ProxyConfig

        # Determine backend to route to
        backend_name = ""
        backend_url = ""

        if self._backend_url:
            backend_url = self._backend_url
            # Infer name from URL pattern
            if ":11434" in backend_url:
                backend_name = "ollama"
            elif ":8000" in backend_url:
                backend_name = "vllm"
            elif ":1234" in backend_url:
                backend_name = "lm_studio"
            else:
                backend_name = "unknown"
        elif self._detected_backends:
            backend_name = self._detected_backends[0].name
            backend_url = self._detected_backends[0].url
        else:
            logger.warning("Proxy enabled but no backend detected — proxy will return 503")

        config = ProxyConfig(
            host="127.0.0.1",
            port=self._proxy_port,
            backend_name=backend_name,
            backend_url=backend_url,
            backend_auth=self._proxy_backend_auth,
        )

        logger.info(
            "Starting local proxy on port %d → %s (%s)",
            self._proxy_port, backend_url or "none", backend_name or "none",
        )
        await start_proxy(
            config,
            fleet_state=self._fleet_state,
            local_node_id=self._identity.node_id,
            routing_log=self._routing_log,
        )

    async def stop(self) -> None:
        """Signal the agent to stop."""
        self._running = False

    async def _poll_loop(self) -> None:
        """Main polling loop: collect telemetry, push to dashboard.

        AG-01: Both _collect_telemetry() and _push_telemetry() are blocking
        (CPU polling with time.sleep, urllib HTTP calls). Running them in
        asyncio.to_thread() prevents blocking the event loop, which would
        otherwise add latency to concurrent proxy requests.
        """
        while self._running:
            try:
                snapshot = await asyncio.to_thread(self._collect_telemetry)
                self._last_snapshot = snapshot
                self._stats["polls"] += 1

                if self._api_key:
                    await asyncio.to_thread(self._push_telemetry, snapshot)
                    self._stats["pushes"] += 1
                else:
                    # Local-only mode: just log
                    logger.debug("Telemetry collected (no API key, local-only)")

            except Exception as e:
                self._stats["errors"] += 1
                logger.error("Poll error: %s", e)

            await asyncio.sleep(self._poll_interval)

    async def _command_loop(self) -> None:
        """Poll for pending commands from the dashboard.

        Polls every 2 seconds (not heartbeat interval) to minimize
        latency for pull/delete commands. The GET request is
        lightweight (no payload), so 0.5 req/s per node is negligible.
        """
        while self._running:
            if self._api_key:
                try:
                    self._fetch_and_execute_commands()
                except Exception as e:
                    logger.debug("Command poll error: %s", e)

            await asyncio.sleep(2)  # 2s for fast inference pickup

    def _collect_telemetry(self) -> TelemetrySnapshot:
        """Collect a full telemetry snapshot."""
        # Re-detect backends periodically (every 6th poll = ~60s).
        # This catches backends started after the agent (e.g. LM Studio,
        # a second vLLM instance). Without this, only backends running at
        # agent start time would be discovered.
        rescan_interval = 6  # polls
        if (not self._detected_backends and not self._backend_url) or \
           (self._stats["polls"] % rescan_interval == 0):
            fresh = backend_detect.detect()
            if fresh != self._detected_backends:
                old_names = {b.name for b in self._detected_backends}
                new_names = {b.name for b in fresh}
                added = new_names - old_names
                removed = old_names - new_names
                if added:
                    logger.info("New backends detected: %s", added)
                if removed:
                    logger.info("Backends gone: %s", removed)
                self._detected_backends = fresh

        backends_data: list[dict] = []

        # Poll each detected backend
        for backend in self._detected_backends:
            url = self._backend_url or backend.url
            if backend.name == "ollama":
                status = ollama_backend.poll(base_url=url)
                backends_data.append(status.to_dict())
            elif backend.name == "vllm":
                telemetry = vllm_backend.collect_telemetry(base_url=url)
                backends_data.append(telemetry)
            elif backend.name == "tgi":
                telemetry = tgi_backend.collect_telemetry(base_url=url)
                backends_data.append(telemetry)
            elif backend.name == "lm_studio":
                telemetry = lm_studio_backend.collect_telemetry(base_url=url)
                backends_data.append(telemetry)
            elif backend.name == "llama_cpp":
                telemetry = llamacpp_backend.collect_telemetry(base_url=url)
                backends_data.append(telemetry)
            else:
                # Unknown backend — basic telemetry stub
                backends_data.append({
                    "backend": backend.name,
                    "url": backend.url,
                    "online": True,
                    "version": backend.version,
                    "model_count": 0,
                    "running_count": 0,
                    "models": [],
                })

        # If manual URL provided but no auto-detect, try Ollama directly
        if not self._detected_backends and self._backend_url:
            status = ollama_backend.poll(base_url=self._backend_url)
            backends_data.append(status.to_dict())

        # GPU metrics
        gpu_status = gpu_collector.collect()

        return TelemetrySnapshot(
            timestamp=time.time(),
            node=self._identity.to_dict(),
            backends=backends_data,
            gpu=gpu_status.to_dict(),
            system=_system_info(),
        )

    @staticmethod
    def _rewrite_backend_urls(
        backends: list[dict], advertise_ip: str,
    ) -> list[dict]:
        """Replace non-advertise-ip hostnames in backend URLs with the advertise IP.

        This is critical for multi-machine fleets: when the heartbeat
        sends backend URLs to the dashboard, other agents receive these URLs
        in the fleet routing table. Remote agents must be able to reach this
        node's backends via the advertise IP.

        Rewrites ANY hostname that is not already the advertise_ip. This covers:
        - Loopback addresses (localhost, 127.0.0.1, ::1)
        - WireGuard/VPN IPs (10.100.0.x) unreachable from other networks
        - Docker bridge IPs (172.17.x.x, 192.168.x.x) unreachable externally

        The local proxy continues to use the original URLs (from detection or
        --ollama flag) for actual backend access. Only the heartbeat telemetry
        is rewritten.

        Args:
            backends: List of backend telemetry dicts (each has a 'url' key).
            advertise_ip: The fleet-routable IP (e.g. Tailscale 100.x.x.x).

        Returns:
            The same list with URLs rewritten in-place.
        """
        if not advertise_ip or advertise_ip == "127.0.0.1":
            return backends  # No usable routable IP — leave URLs as-is

        for backend in backends:
            url = backend.get("url", "")
            if not url:
                continue
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(url)
            if parsed.hostname and parsed.hostname != advertise_ip:
                # Replace hostname, preserve port and scheme
                port_suffix = f":{parsed.port}" if parsed.port else ""
                new_netloc = f"{advertise_ip}{port_suffix}"
                rewritten = urlunparse((
                    parsed.scheme, new_netloc,
                    parsed.path, parsed.params,
                    parsed.query, parsed.fragment,
                ))
                backend["url"] = rewritten

        return backends

    def _push_telemetry(self, snapshot: TelemetrySnapshot) -> None:
        """Push telemetry to the dashboard API.

        Endpoint: POST /mesh/nodes/{node_id}/heartbeat
        Auth: Bearer <api_key>
        """
        url = f"{self._dashboard_url}/mesh/nodes/{self._identity.node_id}/heartbeat"
        # Build telemetry body, including routing events if any
        body_dict = json.loads(snapshot.to_json())

        # Rewrite backend URLs: localhost → LAN IP for fleet routing.
        # Without this, remote agents in the fleet would try to connect
        # to their own localhost instead of this node's backend.
        if "backends" in body_dict:
            self._rewrite_backend_urls(
                body_dict["backends"], self._advertise_ip,
            )

        routing_events = self._routing_log.drain()
        if routing_events:
            body_dict["routing_events"] = routing_events
        body = json.dumps(body_dict).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status != 200:
                    logger.warning("Dashboard push returned %d", resp.status)
                else:
                    # UX Fix: Log at INFO so user sees the agent is working
                    logger.info("Heartbeat sent to dashboard (200 OK)")
                    # Parse fleet routing table from heartbeat response.
                    # This enables the local proxy to route requests to
                    # remote backends in the LAN (model-affinity routing).
                    try:
                        resp_body = json.loads(resp.read().decode("utf-8"))
                        if "fleet_routing" in resp_body:
                            from propagul.mesh.router import parse_fleet_routing_table
                            targets = parse_fleet_routing_table(resp_body)
                            self._fleet_state.update(targets)
                    except Exception as e:
                        logger.debug("Fleet routing parse failed: %s", e)
        except Exception as e:
            logger.debug("Dashboard push failed: %s", e)

        # Config sync: pull fleet config CRDT from dashboard, merge locally
        self._sync_config()

        # Auto-pull: reconcile desired models against local state (async, non-blocking)
        if self._auto_pull and not self._pull_in_progress:
            task = asyncio.create_task(self._async_reconcile(snapshot))
            task.add_done_callback(self._reconcile_done_callback)

    def _fetch_and_execute_commands(self) -> None:
        """Fetch pending commands from dashboard and execute them."""
        url = f"{self._dashboard_url}/mesh/nodes/{self._identity.node_id}/commands"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                commands = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return

        for cmd in commands.get("pending", []):
            command = cmd.get("command", "")
            model = cmd.get("model", "")

            # Inference relay: not supported in free-tier agent.
            # Pro-tier relay is a separate module (propagul-relay).
            if command == "inference":
                logger.warning(
                    "Inference command ignored — relay not enabled. "
                    "Upgrade to Pro for cloud inference relay."
                )
                continue

            logger.info("Executing command: %s %s", command, model)

            # F-01: Route commands to correct backend.
            backend_url = self._backend_url
            backend_name = "ollama"  # default
            if not backend_url and self._detected_backends:
                backend_url = self._detected_backends[0].url
                backend_name = self._detected_backends[0].name

            if not backend_url:
                continue

            result = None  # Initialize for on_command callback

            if backend_name == "ollama" and command == "pull":
                # Stream pull progress back to dashboard
                t = threading.Thread(
                    target=self._handle_pull_with_progress,
                    args=(model, backend_url),
                    daemon=True,
                )
                t.start()
            elif command == "cancel_pull":
                # Signal the pull thread to stop
                self._cancel_pulls.add(model)
                logger.info("Cancel pull requested: %s", model)
            elif command == "eject" and backend_name == "ollama":
                # VRAM eject: unload all running models from GPU memory.
                # Uses POST /api/generate with keep_alive=0 per running model.
                result = ollama_backend.eject_all(base_url=backend_url)
                ejected_count = len(result.get("ejected", []))
                failed_count = len(result.get("failed", []))
                self._stats["ejects"] = self._stats.get("ejects", 0) + 1
                if result.get("status") == "ok":
                    logger.info(
                        "VRAM eject complete: %d models unloaded, "
                        "%d before → %d after",
                        ejected_count,
                        result.get("running_before", 0),
                        result.get("running_after", 0),
                    )
                elif result.get("status") == "partial":
                    logger.warning(
                        "VRAM eject partial: %d ejected, %d failed",
                        ejected_count, failed_count,
                    )
                else:
                    logger.error("VRAM eject failed: %s", result.get("error", "unknown"))
            elif command == "eject":
                # vLLM, TGI, LM Studio: no runtime model unloading.
                # Models are loaded at server start and persist until restart.
                result = {
                    "status": "error",
                    "error": (
                        f"Backend '{backend_name}' does not support VRAM eject. "
                        f"Models are loaded at server start and persist until "
                        f"the inference server is restarted."
                    ),
                }
                logger.warning(
                    "Eject rejected: backend '%s' is read-only (no runtime unload)",
                    backend_name,
                )
            elif backend_name == "ollama":
                result = ollama_backend.execute_command(
                    command=command, model=model, base_url=backend_url,
                )
            else:
                # vLLM, TGI, LM Studio: read-only — no remote model management
                result = {
                    "status": "error",
                    "error": (
                        f"Backend '{backend_name}' does not support remote "
                        f"model management ({command}). Models must be "
                        f"configured at server start."
                    ),
                }
                logger.warning(
                    "Command '%s %s' rejected: backend '%s' is read-only",
                    command, model, backend_name,
                )

            self._stats["commands"] += 1

            if self._on_command:
                self._on_command(cmd, result)


    @property
    def stats(self) -> dict:
        return dict(self._stats)

    @property
    def last_snapshot(self) -> Optional[TelemetrySnapshot]:
        return self._last_snapshot

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def desired_models(self) -> dict:
        """Get desired model state from fleet config CRDT.

        Returns dict mapping model name → action ('pull' or 'delete').
        Updated on each heartbeat cycle via config sync.
        """
        return self._config_map.get_desired_models()

    @property
    def config_map(self):
        """Access the local CRDT config map (for inspection/testing)."""
        return self._config_map

    def _sync_config(self) -> None:
        """Sync fleet config CRDT with the dashboard.

        Pull: GET /mesh/config → merge server snapshot into local CRDT.
        Push: POST /mesh/config/sync → send local snapshot, receive merged result.

        This runs in the heartbeat thread, so it is blocking (urllib).
        Config sync is non-fatal: if the server is unreachable, the local
        config map retains its last known state.
        """
        if not self._api_key:
            return

        # Step 1: Pull server config snapshot
        config_url = f"{self._dashboard_url}/mesh/config"
        req = urllib.request.Request(
            config_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                server_snapshot = data.get("snapshot")
                if isinstance(server_snapshot, dict):
                    delta = self._config_map.merge(server_snapshot)
                    if delta > 0:
                        logger.info("Config sync: merged %d entries from server", delta)
        except Exception as e:
            logger.debug("Config pull failed (non-fatal): %s", e)
            return  # Don't push if pull failed

        # Step 2: Push local config changes to server
        local_snapshot = self._config_map.snapshot()
        # Only push if we have local state
        if local_snapshot.get("entries"):
            sync_url = f"{self._dashboard_url}/mesh/config/sync"
            body = json.dumps({"snapshot": local_snapshot}).encode("utf-8")
            push_req = urllib.request.Request(
                sync_url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(push_req, timeout=5) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    # Merge the server's response back (bidirectional)
                    merged_snapshot = result.get("snapshot")
                    if isinstance(merged_snapshot, dict):
                        self._config_map.merge(merged_snapshot)
            except Exception as e:
                logger.debug("Config push failed (non-fatal): %s", e)

        self._stats["config_syncs"] += 1

    def _reconcile_desired_models(
        self, snapshot: Optional[TelemetrySnapshot] = None,
    ) -> list[dict]:
        """Reconcile desired model state against locally installed models.

        Compares fleet-desired models (from CRDT config map) against the
        models actually present on this node (from last telemetry snapshot).

        Actions:
            - desired=pull, not installed → pull (download)
            - desired=delete, installed → delete (remove)
            - desired=pull, already installed → no-op
            - desired=delete, not installed → no-op

        Returns list of action results for logging/testing.

        Safety:
            - Only runs when auto_pull=True (opt-in)
            - Uses the same backend URL as manual commands
            - Non-fatal: errors are logged but don't crash the agent
            - Pull timeout is generous (300s) for large models
        """
        results: list[dict] = []
        desired = self.desired_models
        if not desired:
            return results

        # Get locally installed model names from last snapshot
        if snapshot is None:
            snapshot = self._last_snapshot
        if snapshot is None:
            return results

        local_models: set[str] = set()
        for backend_data in snapshot.backends:
            for model_info in backend_data.get("models", []):
                name = model_info.get("name", "")
                if name:
                    local_models.add(name)

        # Resolve backend URL
        backend_url = self._backend_url
        if not backend_url and self._detected_backends:
            backend_url = self._detected_backends[0].url
        if not backend_url:
            return results  # No backend to execute against

        for model, action in desired.items():
            if action == "pull" and model not in local_models:
                logger.info("Auto-pull: pulling %s (desired state)", model)
                try:
                    result = ollama_backend.execute_command(
                        command="pull", model=model, base_url=backend_url,
                        timeout=300.0,  # Large models can take minutes
                    )
                    self._stats["auto_pulls"] += 1
                    if result.get("status") == "error":
                        logger.warning(
                            "Auto-pull failed for %s: %s",
                            model, result.get("error"),
                        )
                    else:
                        logger.info("Auto-pull complete: %s", model)
                    results.append({"model": model, "action": "pull", **result})
                except Exception as e:
                    logger.error("Auto-pull error for %s: %s", model, e)
                    results.append({"model": model, "action": "pull", "status": "error", "error": str(e)})

            elif action == "delete" and model in local_models:
                logger.info("Auto-delete: removing %s (desired state)", model)
                try:
                    result = ollama_backend.execute_command(
                        command="delete", model=model, base_url=backend_url,
                    )
                    self._stats["auto_deletes"] += 1
                    if result.get("status") == "error":
                        logger.warning(
                            "Auto-delete failed for %s: %s",
                            model, result.get("error"),
                        )
                    else:
                        logger.info("Auto-delete complete: %s", model)
                    results.append({"model": model, "action": "delete", **result})
                except Exception as e:
                    logger.error("Auto-delete error for %s: %s", model, e)
                    results.append({"model": model, "action": "delete", "status": "error", "error": str(e)})

        return results

    def _handle_pull_with_progress(self, model: str, backend_url: str) -> None:
        """Pull a model with streaming progress reported to dashboard.

        Runs in a daemon thread. Streams Ollama NDJSON progress,
        throttled to one update per 500ms to avoid request flood.
        """
        logger.info("Pull with progress: %s from %s", model, backend_url)
        progress_url = f"/mesh/nodes/{self._identity.node_id}/pull-progress"
        last_post = 0.0

        try:
            for event in ollama_backend.execute_pull_streaming(
                model=model, base_url=backend_url,
            ):
                status = event.get("status", "")
                total = event.get("total", 0)
                completed = event.get("completed", 0)

                # Throttle: max 2 updates/sec
                now = time.time()
                if now - last_post < 0.5 and status not in ("success", "error"):
                    continue

                # Check for cancel signal
                if model in self._cancel_pulls:
                    self._cancel_pulls.discard(model)
                    logger.info("Pull cancelled by user: %s", model)
                    self._post_to_dashboard(progress_url, {
                        "model": model, "status": "cancelled",
                        "total": 0, "completed": 0, "percent": 0,
                    })
                    return

                last_post = now

                progress = {
                    "model": model,
                    "status": status,
                    "total": total,
                    "completed": completed,
                    "percent": round(completed / total * 100, 1) if total > 0 else 0,
                }
                self._post_to_dashboard(progress_url, progress)

                if status == "success":
                    logger.info("Pull complete: %s", model)
                    self._stats["auto_pulls"] += 1
                elif status == "error":
                    logger.warning("Pull failed: %s — %s", model, event.get("error"))

        except Exception as e:
            logger.error("Pull progress error: %s %s", model, e)
            self._post_to_dashboard(progress_url, {
                "model": model, "status": "error",
                "total": 0, "completed": 0, "percent": 0,
            })


    def _post_to_dashboard(self, path: str, body: dict) -> None:
        """POST JSON to dashboard API. Used for pull-progress reporting."""
        url = f"{self._dashboard_url}{path}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                pass  # Fire and forget
        except Exception as e:
            logger.debug("Dashboard POST failed (%s): %s", path, e)

    def _reconcile_done_callback(self, task) -> None:
        """Callback for reconciliation task completion.

        Logs unhandled exceptions from fire-and-forget tasks instead of
        silently losing them (AG-03).
        """
        try:
            exc = task.exception()
            if exc is not None:
                logger.error("Reconciliation task failed: %s", exc)
        except asyncio.CancelledError:
            pass  # Task was cancelled, not an error


    async def _async_reconcile(
        self, snapshot: Optional[TelemetrySnapshot] = None,
    ) -> None:
        """Run model reconciliation in a background thread.

        Uses asyncio.to_thread() to offload the blocking ollama
        HTTP calls to a thread pool, so the heartbeat loop is
        never blocked by multi-minute model downloads.

        Concurrency guard: only one reconciliation runs at a time.
        If a previous reconciliation is still in progress, the
        current heartbeat skips reconciliation (no queue buildup).
        """
        self._pull_in_progress = True
        try:
            results = await asyncio.to_thread(
                self._reconcile_desired_models, snapshot,
            )
            if results:
                logger.info(
                    "Async reconciliation complete: %d actions",
                    len(results),
                )
        except Exception as e:
            logger.error("Async reconciliation error: %s", e)
        finally:
            self._pull_in_progress = False


