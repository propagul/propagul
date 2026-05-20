"""propagul.mesh.cli — Command-line interface for the mesh agent.

Usage:
    propagul-mesh start --name <node-name> [--ollama <url>] [--api-key <key>]
    propagul-mesh detect
    propagul-mesh status
    propagul-mesh version

No external dependencies — uses argparse (stdlib).
"""

import argparse
import asyncio
import json
import logging
import signal
import sys

from propagul.mesh import __version__
from propagul.mesh.agent import MeshAgent
from propagul.mesh.backends.detect import detect as detect_backends
from propagul.mesh.backends.ollama import poll as poll_ollama
from propagul.mesh.gpu import collect as collect_gpu


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_start(args: argparse.Namespace) -> None:
    """Start the mesh agent."""
    _setup_logging(args.verbose)
    logger = logging.getLogger("propagul.mesh.cli")

    logger.info("Propagul Mesh Agent v%s", __version__)
    logger.info("Node: %s", args.name)

    agent = MeshAgent(
        node_id=args.name,
        api_key=args.api_key or "",
        dashboard_url=args.dashboard_url,
        poll_interval=args.interval,
        backend_url=args.ollama,
        proxy_port=args.proxy_port,
        proxy_backend_auth=args.proxy_backend_auth or "",
        advertise_ip=args.advertise_ip or None,
    )

    # Graceful shutdown on SIGINT/SIGTERM
    loop = asyncio.new_event_loop()

    def _shutdown(sig: int, frame) -> None:
        logger.info("Shutting down (signal %d)...", sig)
        loop.call_soon_threadsafe(loop.stop)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        loop.run_until_complete(agent.start())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
        logger.info("Agent stopped.")


def cmd_detect(args: argparse.Namespace) -> None:
    """Auto-detect local inference engines."""
    _setup_logging(args.verbose)

    print("Scanning for local inference engines...\n")
    backends = detect_backends(timeout=3.0)

    if not backends:
        print("  No inference engines detected.")
        print("  Make sure Ollama, vLLM, or llama.cpp is running locally.")
        sys.exit(1)

    for b in backends:
        print(f"  ✅ {b.name} at {b.url} (v{b.version}, confidence: {b.confidence:.0%})")

    print(f"\n  Found {len(backends)} backend(s).")


def cmd_status(args: argparse.Namespace) -> None:
    """Show current status of local backends + GPU."""
    _setup_logging(False)

    # Detect backends
    backends = detect_backends(timeout=3.0)

    print("=" * 60)
    print("  PROPAGUL MESH — Local Status")
    print("=" * 60)

    if not backends:
        print("\n  ❌ No inference engines detected.\n")
    else:
        for b in backends:
            print(f"\n  Backend: {b.name} ({b.url})")
            print(f"  Version: {b.version}")

            if b.name == "ollama":
                status = poll_ollama(base_url=b.url)
                if status.online:
                    print(f"  Status:  ✅ Online")
                    print(f"  Models:  {status.model_count} installed "
                          f"({status.total_model_size_gb} GB total)")
                    print(f"  Running: {status.running_count} loaded in VRAM")

                    if status.models:
                        print("\n  Installed Models:")
                        for m in status.models:
                            print(f"    • {m.name} ({m.size_gb} GB, "
                                  f"{m.parameter_size}, {m.quantization})")

                    if status.running:
                        print("\n  Running Models:")
                        for r in status.running:
                            vram_gb = round(r.vram_bytes / (1024**3), 2)
                            print(f"    • {r.name} ({vram_gb} GB VRAM)")
                else:
                    print(f"  Status:  ❌ Offline ({status.error})")

    # GPU
    gpu = collect_gpu()
    print(f"\n  GPU Backend: {gpu.backend}")
    if gpu.gpus:
        for g in gpu.gpus:
            print(f"    • {g.name}")
            print(f"      VRAM: {g.vram_used_mb}/{g.vram_total_mb} MB "
                  f"({g.vram_utilization_pct}%)")
            print(f"      Utilization: {g.utilization_pct}%")
            if g.temperature_c:
                print(f"      Temperature: {g.temperature_c}°C")
    elif gpu.error:
        print(f"    {gpu.error}")

    print("\n" + "=" * 60)


def cmd_version(args: argparse.Namespace) -> None:
    """Show version."""
    print(f"propagul-mesh {__version__}")


def main() -> None:
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="propagul-mesh",
        description="Local AI Fleet Management — monitor and manage your inference servers",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # start
    p_start = subparsers.add_parser("start", help="Start the mesh agent")
    p_start.add_argument(
        "--name", required=True, help="Node name (e.g. 'workstation-01')"
    )
    p_start.add_argument(
        "--ollama", default=None,
        help="Ollama URL override (default: auto-detect)",
    )
    p_start.add_argument(
        "--api-key", default="",
        help="API key for dashboard (optional — without it, agent runs local-only)",
    )
    p_start.add_argument(
        "--dashboard-url", default="https://cloud.propagul.dev",
        help="Dashboard URL (default: cloud.propagul.dev)",
    )
    p_start.add_argument(
        "--interval", type=int, default=10,
        help="Poll interval in seconds (default: 10)",
    )
    p_start.add_argument(
        "--proxy-port", type=int, default=0,
        help="Start local OpenAI-compatible proxy on this port (default: 0 = disabled, "
             "recommended: 8787). Routes /v1/chat/completions to detected backend.",
    )
    p_start.add_argument(
        "--proxy-backend-auth", default="",
        help="Authorization header for backend requests (e.g. 'Bearer sk-lm-...'). "
             "Used by proxied requests to the backend. Not client pass-through.",
    )
    p_start.add_argument(
        "--advertise-ip", default="",
        help="LAN IP to advertise to other agents for fleet routing. "
             "Default: auto-detected. Use this for multi-NIC hosts, VPN (Tailscale), "
             "or when auto-detection picks the wrong interface.",
    )
    p_start.set_defaults(func=cmd_start)

    # detect
    p_detect = subparsers.add_parser(
        "detect", help="Auto-detect local inference engines"
    )
    p_detect.set_defaults(func=cmd_detect)

    # status
    p_status = subparsers.add_parser(
        "status", help="Show current local status (models, GPU, system)"
    )
    p_status.set_defaults(func=cmd_status)

    # version
    p_version = subparsers.add_parser("version", help="Show version")
    p_version.set_defaults(func=cmd_version)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
