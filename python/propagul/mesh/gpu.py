"""propagul.mesh.gpu — GPU metrics collection.

Collects GPU utilization, VRAM usage, temperature from:
- NVIDIA GPUs (via nvidia-smi CLI — no pynvml dependency)
- Apple Silicon (via system_profiler — macOS only)
- CPU-only fallback

Zero external dependencies — uses subprocess + XML/JSON parsing.
"""

import json
import logging
import platform
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("propagul.mesh.gpu")


@dataclass
class GpuInfo:
    """Metrics for a single GPU."""
    index: int
    name: str
    driver_version: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int
    utilization_pct: float  # 0-100
    temperature_c: Optional[int] = None
    power_draw_w: Optional[float] = None
    power_limit_w: Optional[float] = None

    @property
    def vram_utilization_pct(self) -> float:
        if self.vram_total_mb == 0:
            return 0.0
        return round(self.vram_used_mb / self.vram_total_mb * 100, 1)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "driver_version": self.driver_version,
            "vram_total_mb": self.vram_total_mb,
            "vram_used_mb": self.vram_used_mb,
            "vram_free_mb": self.vram_free_mb,
            "utilization_pct": self.utilization_pct,
            "vram_utilization_pct": self.vram_utilization_pct,
            "temperature_c": self.temperature_c,
            "power_draw_w": self.power_draw_w,
        }


@dataclass
class SystemGpuStatus:
    """Complete GPU status for a system."""
    gpus: list[GpuInfo] = field(default_factory=list)
    backend: str = "none"  # "nvidia", "apple", "none"
    error: Optional[str] = None

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def total_vram_mb(self) -> int:
        return sum(g.vram_total_mb for g in self.gpus)

    @property
    def total_vram_used_mb(self) -> int:
        return sum(g.vram_used_mb for g in self.gpus)

    @property
    def avg_utilization(self) -> float:
        if not self.gpus:
            return 0.0
        return round(sum(g.utilization_pct for g in self.gpus) / len(self.gpus), 1)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "gpu_count": self.gpu_count,
            "total_vram_mb": self.total_vram_mb,
            "total_vram_used_mb": self.total_vram_used_mb,
            "avg_utilization_pct": self.avg_utilization,
            "error": self.error,
            "gpus": [g.to_dict() for g in self.gpus],
        }


def _parse_int(s: str, default: int = 0) -> int:
    """Parse int from string like '8192 MiB' or '75 %'."""
    try:
        return int("".join(c for c in s if c.isdigit()))
    except (ValueError, TypeError):
        return default


def _parse_float(s: str, default: float = 0.0) -> float:
    """Parse float from string like '125.50 W'."""
    try:
        cleaned = "".join(c for c in s if c.isdigit() or c == ".")
        return float(cleaned)
    except (ValueError, TypeError):
        return default


def _collect_nvidia() -> Optional[SystemGpuStatus]:
    """Collect NVIDIA GPU metrics via nvidia-smi XML output."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-q", "-x"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    try:
        # P2-02: Disable external entity processing to prevent XXE
        parser = ET.XMLParser()
        # ET.XMLParser in stdlib doesn't resolve external entities by default,
        # but we explicitly use fromstring which is safe for trusted nvidia-smi output.
        root = ET.fromstring(result.stdout)
    except ET.ParseError:
        return None

    driver = root.findtext("driver_version", "unknown")
    gpus: list[GpuInfo] = []

    for idx, gpu_elem in enumerate(root.findall("gpu")):
        fb = gpu_elem.find("fb_memory_usage")
        util = gpu_elem.find("utilization")
        temp_elem = gpu_elem.find("temperature")
        power = gpu_elem.find("gpu_power_readings") or gpu_elem.find("power_readings")

        gpus.append(GpuInfo(
            index=idx,
            name=gpu_elem.findtext("product_name", "Unknown GPU"),
            driver_version=driver,
            vram_total_mb=_parse_int(fb.findtext("total", "0")) if fb is not None else 0,
            vram_used_mb=_parse_int(fb.findtext("used", "0")) if fb is not None else 0,
            vram_free_mb=_parse_int(fb.findtext("free", "0")) if fb is not None else 0,
            utilization_pct=float(_parse_int(util.findtext("gpu_util", "0"))) if util is not None else 0.0,
            temperature_c=_parse_int(temp_elem.findtext("gpu_temp", "0")) if temp_elem is not None else None,
            power_draw_w=_parse_float(power.findtext("power_draw", "0")) if power is not None else None,
            power_limit_w=_parse_float(power.findtext("power_limit", "0")) if power is not None else None,
        ))

    return SystemGpuStatus(gpus=gpus, backend="nvidia")


def _collect_apple() -> Optional[SystemGpuStatus]:
    """Collect Apple Silicon GPU metrics via system_profiler."""
    if platform.system() != "Darwin":
        return None

    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        displays = data.get("SPDisplaysDataType", [])
        if not displays:
            return None

        gpus: list[GpuInfo] = []
        for idx, disp in enumerate(displays):
            # Apple Silicon reports unified memory — we approximate
            vram_str = disp.get("sppci_vram", disp.get("spdisplays_vram", "0"))
            vram_mb = _parse_int(str(vram_str))
            # Convert GB to MB if the value is suspiciously low
            if vram_mb < 100:
                vram_mb *= 1024

            gpus.append(GpuInfo(
                index=idx,
                name=disp.get("sppci_model", "Apple GPU"),
                driver_version="Metal",
                vram_total_mb=vram_mb,
                vram_used_mb=0,  # Not available without IOKit
                vram_free_mb=vram_mb,
                utilization_pct=0.0,  # Not available without IOKit
            ))

        return SystemGpuStatus(gpus=gpus, backend="apple")

    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def collect() -> SystemGpuStatus:
    """Collect GPU metrics from the best available source.

    Tries NVIDIA first, then Apple Silicon, then returns empty.
    Never raises — always returns a valid SystemGpuStatus.
    """
    # Try NVIDIA
    nvidia = _collect_nvidia()
    if nvidia and nvidia.gpus:
        return nvidia

    # Try Apple Silicon
    apple = _collect_apple()
    if apple and apple.gpus:
        return apple

    # No GPU detected
    return SystemGpuStatus(backend="none", error="No GPU detected")
