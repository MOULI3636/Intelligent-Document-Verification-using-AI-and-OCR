"""
System Resource Profiler and Inference Latency Monitor for DocVision AI.

Measures RAM memory usage, PyTorch CUDA GPU VRAM peak allocation,
execution latency percentiles (mean, p50, p95, p99), and throughput (FPS).
"""

from dataclasses import dataclass, asdict
import os
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch

from src.utils.logger import get_logger

logger = get_logger("SystemResourceMonitor")


@dataclass
class HardwareProfile:
    """
    Data payload holding hardware memory allocation and latency profiling metrics.
    """
    ram_used_mb: float
    gpu_name: str
    gpu_allocated_mb: float
    gpu_peak_allocated_mb: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_fps: float


class SystemResourceMonitor:
    """
    Monitors system memory (RAM & CUDA VRAM) and execution throughput statistics.
    """

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_ram_usage_mb(self) -> float:
        """Returns current process RAM memory usage in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return float(process.memory_info().rss / (1024 * 1024))
        except Exception:
            return 0.0

    def get_gpu_memory_mb(self) -> Tuple[float, float, str]:
        """
        Returns PyTorch GPU VRAM statistics.

        Returns:
            Tuple[float, float, str]: (Allocated MB, Peak Allocated MB, GPU Device Name).
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            device_name = torch.cuda.get_device_name(0)
            return float(allocated), float(peak), device_name
        return 0.0, 0.0, "CPU (No CUDA)"

    def get_system_telemetry(self) -> Dict[str, Any]:
        """Returns consolidated live hardware and memory telemetry."""
        ram_mb = self.get_ram_usage_mb()
        vram_alloc, vram_peak, device_name = self.get_gpu_memory_mb()
        ram_percent = 0.0
        try:
            import psutil
            ram_percent = psutil.virtual_memory().percent
        except Exception:
            pass

        return {
            "process_ram_mb": ram_mb,
            "ram_percent": ram_percent,
            "vram_allocated_mb": vram_alloc,
            "vram_peak_mb": vram_peak,
            "device_name": device_name
        }

    def profile_latencies(self, latencies_ms: List[float]) -> Dict[str, float]:
        """
        Calculates mean, p50, p95, p99 latencies and throughput FPS.

        Args:
            latencies_ms (List[float]): List of per-sample execution times in milliseconds.

        Returns:
            Dict[str, float]: Latency percentiles dictionary.
        """
        if not latencies_ms:
            return {
                "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "fps": 0.0
            }

        arr = np.array(latencies_ms)
        mean_lat = float(np.mean(arr))
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

        return {
            "mean_ms": mean_lat,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "fps": fps
        }

    def capture_profile(self, latencies_ms: List[float]) -> HardwareProfile:
        """Captures complete system resource and performance profile."""
        ram_mb = self.get_ram_usage_mb()
        gpu_alloc, gpu_peak, gpu_name = self.get_gpu_memory_mb()
        lat_stats = self.profile_latencies(latencies_ms)

        profile = HardwareProfile(
            ram_used_mb=ram_mb,
            gpu_name=gpu_name,
            gpu_allocated_mb=gpu_alloc,
            gpu_peak_allocated_mb=gpu_peak,
            mean_latency_ms=lat_stats["mean_ms"],
            p50_latency_ms=lat_stats["p50_ms"],
            p95_latency_ms=lat_stats["p95_ms"],
            p99_latency_ms=lat_stats["p99_ms"],
            throughput_fps=lat_stats["fps"]
        )

        logger.info(f"Resource Profile Captured -> RAM: {ram_mb:.1f}MB | VRAM Peak: {gpu_peak:.1f}MB | Throughput: {profile.throughput_fps:.1f} FPS")
        return profile
