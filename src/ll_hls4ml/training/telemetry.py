"""Low-overhead training-system telemetry."""

from __future__ import annotations

import csv
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover - telemetry remains optional
    psutil = None


GPU_FIELDS = (
    "timestamp",
    "utilization_gpu_percent",
    "utilization_memory_percent",
    "memory_used_mb",
    "power_draw_w",
    "clock_sm_mhz",
    "temperature_c",
    "pstate",
)
_NVIDIA_QUERY = (
    "timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,"
    "clocks.sm,temperature.gpu,pstate"
)
SYSTEM_FIELDS = (
    "monotonic_seconds",
    "system_cpu_percent",
    "training_process_cpu_percent",
    "training_tree_cpu_percent",
    "training_tree_rss_mb",
    "system_disk_read_mb_s",
    "system_disk_write_mb_s",
)


def _number(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


class NvidiaSmiMonitor:
    """Persist a time series from one long-lived ``nvidia-smi`` process."""

    def __init__(self, path: str | Path, interval_ms: int = 500, gpu: int = 0):
        self.path = Path(path)
        self.system_path = self.path.with_name("system_telemetry.csv")
        self.interval_ms = interval_ms
        self.gpu = gpu
        self.process = None
        self.handle = None
        self.system_handle = None
        self.system_thread = None
        self.stop_event = threading.Event()

    def start(self) -> bool:
        executable = shutil.which("nvidia-smi")
        if not executable or self.interval_ms <= 0:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", buffering=1)
        self.handle.write(",".join(GPU_FIELDS) + "\n")
        self.process = subprocess.Popen(
            [
                executable,
                "-i",
                str(self.gpu),
                f"--query-gpu={_NVIDIA_QUERY}",
                "--format=csv,noheader,nounits",
                f"--loop-ms={self.interval_ms}",
            ],
            stdout=self.handle,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._start_system_monitor()
        return True

    def _start_system_monitor(self) -> None:
        if psutil is None:
            return
        self.system_handle = self.system_path.open("w", buffering=1, newline="")
        writer = csv.DictWriter(self.system_handle, fieldnames=SYSTEM_FIELDS)
        writer.writeheader()
        parent = psutil.Process()
        known_processes: dict[int, object] = {parent.pid: parent}
        for process in known_processes.values():
            process.cpu_percent(None)
        psutil.cpu_percent(None)

        def sample() -> None:
            previous_time = time.monotonic()
            previous_disk = psutil.disk_io_counters()
            while not self.stop_event.wait(self.interval_ms / 1000):
                now = time.monotonic()
                elapsed = max(now - previous_time, 1e-12)
                try:
                    current = [parent, *parent.children(recursive=True)]
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    current = [parent]
                for process in current:
                    if process.pid not in known_processes:
                        known_processes[process.pid] = process
                        try:
                            process.cpu_percent(None)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                process_cpu = 0.0
                tree_cpu = 0.0
                rss = 0
                for process in current:
                    try:
                        cpu = process.cpu_percent(None)
                        tree_cpu += cpu
                        if process.pid == parent.pid:
                            process_cpu = cpu
                        rss += process.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                disk = psutil.disk_io_counters()
                read_rate = write_rate = 0.0
                if disk is not None and previous_disk is not None:
                    read_rate = max(disk.read_bytes - previous_disk.read_bytes, 0) / elapsed / 2**20
                    write_rate = max(disk.write_bytes - previous_disk.write_bytes, 0) / elapsed / 2**20
                writer.writerow(
                    {
                        "monotonic_seconds": now,
                        "system_cpu_percent": psutil.cpu_percent(None),
                        "training_process_cpu_percent": process_cpu,
                        "training_tree_cpu_percent": tree_cpu,
                        "training_tree_rss_mb": rss / 2**20,
                        "system_disk_read_mb_s": read_rate,
                        "system_disk_write_mb_s": write_rate,
                    }
                )
                previous_time = now
                previous_disk = disk

        self.stop_event.clear()
        self.system_thread = threading.Thread(target=sample, daemon=True)
        self.system_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.system_thread is not None:
            self.system_thread.join(timeout=5)
            self.system_thread = None
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            self.process = None
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        if self.system_handle is not None:
            self.system_handle.close()
            self.system_handle = None

    def summary(self) -> dict | None:
        if not self.path.exists():
            return None
        with self.path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        utilization = np.asarray(
            [
                value
                for row in rows
                if (value := _number(row["utilization_gpu_percent"])) is not None
            ]
        )
        if not utilization.size:
            return None
        result = {
            "samples": int(utilization.size),
            "interval_ms": self.interval_ms,
            "gpu_utilization_mean_percent": float(utilization.mean()),
            "gpu_utilization_p10_percent": float(np.percentile(utilization, 10)),
            "gpu_utilization_p50_percent": float(np.percentile(utilization, 50)),
            "gpu_utilization_p90_percent": float(np.percentile(utilization, 90)),
            "gpu_zero_utilization_fraction": float((utilization == 0).mean()),
            "gpu_below_25_percent_fraction": float((utilization < 25).mean()),
            "gpu_active_mean_percent": float(
                utilization[utilization >= 25].mean()
            ) if (utilization >= 25).any() else 0.0,
            "gpu_utilization_coefficient_of_variation": float(
                utilization.std() / max(utilization.mean(), 1e-12)
            ),
        }
        low = utilization < 25
        streak = longest = 0
        bursts = 0
        for index, is_low in enumerate(low):
            streak = streak + 1 if is_low else 0
            longest = max(longest, streak)
            if not is_low and (index == 0 or low[index - 1]):
                bursts += 1
        duration_seconds = utilization.size * self.interval_ms / 1000
        result["gpu_longest_below_25_seconds"] = longest * self.interval_ms / 1000
        result["gpu_active_bursts_per_minute"] = bursts * 60 / max(duration_seconds, 1e-12)
        for field, output in (
            ("utilization_memory_percent", "memory_utilization_mean_percent"),
            ("memory_used_mb", "resident_gpu_memory_mean_mb"),
            ("power_draw_w", "power_draw_mean_w"),
            ("clock_sm_mhz", "sm_clock_mean_mhz"),
            ("temperature_c", "temperature_mean_c"),
        ):
            values = np.asarray(
                [
                    value
                    for row in rows
                    if (value := _number(row[field])) is not None
                ]
            )
            result[output] = float(values.mean()) if values.size else None
        states = [row["pstate"].strip() for row in rows if row["pstate"].strip()]
        if states:
            result["predominant_pstate"] = max(set(states), key=states.count)
        if self.system_path.exists():
            with self.system_path.open(newline="") as handle:
                system_rows = list(csv.DictReader(handle))
            for field, output in (
                ("system_cpu_percent", "system_cpu_mean_percent"),
                ("training_process_cpu_percent", "driver_cpu_mean_percent"),
                ("training_tree_cpu_percent", "training_tree_cpu_mean_percent"),
                ("training_tree_rss_mb", "training_tree_rss_mean_mb"),
                ("system_disk_read_mb_s", "system_disk_read_mean_mb_s"),
                ("system_disk_write_mb_s", "system_disk_write_mean_mb_s"),
            ):
                values = np.asarray(
                    [
                        value
                        for row in system_rows
                        if (value := _number(row[field])) is not None
                    ]
                )
                result[output] = float(values.mean()) if values.size else None
                result[output.replace("mean", "p90")] = (
                    float(np.percentile(values, 90)) if values.size else None
                )
        return result

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.stop()
