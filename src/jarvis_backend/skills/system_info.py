"""Навык-пример: сводка о системе (ОС, CPU, память, диск, батарея)."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": "Сводка о компьютере: ОС, процессор, загрузка, свободное место на диске, батарея.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _battery() -> str | None:
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
        except Exception:  # noqa: BLE001
            return None
        lines = [line.strip() for line in out.splitlines() if "%" in line]
        return lines[0] if lines else None
    capacity = Path("/sys/class/power_supply/BAT0/capacity")
    if capacity.exists():
        return capacity.read_text().strip() + "%"
    return None


def get_system_info() -> dict[str, Any]:
    disk = shutil.disk_usage(Path.home())
    info: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "disk_free_gb": round(disk.free / 1e9, 1),
        "disk_total_gb": round(disk.total / 1e9, 1),
    }
    if hasattr(os, "getloadavg"):
        info["load_avg"] = [round(x, 2) for x in os.getloadavg()]
    battery = _battery()
    if battery:
        info["battery"] = battery
    return {"ok": True, "tool": "get_system_info", "result": info}
