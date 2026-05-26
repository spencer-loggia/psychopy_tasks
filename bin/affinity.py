"""
Helpers for process CPU affinity management.

These helpers are intentionally lightweight and degrade gracefully when
`psutil` is unavailable or CPU affinity is not supported on the host OS.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency / platform support
    psutil = None


def describe_cpu_set(cpu_ids: Sequence[int]) -> str:
    normalized = sorted({int(cpu_id) for cpu_id in cpu_ids})
    return ",".join(str(cpu_id) for cpu_id in normalized)


def get_process_cpu_affinity(pid: Optional[int] = None) -> Optional[list[int]]:
    if psutil is None:
        return None
    try:
        proc = psutil.Process() if pid is None else psutil.Process(int(pid))
        return sorted(int(cpu_id) for cpu_id in proc.cpu_affinity())
    except Exception:
        return None


def set_process_cpu_affinity(
    cpu_ids: Sequence[int],
    *,
    pid: Optional[int] = None,
) -> tuple[bool, str]:
    if psutil is None:
        return False, "psutil is not available"

    normalized = sorted({int(cpu_id) for cpu_id in cpu_ids})
    if not normalized:
        return False, "no CPU cores were provided"

    try:
        proc = psutil.Process() if pid is None else psutil.Process(int(pid))
        proc.cpu_affinity(normalized)
        actual = sorted(int(cpu_id) for cpu_id in proc.cpu_affinity())
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    if actual != normalized:
        return False, f"requested [{describe_cpu_set(normalized)}] but kernel applied [{describe_cpu_set(actual)}]"

    target_desc = "current process" if pid is None else f"pid={int(pid)}"
    return True, f"{target_desc} cpu_affinity=[{describe_cpu_set(actual)}]"


def build_main_and_worker_affinity_plan(main_core: int = 0) -> Dict[str, Any]:
    current_affinity = get_process_cpu_affinity()
    if current_affinity is None:
        return {
            "supported": False,
            "reason": "CPU affinity is unavailable on this host or psutil is not installed",
            "current_affinity": None,
            "main_cpu_affinity": None,
            "worker_cpu_affinity": None,
        }

    main_core = int(main_core)
    if main_core not in current_affinity:
        return {
            "supported": False,
            "reason": (
                f"requested main core {main_core} is not present in the current process affinity mask "
                f"[{describe_cpu_set(current_affinity)}]"
            ),
            "current_affinity": current_affinity,
            "main_cpu_affinity": None,
            "worker_cpu_affinity": None,
        }

    worker_cpu_affinity = [cpu_id for cpu_id in current_affinity if cpu_id != main_core]
    warning = None
    if not worker_cpu_affinity:
        warning = (
            f"no worker-only CPU cores are available beyond main core {main_core}; "
            "non-critical child processes cannot be kept off the main timing core"
        )

    return {
        "supported": True,
        "reason": None,
        "warning": warning,
        "current_affinity": current_affinity,
        "main_cpu_affinity": [main_core],
        "worker_cpu_affinity": worker_cpu_affinity,
    }
