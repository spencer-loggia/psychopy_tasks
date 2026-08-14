#!/usr/bin/env python3
"""Run launcher dependency, hardware, and main-display timing diagnostics."""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


DEFAULT_DAQ_ADDRESS = 0
FLIP_TEST_INTERVALS = 120
FLIP_WARMUP_FRAMES = 20
MIN_LOCKED_FRACTION = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check neuro_tasks system readiness")
    parser.add_argument("--config", required=True, help="Launcher configuration JSON")
    parser.add_argument("--result", required=True, help="Path for the diagnostic JSON result")
    return parser.parse_args()


def _check(
    name: str,
    status: str,
    detail: str,
    *,
    error: Optional[str] = None,
) -> dict[str, Any]:
    result = {"name": name, "status": status, "detail": detail}
    if error:
        result["error"] = error
    return result


def _error_text(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def probe_psychopy() -> tuple[dict[str, Any], Optional[object]]:
    try:
        psychopy = importlib.import_module("psychopy")
        visual = importlib.import_module("psychopy.visual")
        version = str(getattr(psychopy, "__version__", "unknown"))
        return _check("PsychoPy", "pass", f"PsychoPy {version} is importable"), visual
    except Exception as exc:
        error = _error_text(exc)
        return (
            _check(
                "PsychoPy",
                "fail",
                "PsychoPy could not be imported by the configured task interpreter",
                error=error,
            ),
            None,
        )


def probe_gpio() -> dict[str, Any]:
    chip = None
    lgpio = None
    try:
        lgpio = importlib.import_module("lgpio")
        chip = lgpio.gpiochip_open(0)
        if isinstance(chip, int) and chip < 0:
            raise RuntimeError(f"gpiochip_open(0) returned error code {chip}")
        return _check(
            "Raspberry Pi GPIO",
            "pass",
            "lgpio imported and GPIO chip 0 opened successfully",
        )
    except Exception as exc:
        error = _error_text(exc)
        return _check(
            "Raspberry Pi GPIO",
            "fail",
            "GPIO chip 0 is not available through lgpio",
            error=error,
        )
    finally:
        if lgpio is not None and chip is not None and not (
            isinstance(chip, int) and chip < 0
        ):
            try:
                lgpio.gpiochip_close(chip)
            except Exception:
                pass


def probe_piplate(
    *,
    address: int = DEFAULT_DAQ_ADDRESS,
    module_name: str = "piplates.DAQC2plate",
) -> dict[str, Any]:
    try:
        address = int(address)
        if address < 0 or address > 7:
            raise ValueError("DAQC2 address must be in the range 0 through 7")
        module = importlib.import_module(str(module_name))
        get_adc = getattr(module, "getADC", None)
        if not callable(get_adc):
            raise RuntimeError(f"{module_name} does not expose getADC(address, channel)")
        supply_voltage = float(get_adc(address, 8))
        if not math.isfinite(supply_voltage) or supply_voltage <= 0.0:
            raise RuntimeError(
                f"DAQC2 channel 8 returned invalid supply voltage {supply_voltage!r}"
            )
        return _check(
            "Pi-Plate DAQC2",
            "pass",
            (
                f"{module_name} communicated with address {address}; "
                f"supply readback is {supply_voltage:.3f} V"
            ),
        )
    except Exception as exc:
        error = _error_text(exc)
        return _check(
            "Pi-Plate DAQC2",
            "fail",
            f"DAQC2 plate address {address} is not available through {module_name}",
            error=error,
        )


def pin_diagnostic_to_cpu_zero() -> dict[str, Any]:
    """Pin the timing-diagnostic process to the presentation CPU."""
    try:
        from bin.affinity import (
            build_main_and_worker_affinity_plan,
            set_process_cpu_affinity,
        )

        plan = build_main_and_worker_affinity_plan(main_core=0)
        if not plan.get("supported"):
            raise RuntimeError(str(plan.get("reason") or "CPU affinity is unavailable"))
        applied, detail = set_process_cpu_affinity([0])
        if not applied:
            raise RuntimeError(detail)
        return _check(
            "CPU 0 affinity",
            "pass",
            f"Diagnostic frame-timing process pinned successfully: {detail}",
        )
    except Exception as exc:
        error = _error_text(exc)
        return _check(
            "CPU 0 affinity",
            "fail",
            "The diagnostic process could not be pinned to CPU 0",
            error=error,
        )


def evaluate_flip_lock(
    refresh_rate_hz: float,
    intervals_s: Iterable[float],
) -> tuple[bool, dict[str, float | int | bool]]:
    """Evaluate whether consecutive flips occur once per measured refresh."""
    refresh_rate_hz = float(refresh_rate_hz)
    if not math.isfinite(refresh_rate_hz) or refresh_rate_hz <= 0.0:
        raise ValueError("refresh_rate_hz must be a positive finite value")
    intervals = [
        float(value)
        for value in intervals_s
        if math.isfinite(float(value)) and float(value) > 0.0
    ]
    if not intervals:
        raise ValueError("No valid flip intervals were recorded")

    expected_s = 1.0 / refresh_rate_hz
    interval_tolerance_s = max(0.0015, expected_s * 0.12)
    median_tolerance_s = max(0.00075, expected_s * 0.05)
    median_s = float(statistics.median(intervals))
    locked_count = sum(
        1 for value in intervals if abs(value - expected_s) <= interval_tolerance_s
    )
    locked_fraction = locked_count / len(intervals)
    dropped_count = sum(1 for value in intervals if value > expected_s * 1.5)
    median_error_s = abs(median_s - expected_s)
    median_matches = median_error_s <= median_tolerance_s
    passed = bool(median_matches and locked_fraction >= MIN_LOCKED_FRACTION)
    return passed, {
        "sample_count": len(intervals),
        "expected_interval_ms": expected_s * 1000.0,
        "median_interval_ms": median_s * 1000.0,
        "interval_tolerance_ms": interval_tolerance_s * 1000.0,
        "median_tolerance_ms": median_tolerance_s * 1000.0,
        "median_error_ms": median_error_s * 1000.0,
        "median_matches": median_matches,
        "locked_fraction": locked_fraction,
        "dropped_interval_count": dropped_count,
    }


def parse_xrandr_active_refresh_rate(
    output: str,
    output_name: str,
) -> Optional[float]:
    """Return the starred active-mode refresh for one xrandr output."""
    in_requested_output = False
    for raw_line in str(output).splitlines():
        if raw_line and not raw_line[0].isspace():
            connected_match = re.match(r"^(\S+)\s+(connected|disconnected)\b", raw_line)
            in_requested_output = bool(
                connected_match
                and connected_match.group(1) == output_name
                and connected_match.group(2) == "connected"
            )
            continue
        if not in_requested_output:
            continue
        active_match = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\*", raw_line)
        if active_match is None:
            continue
        refresh_rate_hz = float(active_match.group(1))
        if math.isfinite(refresh_rate_hz) and refresh_rate_hz > 0.0:
            return refresh_rate_hz
    return None


def query_main_monitor_refresh_rate(
    screen_info: object,
) -> tuple[Optional[float], str]:
    """Read the main output's active hardware mode without timing PsychoPy flips."""
    output_name = str(getattr(screen_info, "name", "") or "").strip()
    if not output_name:
        return None, "resolved main monitor has no output name"
    try:
        completed = subprocess.run(
            ["xrandr", "--current"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:
        return None, f"xrandr query failed: {_error_text(exc)}"
    refresh_rate_hz = parse_xrandr_active_refresh_rate(
        completed.stdout,
        output_name,
    )
    if refresh_rate_hz is None:
        return None, f"xrandr did not report an active mode for output {output_name}"
    geometry = (
        f"{getattr(screen_info, 'width', '?')}x{getattr(screen_info, 'height', '?')}"
        f"+{getattr(screen_info, 'x', '?')}+{getattr(screen_info, 'y', '?')}"
    )
    return (
        refresh_rate_hz,
        f"xrandr active mode for resolved main output {output_name} ({geometry})",
    )


def query_glx_swap_interval() -> tuple[Optional[int], str]:
    """Read the swap interval acknowledged for the current X11 drawable."""
    if not sys.platform.startswith("linux"):
        return None, "native GLX swap-interval queries are available only on Linux/X11"
    try:
        libgl = ctypes.CDLL(ctypes.util.find_library("GL") or "libGL.so.1")
        libx11 = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")

        libgl.glXGetCurrentDisplay.restype = ctypes.c_void_p
        libgl.glXGetCurrentDrawable.restype = ctypes.c_ulong
        display = libgl.glXGetCurrentDisplay()
        drawable = libgl.glXGetCurrentDrawable()
        if not display or not drawable:
            return None, "the PsychoPy window has no current GLX display/drawable"

        libx11.XDefaultScreen.argtypes = [ctypes.c_void_p]
        libx11.XDefaultScreen.restype = ctypes.c_int
        libgl.glXQueryExtensionsString.argtypes = [ctypes.c_void_p, ctypes.c_int]
        libgl.glXQueryExtensionsString.restype = ctypes.c_char_p
        raw_extensions = libgl.glXQueryExtensionsString(
            display,
            libx11.XDefaultScreen(display),
        )
        extensions = set((raw_extensions or b"").decode("ascii", "replace").split())

        if "GLX_EXT_swap_control" in extensions:
            value = ctypes.c_uint()
            libgl.glXQueryDrawable.argtypes = [
                ctypes.c_void_p,
                ctypes.c_ulong,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint),
            ]
            libgl.glXQueryDrawable(
                display,
                drawable,
                0x20F1,  # GLX_SWAP_INTERVAL_EXT
                ctypes.byref(value),
            )
            return int(value.value), "GLX_EXT_swap_control"

        if "GLX_MESA_swap_control" in extensions:
            libgl.glXGetProcAddressARB.argtypes = [ctypes.c_char_p]
            libgl.glXGetProcAddressARB.restype = ctypes.c_void_p
            address = libgl.glXGetProcAddressARB(b"glXGetSwapIntervalMESA")
            if address:
                get_interval = ctypes.CFUNCTYPE(ctypes.c_int)(address)
                return int(get_interval()), "GLX_MESA_swap_control"
            return None, "GLX_MESA_swap_control is advertised but not callable"

        return None, "the current GLX driver exposes no queryable swap-control extension"
    except Exception as exc:
        return None, f"GLX swap-interval query failed: {_error_text(exc)}"


def _measure_flip_intervals(win, *, interval_count: int = FLIP_TEST_INTERVALS) -> list[float]:
    for index in range(FLIP_WARMUP_FRAMES):
        win.color = (-0.02, -0.02, -0.02) if index % 2 else (0.02, 0.02, 0.02)
        win.flip()

    timestamps = []
    for index in range(int(interval_count) + 1):
        win.color = (-0.02, -0.02, -0.02) if index % 2 else (0.02, 0.02, 0.02)
        win.flip()
        timestamps.append(time.perf_counter())
    return [later - earlier for earlier, later in zip(timestamps, timestamps[1:])]


def _diagnostic_settings(cfg: Mapping[str, Any]) -> tuple[int, str]:
    diagnostic_cfg = cfg.get("diagnostic", {})
    if diagnostic_cfg is None:
        diagnostic_cfg = {}
    if not isinstance(diagnostic_cfg, dict):
        raise ValueError("Launcher config field 'diagnostic' must be a JSON object")
    return (
        int(diagnostic_cfg.get("daq_address", DEFAULT_DAQ_ADDRESS)),
        str(diagnostic_cfg.get("daq_module", "piplates.DAQC2plate")),
    )


def _skipped_display_checks(reason: str) -> list[dict[str, Any]]:
    return [
        _check("Vsync request", "skip", reason),
        _check("Monitor refresh rate", "skip", reason),
        _check("Flip synchronization", "skip", reason),
    ]


def run_display_diagnostic(
    *,
    cfg: Mapping[str, Any],
    visual_module: object,
) -> tuple[list[dict[str, Any]], Optional[float], Optional[dict[str, Any]]]:
    win = None
    try:
        from bin.screen import (
            enforce_window_vsync,
            get_psychopy_window_kwargs,
            load_screen_config,
            resolve_task_screens,
        )
        from bin.task_lifecycle import signal_task_window_ready

        screen_cfg = load_screen_config(dict(cfg))
        main_screen, _experimenter_screen = resolve_task_screens(
            screen_cfg,
            allow_same_screen=True,
        )
        monitor_refresh_rate_hz, monitor_rate_detail = query_main_monitor_refresh_rate(
            main_screen
        )
        win_kwargs = {
            "color": (0.0, 0.0, 0.0),
            "colorSpace": "rgb",
            "units": "pix",
            "allowGUI": False,
            "allowStencil": False,
            "waitBlanking": True,
        }
        win_kwargs.update(
            get_psychopy_window_kwargs(main_screen, fullscreen=True)
        )
        window_mode = "fullscreen" if win_kwargs.get("fullscr") else "borderless"
        win = visual_module.Window(**win_kwargs)
        signal_task_window_ready()
    except Exception as exc:
        error = _error_text(exc)
        if win is not None:
            try:
                win.close()
            except Exception:
                pass
        checks = [
            _check(
                "Vsync request",
                "fail",
                "PsychoPy could not open a diagnostic window on the main monitor",
                error=error,
            ),
            _check(
                "Monitor refresh rate",
                "skip",
                "Skipped because the main-monitor window could not be opened",
            ),
            _check(
                "Flip synchronization",
                "skip",
                "Skipped because the main-monitor window could not be opened",
            ),
        ]
        return checks, None, None

    try:
        vsync_requested = bool(enforce_window_vsync(win)) and bool(
            getattr(win, "waitBlanking", False)
        )
        swap_interval = None
        swap_interval_detail = "vsync was not requested"
        if vsync_requested:
            # EXT swap-control changes take effect after the next buffer swap.
            win.flip()
            swap_interval, swap_interval_detail = query_glx_swap_interval()

        if swap_interval == 1:
            vsync_check = _check(
                "Vsync request",
                "pass",
                (
                    "PsychoPy requested blocking vsync and the driver acknowledged "
                    f"swap interval 1 ({swap_interval_detail})"
                ),
            )
        elif vsync_requested and swap_interval is not None:
            vsync_check = _check(
                "Vsync request",
                "fail",
                (
                    "PsychoPy requested blocking vsync, but the driver reports "
                    f"swap interval {swap_interval} ({swap_interval_detail})"
                ),
                error="The main window was not acknowledged for one swap per refresh",
            )
        elif vsync_requested:
            vsync_check = _check(
                "Vsync request",
                "fail",
                "PsychoPy requested blocking vsync, but driver acknowledgment is unavailable",
                error=swap_interval_detail,
            )
        else:
            vsync_check = _check(
                "Vsync request",
                "fail",
                "PsychoPy did not accept a blocking vsync request for the main monitor",
                error="waitBlanking/vsync could not be enabled on the active window backend",
            )

        psychopy_rate = None
        try:
            measured = win.getActualFrameRate(
                nIdentical=20,
                nMaxFrames=180,
                nWarmUpFrames=20,
                threshold=1,
            )
            if measured is not None:
                measured = float(measured)
                if math.isfinite(measured) and measured > 0.0:
                    psychopy_rate = measured
        except Exception:
            pass

        intervals = _measure_flip_intervals(win)
        median_interval_s = statistics.median(intervals) if intervals else None
        interval_rate = (
            1.0 / float(median_interval_s)
            if median_interval_s is not None and median_interval_s > 0.0
            else None
        )
        if monitor_refresh_rate_hz is None:
            refresh_rate_hz = None
            estimate_detail = (
                f" PsychoPy observed {psychopy_rate:.3f} Hz;"
                if psychopy_rate is not None
                else ""
            ) + (
                f" the diagnostic intervals suggest {interval_rate:.3f} Hz."
                if interval_rate is not None else ""
            )
            refresh_check = _check(
                "Monitor refresh rate",
                "fail",
                (
                    "The resolved main output has no confirmed xrandr active-mode rate."
                    f"{estimate_detail} Neither observed rate is used as the hardware rate."
                ),
                error=monitor_rate_detail,
            )
            flip_check = _check(
                "Flip synchronization",
                "skip",
                (
                    "Flip intervals were recorded, but lock cannot be confirmed without "
                    "an independently measured main-monitor refresh rate"
                ),
            )
            return [vsync_check, refresh_check, flip_check], refresh_rate_hz, None

        refresh_rate_hz = monitor_refresh_rate_hz
        source = monitor_rate_detail
        observed = (
            f"; PsychoPy observed {psychopy_rate:.3f} Hz"
            if psychopy_rate is not None
            else "; PsychoPy stable-rate measurement unavailable"
        )
        refresh_check = _check(
            "Monitor refresh rate",
            "pass",
            f"Main monitor refresh rate: {refresh_rate_hz:.3f} Hz ({source}){observed}",
        )

        flip_passed, metrics = evaluate_flip_lock(refresh_rate_hz, intervals)
        metrics["monitor_refresh_rate_hz"] = refresh_rate_hz
        metrics["psychopy_measured_rate_hz"] = psychopy_rate
        metrics["observed_median_rate_hz"] = interval_rate
        metrics["glx_swap_interval"] = swap_interval
        metrics["window_mode"] = window_mode
        locked_percent = float(metrics["locked_fraction"]) * 100.0
        psychopy_rate_text = (
            f"{psychopy_rate:.3f} Hz" if psychopy_rate is not None else "unavailable"
        )
        interval_rate_text = (
            f"{interval_rate:.3f} Hz" if interval_rate is not None else "unavailable"
        )
        detail = (
            f"{window_mode} window; target {refresh_rate_hz:.3f} Hz "
            f"({metrics['expected_interval_ms']:.3f} ms); "
            f"PsychoPy measured {psychopy_rate_text}; median flip interval "
            f"{metrics['median_interval_ms']:.3f} ms ({interval_rate_text}); "
            f"{locked_percent:.1f}% of {metrics['sample_count']} intervals within "
            f"±{metrics['interval_tolerance_ms']:.3f} ms; "
            f"long/dropped intervals {metrics['dropped_interval_count']}"
        )
        if flip_passed:
            flip_check = _check("Flip synchronization", "pass", detail)
        else:
            failure_reasons = []
            if locked_percent < MIN_LOCKED_FRACTION * 100.0:
                failure_reasons.append(
                    f"only {locked_percent:.1f}% of intervals matched; at least "
                    f"{MIN_LOCKED_FRACTION * 100.0:.1f}% is required"
                )
            if not bool(metrics["median_matches"]):
                failure_reasons.append(
                    f"median interval error {metrics['median_error_ms']:.3f} ms exceeded "
                    f"the {metrics['median_tolerance_ms']:.3f} ms limit"
                )
            flip_check = _check(
                "Flip synchronization",
                "fail",
                detail,
                error="; ".join(failure_reasons),
            )
        return [vsync_check, refresh_check, flip_check], refresh_rate_hz, metrics
    except Exception as exc:
        error = _error_text(exc)
        checks = [
            _check(
                "Vsync request",
                "fail",
                "The main-display timing diagnostic did not complete",
                error=error,
            ),
            _check("Monitor refresh rate", "skip", "Timing diagnostic did not complete"),
            _check("Flip synchronization", "skip", "Timing diagnostic did not complete"),
        ]
        return checks, None, None
    finally:
        if win is not None:
            try:
                win.close()
            except Exception:
                pass


def run_system_diagnostic(cfg: Mapping[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    psychopy_check, visual_module = probe_psychopy()
    checks.append(psychopy_check)
    checks.append(probe_gpio())

    try:
        daq_address, daq_module = _diagnostic_settings(cfg)
        checks.append(probe_piplate(address=daq_address, module_name=daq_module))
    except Exception as exc:
        checks.append(
            _check(
                "Pi-Plate DAQC2",
                "fail",
                "The launcher diagnostic DAQC2 settings are invalid",
                error=_error_text(exc),
            )
        )

    # Match active_foraging's timing-critical placement before opening the
    # PsychoPy window or measuring any main-display flips.
    checks.append(pin_diagnostic_to_cpu_zero())

    refresh_rate_hz = None
    flip_metrics = None
    if visual_module is None:
        checks.extend(
            _skipped_display_checks("Skipped because PsychoPy is unavailable")
        )
    else:
        display_checks, refresh_rate_hz, flip_metrics = run_display_diagnostic(
            cfg=cfg,
            visual_module=visual_module,
        )
        checks.extend(display_checks)

    return {
        "success": all(check_result["status"] == "pass" for check_result in checks),
        "refresh_rate_hz": refresh_rate_hz,
        "checks": checks,
        "flip_metrics": flip_metrics,
    }


def write_result(path: Path, result: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(dict(result), indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


def main() -> int:
    args = parse_args()
    result_path = Path(args.result).expanduser().resolve()
    try:
        config_path = Path(args.config).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            cfg = json.load(handle)
        if not isinstance(cfg, dict):
            raise ValueError("Launcher config must contain a top-level JSON object")
        result = run_system_diagnostic(cfg)
    except BaseException as exc:
        error = _error_text(exc)
        result = {
            "success": False,
            "refresh_rate_hz": None,
            "checks": [
                _check(
                    "Diagnostic runner",
                    "fail",
                    "The system diagnostic could not complete",
                    error=error,
                )
            ],
            "flip_metrics": None,
        }
    try:
        write_result(result_path, result)
    except Exception as exc:
        print(f"Could not write diagnostic result: {_error_text(exc)}", file=sys.stderr)
        return 2
    return 0 if bool(result.get("success")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
