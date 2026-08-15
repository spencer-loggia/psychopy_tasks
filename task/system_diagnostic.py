#!/usr/bin/env python3
"""Run launcher dependency, hardware, and main-display timing diagnostics."""
from __future__ import annotations

import argparse
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

from bin.glx_timing import query_glx_swap_interval, query_glx_sync_values


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


def prepare_diagnostic_affinity() -> tuple[dict[str, Any], tuple[bool, str]]:
    """Stage GL/window initialization off the presentation core."""
    from bin.affinity import (
        build_main_and_worker_affinity_plan,
        set_process_cpu_affinity,
    )

    plan = build_main_and_worker_affinity_plan(main_core=0)
    if not plan.get("supported"):
        return plan, (False, str(plan.get("reason") or "CPU affinity is unavailable"))
    worker_cores = plan.get("worker_cpu_affinity") or []
    if not worker_cores:
        return plan, (True, str(plan.get("warning") or "no worker cores available"))
    return plan, set_process_cpu_affinity(worker_cores)


def pin_diagnostic_to_cpu_zero(
    affinity_plan: Optional[Mapping[str, Any]] = None,
    *,
    preparation: Optional[tuple[bool, str]] = None,
) -> dict[str, Any]:
    """Pin the timing-diagnostic process to the presentation CPU."""
    try:
        from bin.affinity import (
            build_main_and_worker_affinity_plan,
            set_process_cpu_affinity,
        )

        plan = (
            dict(affinity_plan)
            if affinity_plan is not None
            else build_main_and_worker_affinity_plan(main_core=0)
        )
        if not plan.get("supported"):
            raise RuntimeError(str(plan.get("reason") or "CPU affinity is unavailable"))
        main_cores = plan.get("main_cpu_affinity") or [0]
        applied, detail = set_process_cpu_affinity(main_cores)
        if not applied:
            raise RuntimeError(detail)
        if preparation is not None and not preparation[0]:
            raise RuntimeError(
                f"{detail}; GL/window initialization was not staged off CPU 0: "
                f"{preparation[1]}"
            )
        preparation_detail = (
            f"; before window creation: {preparation[1]}"
            if preparation is not None
            else ""
        )
        return _check(
            "CPU 0 affinity",
            "pass",
            f"Diagnostic frame-timing process pinned successfully: {detail}{preparation_detail}",
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


def _measure_flip_phase(
    win,
) -> tuple[list[float], Optional[dict[str, float | int]], str]:
    before, before_detail = query_glx_sync_values()
    intervals = _measure_flip_intervals(win)
    after, after_detail = query_glx_sync_values()

    if before is None or after is None:
        return intervals, None, before_detail if before is None else after_detail
    delta_ust = int(after["ust"]) - int(before["ust"])
    delta_msc = int(after["msc"]) - int(before["msc"])
    delta_sbc = int(after["sbc"]) - int(before["sbc"])
    progress: dict[str, float | int] = {
        "delta_ust_us": delta_ust,
        "delta_msc": delta_msc,
        "delta_sbc": delta_sbc,
        "submitted_swaps": FLIP_WARMUP_FRAMES + FLIP_TEST_INTERVALS + 1,
    }
    if delta_ust > 0:
        progress["msc_rate_hz"] = delta_msc * 1_000_000.0 / delta_ust
    if delta_sbc > 0:
        progress["msc_per_completed_swap"] = delta_msc / delta_sbc
    return intervals, progress, after_detail


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
        _check("Main display placement", "skip", reason),
        _check("Vsync request", "skip", reason),
        _check("Monitor refresh rate", "skip", reason),
        _check("Flip synchronization", "skip", reason),
    ]


def run_display_diagnostic(
    *,
    cfg: Mapping[str, Any],
    visual_module: object,
    affinity_plan: Optional[Mapping[str, Any]] = None,
    affinity_preparation: Optional[tuple[bool, str]] = None,
) -> tuple[list[dict[str, Any]], Optional[float], Optional[dict[str, Any]]]:
    win = None
    monitor_refresh_rate_hz = None
    monitor_rate_detail = "main output was not resolved"
    timing_refresh_rate_hz = None
    timing_rate_detail = "timing output was not resolved"
    affinity_check = None

    def _with_affinity(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return ([affinity_check] if affinity_check is not None else []) + checks

    try:
        from bin.screen import (
            activate_psychopy_window,
            enforce_window_vsync,
            load_screen_config,
            open_psychopy_window,
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
        win = open_psychopy_window(
            visual_module,
            main_screen,
            fullscreen=True,
            require_correct_placement=False,
            **win_kwargs,
        )
        placement_detail = win._neuro_tasks_screen_placement
        placement_error = getattr(win, "_neuro_tasks_screen_placement_error", None)
        if not isinstance(placement_error, str):
            placement_error = None
        selection_detail = getattr(win, "_neuro_tasks_pyglet_selection", "")
        if not isinstance(selection_detail, str) or not selection_detail:
            selection_detail = "selection details unavailable"
        realized_screen = getattr(win, "_neuro_tasks_realized_screen", None)
        if placement_error and realized_screen is not None:
            timing_refresh_rate_hz, timing_rate_detail = query_main_monitor_refresh_rate(
                realized_screen
            )
        elif placement_error:
            timing_refresh_rate_hz = monitor_refresh_rate_hz
            timing_rate_detail = (
                "fallback main-output reference because the realized output could "
                f"not be identified; {monitor_rate_detail}"
            )
        else:
            timing_refresh_rate_hz = monitor_refresh_rate_hz
            timing_rate_detail = monitor_rate_detail
        window_mode = getattr(
            win,
            "_neuro_tasks_fullscreen_path",
            "native PsychoPy fullscreen",
        )
        if not isinstance(window_mode, str):
            window_mode = "native PsychoPy fullscreen"
        signal_task_window_ready()
        activate_psychopy_window(win)
        if affinity_plan is not None:
            affinity_check = pin_diagnostic_to_cpu_zero(
                affinity_plan,
                preparation=affinity_preparation,
            )
    except Exception as exc:
        error = _error_text(exc)
        if affinity_plan is not None and affinity_check is None:
            affinity_check = _check(
                "CPU 0 affinity",
                "skip",
                "Skipped because no main-display frame checks could run",
            )
        if win is not None:
            try:
                win.close()
            except Exception:
                pass
        if monitor_refresh_rate_hz is not None:
            refresh_check = _check(
                "Monitor refresh rate",
                "pass",
                f"Main monitor refresh rate: {monitor_refresh_rate_hz:.3f} Hz ({monitor_rate_detail})",
            )
        else:
            refresh_check = _check(
                "Monitor refresh rate",
                "fail",
                "The resolved main output has no confirmed xrandr active-mode rate",
                error=monitor_rate_detail,
            )
        checks = [
            _check(
                "Main display placement",
                "fail",
                "PsychoPy could not place a fullscreen window on the resolved main monitor",
                error=error,
            ),
            _check("Vsync request", "skip", "Skipped because the PsychoPy window could not be created"),
            refresh_check,
            _check(
                "Flip synchronization",
                "skip",
                "Skipped because the PsychoPy window could not be created",
            ),
        ]
        return _with_affinity(checks), monitor_refresh_rate_hz, None

    try:
        if placement_error:
            placement_check = _check(
                "Main display placement",
                "fail",
                (
                    f"{window_mode} realized on {placement_detail}; timing checks "
                    f"continue on that realized output; {selection_detail}"
                ),
                error=placement_error,
            )
        else:
            placement_check = _check(
                "Main display placement",
                "pass",
                f"{window_mode} verified on {placement_detail}; {selection_detail}",
            )
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
                    "PsychoPy requested vsync and the driver stored swap interval 1 "
                    f"({swap_interval_detail}); completed-swap timing is checked separately"
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
                error="The diagnostic window did not store the requested swap interval 1",
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
                "PsychoPy did not accept a blocking vsync request for the diagnostic window",
                error="waitBlanking/vsync could not be enabled on the active window backend",
            )

        intervals, sync_progress, sync_progress_detail = _measure_flip_phase(win)
        median_interval_s = statistics.median(intervals) if intervals else None
        interval_rate = (
            1.0 / float(median_interval_s)
            if median_interval_s is not None and median_interval_s > 0.0
            else None
        )
        refresh_rate_hz = monitor_refresh_rate_hz
        if monitor_refresh_rate_hz is None:
            estimate_detail = (
                f" PsychoPy simple flips observed {interval_rate:.3f} Hz."
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
        else:
            observed = (
                f"; PsychoPy simple flips observed {interval_rate:.3f} Hz"
                if interval_rate is not None
                else "; PsychoPy stable-rate measurement unavailable"
            )
            refresh_check = _check(
                "Monitor refresh rate",
                "pass",
                (
                    f"Main monitor refresh rate: {refresh_rate_hz:.3f} Hz "
                    f"({monitor_rate_detail}){observed}"
                ),
            )

        if timing_refresh_rate_hz is None:
            interval_rate_text = (
                f"{interval_rate:.3f} Hz" if interval_rate is not None else "unavailable"
            )
            flip_check = _check(
                "Flip synchronization",
                "fail",
                (
                    f"Recorded {len(intervals)} flip intervals; median interval rate "
                    f"{interval_rate_text}. "
                    "Lock cannot be confirmed without an independent monitor rate"
                ),
                error=timing_rate_detail,
            )
            return _with_affinity(
                [placement_check, vsync_check, refresh_check, flip_check]
            ), refresh_rate_hz, None

        flip_passed, metrics = evaluate_flip_lock(timing_refresh_rate_hz, intervals)
        metrics["monitor_refresh_rate_hz"] = refresh_rate_hz
        metrics["timing_output_refresh_rate_hz"] = timing_refresh_rate_hz
        metrics["timing_output_name"] = str(
            getattr(realized_screen, "name", "unmatched output") or "unmatched output"
        )
        metrics["main_display_placement_verified"] = not bool(placement_error)
        metrics["observed_median_rate_hz"] = interval_rate
        metrics["glx_swap_interval"] = swap_interval
        metrics["glx_sync_progress"] = sync_progress
        metrics["glx_sync_progress_detail"] = sync_progress_detail
        metrics["window_mode"] = window_mode
        locked_percent = float(metrics["locked_fraction"]) * 100.0
        interval_rate_text = (
            f"{interval_rate:.3f} Hz" if interval_rate is not None else "unavailable"
        )
        if not placement_error:
            timing_label = f"main output {getattr(main_screen, 'name', '')}"
        elif realized_screen is not None:
            timing_label = f"WRONG-OUTPUT TIMING ONLY ({placement_detail})"
        else:
            timing_label = f"UNMATCHED-OUTPUT TIMING ONLY ({placement_detail})"
        detail = (
            f"{timing_label}; {window_mode} window; "
            f"target {timing_refresh_rate_hz:.3f} Hz "
            f"({metrics['expected_interval_ms']:.3f} ms); "
            f"simple-flip median {metrics['median_interval_ms']:.3f} ms "
            f"({interval_rate_text}); "
            f"{locked_percent:.1f}% of {metrics['sample_count']} intervals within "
            f"±{metrics['interval_tolerance_ms']:.3f} ms; "
            f"long/dropped intervals {metrics['dropped_interval_count']}"
            f" ({timing_rate_detail})"
        )
        if sync_progress is not None:
            msc_per_swap = sync_progress.get("msc_per_completed_swap")
            msc_rate = sync_progress.get("msc_rate_hz")
            detail += (
                f"; GLX counters ΔMSC={sync_progress['delta_msc']}, "
                f"ΔSBC={sync_progress['delta_sbc']} completed / "
                f"{sync_progress['submitted_swaps']} submitted"
                + (
                    f", {float(msc_per_swap):.3f} GLX retraces/completed swap"
                    if msc_per_swap is not None
                    else ""
                )
                + (
                    f", counter rate {float(msc_rate):.3f} Hz"
                    if msc_rate is not None
                    else ""
                )
            )
        else:
            detail += f"; GLX presentation counters unavailable ({sync_progress_detail})"
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
        return _with_affinity(
            [placement_check, vsync_check, refresh_check, flip_check]
        ), refresh_rate_hz, metrics
    except Exception as exc:
        error = _error_text(exc)
        if monitor_refresh_rate_hz is not None:
            refresh_check = _check(
                "Monitor refresh rate",
                "pass",
                f"Main monitor refresh rate: {monitor_refresh_rate_hz:.3f} Hz ({monitor_rate_detail})",
            )
        else:
            refresh_check = _check(
                "Monitor refresh rate",
                "fail",
                "The resolved main output has no confirmed xrandr active-mode rate",
                error=monitor_rate_detail,
            )
        checks = [
            placement_check,
            _check(
                "Vsync request",
                "fail",
                "The main-display timing diagnostic did not complete",
                error=error,
            ),
            refresh_check,
            _check("Flip synchronization", "skip", "Timing diagnostic did not complete"),
        ]
        return _with_affinity(checks), monitor_refresh_rate_hz, None
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

    refresh_rate_hz = None
    flip_metrics = None
    if visual_module is None:
        checks.extend(
            _skipped_display_checks("Skipped because PsychoPy is unavailable")
        )
    else:
        affinity_plan, affinity_preparation = prepare_diagnostic_affinity()
        display_checks, refresh_rate_hz, flip_metrics = run_display_diagnostic(
            cfg=cfg,
            visual_module=visual_module,
            affinity_plan=affinity_plan,
            affinity_preparation=affinity_preparation,
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
