"""Display-independent presentation and outcome timing for active foraging."""

import math
import time

from bin.frame_timing import validate_requested_durations, wait_until


def reward_interval_duration(
    *, pump_delay_time, pump_pulse_time_seconds, inter_pump_interval,
    reward_to_pulse_map,
):
    """Reserve the longest configured reward train, even when it is not sampled."""
    validate_requested_durations({
        "pump_delay_time": pump_delay_time,
        "pump_pulse_time_seconds": pump_pulse_time_seconds,
        "inter_pump_interval": inter_pump_interval,
    }, context="active_foraging")
    counts = list((reward_to_pulse_map or {}).values())
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts):
        raise ValueError("reward_to_pulse_map values must be non-negative integers")
    maximum = max(counts, default=0)
    if maximum == 0:
        return 0.0
    duration = (float(pump_delay_time) + maximum * float(pump_pulse_time_seconds)
                + (maximum - 1) * float(inter_pump_interval))
    if not math.isfinite(duration):
        raise ValueError("reward interval must be finite")
    return duration


def run_outcome_interval(
    *, trial_num, onset_perf_ns, num_pulses, apply_timeout,
    reward_interval_s, timeout_duration_s, pump_delay_s, pump_pulse_s,
    inter_pump_interval_s, logger, set_pump, set_buzzer, poll_callback=None,
):
    """Run reward OR timeout, returning the actual ITI start, or None on abort.

    All non-timeout outcomes share one deadline anchored to the choice-clear
    flip. Pulse widths/gaps are anchored to hardware edges, so event logging
    does not add a fresh delay. Hardware overruns never truncate a pump pulse.
    The caller must still run a complete ITI after this function returns.
    """
    duration = timeout_duration_s if apply_timeout else reward_interval_s
    deadline_ns = onset_perf_ns + round(duration * 1_000_000_000)
    phase = "timeout_interval" if apply_timeout else "reward_interval"
    signals = []

    def wait_to(deadline):
        return wait_until(deadline / 1_000_000_000, poll_callback=poll_callback)

    def edge(name, active, setter, requested_duration=None):
        requested_ns = time.perf_counter_ns()
        setter(int(active))
        actual_ns = time.perf_counter_ns()
        signals.append(dict(
            trial_num=trial_num, event=f"{name}_{'on' if active else 'off'}",
            timestamp_perf_ns=actual_ns, requested_timestamp_perf_ns=requested_ns,
            requested_duration=requested_duration,
        ))
        return actual_ns

    try:
        if apply_timeout:
            if duration > 0:
                # Timeout replaces the entire reward interval, including any
                # pump delay or pulses in a conflicting reward mapping.
                try:
                    edge("buzzer", True, set_buzzer, duration)
                    if not wait_to(deadline_ns):
                        return None
                finally:
                    edge("buzzer", False, set_buzzer)
        elif num_pulses > 0 and pump_pulse_s > 0:
            if not wait_to(onset_perf_ns + round(pump_delay_s * 1_000_000_000)):
                return None
            for pulse_index in range(num_pulses):
                try:
                    on_ns = edge("pump", True, set_pump, pump_pulse_s)
                    if not wait_to(on_ns + round(pump_pulse_s * 1_000_000_000)):
                        return None
                finally:
                    off_ns = edge("pump", False, set_pump)
                if pulse_index + 1 < num_pulses:
                    if not wait_to(off_ns + round(inter_pump_interval_s * 1_000_000_000)):
                        return None

        if not wait_to(deadline_ns):
            return None
        return time.perf_counter_ns()
    finally:
        # Preserve edge timestamps, but submit rows only after outputs are off
        # and the ITI start is captured. More pump events must not add reward time.
        if duration > 0:
            logger.log_event(trial_num=trial_num, event=phase, event_type="lifecycle",
                             timestamp_perf_ns=onset_perf_ns, requested_duration=duration)
        for signal in signals:
            logger.log_signal(**signal)


def duration_requires_visible_phase(*, sequential: bool, is_memory: bool) -> bool:
    """Return whether duration represents a visible stimulus phase."""
    return bool(sequential) or bool(is_memory)


def validate_duration_for_presentation_mode(
    duration: float,
    *,
    sequential: bool,
    is_memory: bool,
    context: str = "active_foraging",
) -> None:
    """Validate duration semantics independently of display refresh rate."""
    duration_s = float(duration)
    if not math.isfinite(duration_s):
        raise ValueError(f"Invalid {context} timing config: duration must be finite.")
    if duration_requires_visible_phase(sequential=sequential, is_memory=is_memory):
        if duration_s <= 0.0:
            raise ValueError(
                f"Invalid {context} timing config: duration must be positive when "
                "sequential=true or is_memory=true. In memory modes, duration is the "
                "stimulus display time before the dot-only choice period."
            )
        return

    if duration_s != 0.0:
        raise ValueError(
            f"Invalid {context} timing config: when sequential=false and "
            "is_memory=false, duration must be exactly 0. In this mode the stimuli "
            "appear on the first choice frame and remain visible for choice_time only."
        )
