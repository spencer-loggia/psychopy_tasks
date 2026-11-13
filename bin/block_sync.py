"""Block identity and acquisition gate shared by presentation tasks."""
from __future__ import annotations

import datetime as dt
import os

EXPERIMENT_NUMBER_ENV = "NEURO_SYNC_EXPERIMENT_NUMBER"
BLOCK_NUMBER_ENV = "NEURO_SYNC_BLOCK_NUMBER"
EXPERIMENT_DATE_ENV = "NEURO_SYNC_EXPERIMENT_DATE"
VIDEO_PATH_ENV = "NEURO_BLOCK_VIDEO_PATH"


def validate_identity(experiment_number, block_number, experiment_date):
    for name, number in (("experiment", experiment_number), ("block", block_number)):
        if type(number) is not int or not 0 <= number <= 255:
            raise ValueError(f"Sync {name} number must fit eight bits (0–255); numbering cannot wrap")
    if not 2000 <= experiment_date.year <= 2127:
        raise ValueError("Sync experiment date year must be between 2000 and 2127")
    return experiment_number, block_number, experiment_date


def identity_from_environment(started_at):
    return validate_identity(
        int(os.environ.get(EXPERIMENT_NUMBER_ENV, "0")),
        int(os.environ.get(BLOCK_NUMBER_ENV, "0")),
        dt.date.fromisoformat(os.environ[EXPERIMENT_DATE_ENV])
        if EXPERIMENT_DATE_ENV in os.environ else started_at.date(),
    )


def require_eye_acquisition(tracker, message_logger=None, *, screen_config=None, timeout_s=None):
    """Block presentation until ready or the operator explicitly accepts missing data."""
    if tracker is None:
        return True
    from .eye_tracker_warnings import confirm_without_eye_tracking

    try:
        if timeout_s is None:
            tracker.wait_until_ready()
        else:
            tracker.wait_until_ready(timeout_s=timeout_s)
    except (RuntimeError, TimeoutError) as exc:
        accepted = confirm_without_eye_tracking(str(exc), screen_config=screen_config)
        if message_logger is not None:
            message_logger.log("WARN", f"eye_acquisition_unavailable accepted={int(accepted)} error={exc}")
        if not accepted:
            raise RuntimeError("Block cancelled: eye acquisition was not confirmed") from exc
        return False
    else:
        if message_logger is not None:
            message_logger.log("INFO", "eye_acquisition_ready")
        return True


def configure_task_sync(win, config=None, *, session_logs=None, eye_tracker=None,
                       screen_config=None, **kwargs):
    import atexit
    import datetime as dt
    import json
    from .sync_display import configure_sync_bar as configure
    from .sync_bar import SyncBarConfig, START_CODE, END_CODE

    tracker = eye_tracker if eye_tracker is not None else (
        session_logs.eye_tracking if session_logs is not None else None
    )
    if not SyncBarConfig.from_mapping(config, sync_pin=kwargs.get("sync_pin", 18)).enabled:
        require_eye_acquisition(tracker, kwargs.get("msg_logger"), screen_config=screen_config)
        return None
    started_at = session_logs.session_clock.started_at if session_logs is not None else dt.datetime.now()
    experiment, block, date = identity_from_environment(started_at)
    def before_start():
        confirmed = require_eye_acquisition(tracker, kwargs.get("msg_logger"), screen_config=screen_config)
        if os.environ.get(VIDEO_PATH_ENV):
            from video.recorder import wait_for_fresh_picture
            wait_for_fresh_picture(os.environ[VIDEO_PATH_ENV])
            # Video startup may have taken seconds. Recheck eye freshness now;
            # if this requires operator confirmation, refresh video afterward.
            if tracker is not None and confirmed and not require_eye_acquisition(
                tracker, kwargs.get("msg_logger"), screen_config=screen_config, timeout_s=0,
            ):
                wait_for_fresh_picture(os.environ[VIDEO_PATH_ENV])

    display = configure(win, config, experiment_number=experiment, block_number=block,
                        before_start=before_start, **kwargs)
    finished = False
    failure = None
    protocol_path = session_logs.session_dir / "sync_protocol.json" if session_logs is not None else None
    protocol = {"version": 1, "experiment_number": experiment, "block_number": block,
                "experiment_date": date.isoformat(), "start_code": START_CODE,
                "end_code": END_CODE, "bit_order": "MSB first", "end_complete": False}

    def save_protocol():
        if protocol_path is not None:
            temporary = protocol_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(protocol, indent=2) + "\n")
            temporary.replace(protocol_path)

    def finalize():
        nonlocal finished, failure
        if finished:
            if failure is not None:
                raise failure
            return
        finished = True
        atexit.unregister(finalize)
        try:
            display.finish_block_signal(date)
            protocol["end_complete"] = True
            save_protocol()
        except Exception as exc:
            failure = exc
            if kwargs.get("msg_logger") is not None:
                kwargs["msg_logger"].log("ERROR", f"sync_footer_failed error={exc}")
            raise

    win._neuro_tasks_finish_sync = finalize
    if session_logs is not None:
        session_logs.finish_sync = finalize
    atexit.register(finalize)
    save_protocol()
    return display

