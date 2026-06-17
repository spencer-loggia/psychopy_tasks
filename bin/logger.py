from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping, Optional


EVENT_LOG_FILENAME = "event_log.tsv"
MESSAGE_LOG_FILENAME = "message_log.tsv"
BEHAVIOR_LOG_FILENAME = "behavior_log.tsv"
EVENT_CODE_LIBRARY_FILENAME = "event_code_library.json"
EVENT_NAME_LIBRARY_FILENAME = "event_name_library.json"
ALLOWED_EVENT_TYPES = frozenset({"frame_flip", "interaction", "signal"})
ALLOWED_MESSAGE_LEVELS = frozenset({"INFO", "WARN", "ERROR"})


@dataclass(frozen=True)
class EventDefinition:
    code: int
    description: str
    default_type: str


@dataclass(frozen=True)
class EventPatternDefinition:
    event_name_regex: re.Pattern[str]
    code_base: int
    code_group: str
    description_template: str
    default_type: str

    def resolve(self, event_name: str, event_type: str) -> Optional[EventDefinition]:
        normalized_type = _normalize_event_type(event_type)
        if normalized_type != self.default_type:
            return None
        match = self.event_name_regex.fullmatch(event_name)
        if match is None:
            return None
        group_value = match.group(self.code_group)
        code = int(self.code_base) + int(group_value)
        return EventDefinition(
            code=code,
            description=self.description_template.format(**match.groupdict()),
            default_type=self.default_type,
        )


def sanitize_filename_component(value: str) -> str:
    text = str(value).strip()
    if not text:
        return "unnamed"
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("._-") or "unnamed"


def _uniquify_dir(path: Path) -> Path:
    if not path.exists():
        return path
    idx = 1
    while True:
        candidate = path.parent / f"{path.name}_{idx}"
        if not candidate.exists():
            return candidate
        idx += 1


def _normalize_event_type(event_type: str) -> str:
    normalized = str(event_type).strip()
    if normalized not in ALLOWED_EVENT_TYPES:
        raise ValueError(
            f"Invalid event_type '{event_type}'. Expected one of {sorted(ALLOWED_EVENT_TYPES)}."
        )
    return normalized


def _normalize_message_level(level: str) -> str:
    normalized = str(level).strip().upper()
    if normalized not in ALLOWED_MESSAGE_LEVELS:
        raise ValueError(
            f"Invalid message level '{level}'. Expected one of {sorted(ALLOWED_MESSAGE_LEVELS)}."
        )
    return normalized


def default_event_name_library_path() -> Path:
    return Path(__file__).resolve().parents[1] / EVENT_NAME_LIBRARY_FILENAME


def _load_event_name_library_payload(library_path: Optional[str | Path] = None) -> dict[str, Any]:
    path = Path(library_path) if library_path is not None else default_event_name_library_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a top-level JSON object.")
    return payload


def load_task_event_definitions(
    task_name: Optional[str] = None,
    *,
    library_path: Optional[str | Path] = None,
) -> tuple[dict[str, EventDefinition], list[EventPatternDefinition]]:
    payload = _load_event_name_library_payload(library_path=library_path)
    events_section = payload.get("events")
    task_event_sets = payload.get("task_event_sets")
    template_section = payload.get("event_templates", {})

    if not isinstance(events_section, dict):
        raise ValueError("event_name_library.json must contain an 'events' object.")
    if task_name is not None and not isinstance(task_event_sets, dict):
        raise ValueError("event_name_library.json must contain a 'task_event_sets' object.")

    if task_name is None:
        selected_event_names = sorted(events_section)
        selected_template_payloads = list(template_section.values()) if isinstance(template_section, dict) else []
    else:
        if task_name not in task_event_sets:
            raise KeyError(f"Task '{task_name}' is not defined in {EVENT_NAME_LIBRARY_FILENAME}.")
        selected_event_names = list(task_event_sets[task_name])
        if not isinstance(selected_event_names, list):
            raise ValueError(f"task_event_sets['{task_name}'] must be a list of event names.")
        if not isinstance(template_section, dict):
            raise ValueError("event_name_library.json 'event_templates' must be an object when provided.")
        selected_template_payloads = [
            template_payload
            for template_payload in template_section.values()
            if task_name in tuple(template_payload.get("task_names", []))
        ]

    definitions: dict[str, EventDefinition] = {}
    for event_name in selected_event_names:
        event_payload = events_section.get(event_name)
        if not isinstance(event_payload, dict):
            raise KeyError(
                f"Event '{event_name}' referenced by task '{task_name}' is missing from {EVENT_NAME_LIBRARY_FILENAME}."
            )
        definitions[event_name] = EventDefinition(
            code=int(event_payload["code"]),
            description=str(event_payload["description"]),
            default_type=_normalize_event_type(event_payload["event_type"]),
        )

    patterns: list[EventPatternDefinition] = []
    for template_payload in selected_template_payloads:
        patterns.append(
            EventPatternDefinition(
                event_name_regex=re.compile(str(template_payload["event_name_regex"])),
                code_base=int(template_payload["code_base"]),
                code_group=str(template_payload["code_group"]),
                description_template=str(template_payload["description_template"]),
                default_type=_normalize_event_type(template_payload["event_type"]),
            )
        )
    return definitions, patterns


def build_session_dirname(
    task_name: str,
    config_name: str,
    when: Optional[dt.datetime] = None,
) -> str:
    timestamp = when or dt.datetime.now()
    safe_task = sanitize_filename_component(task_name)
    safe_config = sanitize_filename_component(config_name)
    return f"L_{timestamp:%Y%m%d%H%M%S}_{safe_task}_{safe_config}"


def build_session_output_dir(
    output_root: str | Path,
    task_name: str,
    config_name: str,
    when: Optional[dt.datetime] = None,
) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    session_dir = root / build_session_dirname(task_name, config_name, when=when)
    session_dir = _uniquify_dir(session_dir)
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


class SessionClock:
    def __init__(self, started_at: Optional[dt.datetime] = None, start_perf_s: Optional[float] = None):
        self.started_at = started_at or dt.datetime.now()
        self.start_perf_s = float(start_perf_s) if start_perf_s is not None else time.perf_counter()

    def seconds_since_start(self, timestamp_perf_s: Optional[float] = None) -> float:
        now_perf = float(timestamp_perf_s) if timestamp_perf_s is not None else time.perf_counter()
        return now_perf - self.start_perf_s


class EventCodeLibrary:
    def __init__(
        self,
        definitions: Optional[Mapping[str, EventDefinition]] = None,
        event_patterns: Optional[Iterable[EventPatternDefinition]] = None,
    ):
        self._definitions: dict[str, EventDefinition] = {}
        self._used_event_names: set[str] = set()
        self._event_name_by_code: dict[int, str] = {}
        self._patterns = list(event_patterns or [])

        for event_name, definition in dict(definitions or {}).items():
            self._register_definition(event_name, definition)

    def _register_definition(self, event_name: str, definition: EventDefinition) -> EventDefinition:
        normalized_event = str(event_name).strip()
        if not normalized_event:
            raise ValueError("Event name must be a non-empty string.")

        normalized_definition = EventDefinition(
            code=int(definition.code),
            description=str(definition.description),
            default_type=_normalize_event_type(definition.default_type),
        )
        existing_name = self._event_name_by_code.get(normalized_definition.code)
        if existing_name is not None and existing_name != normalized_event:
            raise ValueError(
                f"Duplicate event code {normalized_definition.code} for events '{existing_name}' and '{normalized_event}'."
            )
        self._definitions[normalized_event] = normalized_definition
        self._event_name_by_code[normalized_definition.code] = normalized_event
        return normalized_definition

    def ensure(self, event: str, event_type: str, description: Optional[str] = None) -> EventDefinition:
        normalized_event = str(event).strip()
        if not normalized_event:
            raise ValueError("Event name must be a non-empty string.")
        normalized_type = _normalize_event_type(event_type)

        existing = self._definitions.get(normalized_event)
        if existing is None:
            for pattern_definition in self._patterns:
                existing = pattern_definition.resolve(normalized_event, normalized_type)
                if existing is not None:
                    existing = self._register_definition(normalized_event, existing)
                    break

        if existing is None:
            raise KeyError(
                f"Event '{normalized_event}' is not defined in {EVENT_NAME_LIBRARY_FILENAME}."
            )
        if existing.default_type != normalized_type:
            raise ValueError(
                f"Event '{normalized_event}' is registered as type '{existing.default_type}', not '{normalized_type}'."
            )
        if description is not None and str(description).strip() and str(description).strip() != existing.description:
            raise ValueError(
                f"Event '{normalized_event}' description does not match the shared {EVENT_NAME_LIBRARY_FILENAME} entry."
            )
        self._used_event_names.add(normalized_event)
        return existing

    def export_used(self, out_path: str | Path) -> Path:
        payload: dict[str, Any] = {}
        for event_name in sorted(self._used_event_names, key=lambda name: self._definitions[name].code):
            definition = self._definitions[event_name]
            payload[str(definition.code)] = {
                "event": event_name,
                "event_type": definition.default_type,
                "description": definition.description,
            }
        out_file = Path(out_path)
        out_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return out_file


class EventLogger:
    def __init__(
        self,
        out_dir: str | Path,
        session_clock: SessionClock,
        event_library: EventCodeLibrary,
        filename: str = EVENT_LOG_FILENAME,
        auto_flush: bool = True,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / filename
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file, delimiter="\t")
        self.auto_flush = bool(auto_flush)
        self._pending_rows: list[list[str]] = []
        self._clock = session_clock
        self._library = event_library
        self._writer.writerow(
            [
                "trial_num",
                "time_since_session_start",
                "event",
                "event_code",
                "event_type",
                "requested_duration",
            ]
        )

    def seconds_since_session_start(self, timestamp_perf_s: Optional[float] = None) -> float:
        return self._clock.seconds_since_start(timestamp_perf_s)

    def log_event(
        self,
        *,
        trial_num: Optional[int],
        event: str,
        event_type: str,
        requested_duration: Optional[float] = None,
        timestamp_perf_s: Optional[float] = None,
        description: Optional[str] = None,
    ) -> None:
        if requested_duration is not None and float(requested_duration) < 0:
            raise ValueError("requested_duration must be non-negative when provided.")
        definition = self._library.ensure(event, event_type=event_type, description=description)
        timestamp_s = self.seconds_since_session_start(timestamp_perf_s)
        row = [
            "" if trial_num is None else str(int(trial_num)),
            f"{timestamp_s:.9f}",
            str(event).strip(),
            str(definition.code),
            definition.default_type,
            "" if requested_duration is None else f"{float(requested_duration):.9f}",
        ]
        if self.auto_flush:
            self._writer.writerow(row)
            self._file.flush()
        else:
            self._pending_rows.append(row)

    def log_frame_flip(
        self,
        *,
        trial_num: Optional[int],
        event: str,
        timestamp_perf_s: Optional[float],
        requested_duration: Optional[float] = None,
        description: Optional[str] = None,
    ) -> None:
        self.log_event(
            trial_num=trial_num,
            event=event,
            event_type="frame_flip",
            requested_duration=requested_duration,
            timestamp_perf_s=timestamp_perf_s,
            description=description,
        )

    def log_interaction(
        self,
        *,
        trial_num: Optional[int],
        event: str,
        timestamp_perf_s: Optional[float],
        description: Optional[str] = None,
    ) -> None:
        self.log_event(
            trial_num=trial_num,
            event=event,
            event_type="interaction",
            timestamp_perf_s=timestamp_perf_s,
            description=description,
        )

    def log_signal(
        self,
        *,
        trial_num: Optional[int],
        event: str,
        timestamp_perf_s: Optional[float],
        requested_duration: Optional[float] = None,
        description: Optional[str] = None,
    ) -> None:
        self.log_event(
            trial_num=trial_num,
            event=event,
            event_type="signal",
            requested_duration=requested_duration,
            timestamp_perf_s=timestamp_perf_s,
            description=description,
        )

    def flush(self) -> None:
        if self._pending_rows:
            self._writer.writerows(self._pending_rows)
            self._pending_rows.clear()
        self._file.flush()

    def close(self) -> None:
        self.flush()
        self._file.close()


class MessageLogger:
    def __init__(
        self,
        out_dir: str | Path,
        session_clock: SessionClock,
        filename: str = MESSAGE_LOG_FILENAME,
        auto_flush: bool = True,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / filename
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file, delimiter="\t")
        self.auto_flush = bool(auto_flush)
        self._pending_rows: list[list[str]] = []
        self._clock = session_clock
        self._writer.writerow(["time_since_session_start", "level", "message"])

    def log(self, level: str, message: str, *, timestamp_perf_s: Optional[float] = None) -> None:
        normalized_level = _normalize_message_level(level)
        timestr = f"{self._clock.seconds_since_start(timestamp_perf_s):.9f}"
        row = [timestr, normalized_level, str(message)]
        if self.auto_flush:
            self._writer.writerow(row)
            self._file.flush()
        else:
            self._pending_rows.append(row)

    def flush(self) -> None:
        if self._pending_rows:
            self._writer.writerows(self._pending_rows)
            self._pending_rows.clear()
        self._file.flush()

    def close(self) -> None:
        self.flush()
        self._file.close()


class BehaviorLogger:
    def __init__(
        self,
        out_dir: str | Path,
        fieldnames: list[str],
        filename: str = BEHAVIOR_LOG_FILENAME,
        auto_flush: bool = True,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / filename
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames, delimiter="\t")
        self._writer.writeheader()
        self.auto_flush = bool(auto_flush)
        self._pending_rows: list[dict[str, Any]] = []

    def writerow(self, row: Mapping[str, Any]) -> None:
        normalized = {key: row.get(key, "") for key in self._writer.fieldnames}
        if self.auto_flush:
            self._writer.writerow(normalized)
            self._file.flush()
        else:
            self._pending_rows.append(normalized)

    def flush(self) -> None:
        if self._pending_rows:
            self._writer.writerows(self._pending_rows)
            self._pending_rows.clear()
        self._file.flush()

    def close(self) -> None:
        self.flush()
        self._file.close()


class SessionLogBundle:
    def __init__(
        self,
        *,
        output_root: str | Path,
        task_name: str,
        config_name: str,
        behavior_fieldnames: Optional[list[str]] = None,
        when: Optional[dt.datetime] = None,
        auto_flush: bool = True,
        event_name_library_path: Optional[str | Path] = None,
    ):
        self.session_clock = SessionClock(started_at=when)
        self.session_dir = build_session_output_dir(
            output_root=output_root,
            task_name=task_name,
            config_name=config_name,
            when=self.session_clock.started_at,
        )
        definitions, event_patterns = load_task_event_definitions(
            task_name=task_name,
            library_path=event_name_library_path,
        )
        self.event_library = EventCodeLibrary(definitions, event_patterns=event_patterns)
        self.event_logger = EventLogger(
            self.session_dir,
            session_clock=self.session_clock,
            event_library=self.event_library,
            auto_flush=auto_flush,
        )
        self.message_logger = MessageLogger(
            self.session_dir,
            session_clock=self.session_clock,
            auto_flush=auto_flush,
        )
        self.behavior_logger = (
            BehaviorLogger(self.session_dir, fieldnames=behavior_fieldnames, auto_flush=auto_flush)
            if behavior_fieldnames
            else None
        )

    def flush(self) -> None:
        self.event_logger.flush()
        self.message_logger.flush()
        if self.behavior_logger is not None:
            self.behavior_logger.flush()

    def close(self) -> None:
        self.flush()
        self.event_library.export_used(self.session_dir / EVENT_CODE_LIBRARY_FILENAME)
        self.event_logger.close()
        self.message_logger.close()
        if self.behavior_logger is not None:
            self.behavior_logger.close()
