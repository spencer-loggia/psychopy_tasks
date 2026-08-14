"""Reversibly pause configured Raspberry Pi maintenance activity."""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Callable, Mapping, Optional


DEFAULT_SYSTEMD_UNITS = (
    "apt-daily.timer",
    "apt-daily-upgrade.timer",
    "man-db.timer",
    "logrotate.timer",
    "fstrim.timer",
    "e2scrub_all.timer",
    "cron.service",
    "anacron.service",
    "systemd-timesyncd.service",
)

BLOCKING_MAINTENANCE_UNITS = (
    "apt-daily.service",
    "apt-daily-upgrade.service",
)


class ExperimentQuietMode:
    """Stop active maintenance units and restore exactly those units later."""

    def __init__(
        self,
        units: tuple[str, ...],
        *,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        self.units = units
        self.runner = runner
        self.stopped_units: list[str] = []
        self.active = False

    @staticmethod
    def _mutating_command(action: str, unit: str) -> list[str]:
        prefix = [] if hasattr(os, "geteuid") and os.geteuid() == 0 else ["sudo", "-n"]
        return [*prefix, "systemctl", action, unit]

    def _run(self, command: list[str], *, check: bool) -> subprocess.CompletedProcess:
        try:
            return self.runner(
                command,
                check=check,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise RuntimeError(f"{' '.join(command)} failed: {detail}") from exc

    def enter(self) -> None:
        if self.active or not sys.platform.startswith("linux"):
            return
        self.active = True
        try:
            for unit in self.units:
                status = self._run(["systemctl", "is-active", "--quiet", unit], check=False)
                if status.returncode != 0:
                    continue
                self._run(self._mutating_command("stop", unit), check=True)
                self.stopped_units.append(unit)
            running = [
                unit
                for unit in BLOCKING_MAINTENANCE_UNITS
                if self._run(
                    ["systemctl", "is-active", "--quiet", unit], check=False
                ).returncode == 0
            ]
            if running:
                raise RuntimeError(
                    "Wait for active system maintenance to finish: " + ", ".join(running)
                )
        except Exception:
            self.exit()
            raise

    def exit(self) -> None:
        if not self.active:
            return
        stopped = list(reversed(self.stopped_units))
        self.stopped_units.clear()
        self.active = False
        errors = []
        failed = []
        for unit in stopped:
            try:
                self._run(self._mutating_command("start", unit), check=True)
            except Exception as exc:
                failed.append(unit)
                errors.append(f"{unit}: {exc}")
        if errors:
            self.stopped_units = list(reversed(failed))
            self.active = True
            raise RuntimeError("Could not restore quiet-mode units: " + "; ".join(errors))


def create_experiment_quiet_mode(
    cfg: Mapping[str, Any],
) -> Optional[ExperimentQuietMode]:
    value = cfg.get("experiment_quiet_mode", False)
    if value is False or value is None:
        return None
    if value is True:
        units = DEFAULT_SYSTEMD_UNITS
    elif isinstance(value, dict):
        raw_units = value.get("systemd_units", DEFAULT_SYSTEMD_UNITS)
        if not isinstance(raw_units, list) or not all(
            isinstance(unit, str) and unit.strip() for unit in raw_units
        ):
            raise ValueError(
                "experiment_quiet_mode.systemd_units must be a list of unit names"
            )
        units = tuple(dict.fromkeys(unit.strip() for unit in raw_units))
    else:
        raise ValueError("experiment_quiet_mode must be true, false, or an object")
    return ExperimentQuietMode(tuple(units))
