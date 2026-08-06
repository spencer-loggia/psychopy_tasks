"""Pi-Plates DAQC2 digital-output helpers."""
from __future__ import annotations

import importlib
from typing import Callable, Optional


class DAQC2DigitalOutputs:
    """Small adapter around DAQC2plate open-drain DOUT writes."""

    def __init__(
        self,
        *,
        address: int = 0,
        module_name: str = "piplates.DAQC2plate",
        enabled: bool = True,
    ):
        self.address = self.validate_address(address)
        self.module_name = str(module_name)
        self.enabled = bool(enabled)
        self._module: Optional[object] = None
        self._set_dout_bit: Optional[Callable[[int, int], object]] = None
        self._clear_dout_bit: Optional[Callable[[int, int], object]] = None

    def open(self) -> None:
        if not self.enabled:
            return
        module = importlib.import_module(self.module_name)
        missing = [
            name
            for name in ("setDOUTbit", "clrDOUTbit")
            if not hasattr(module, name)
        ]
        if missing:
            raise RuntimeError(
                f"{self.module_name} does not expose required DAQC2 DOUT functions: {', '.join(missing)}"
            )
        self._module = module
        self._set_dout_bit = module.setDOUTbit
        self._clear_dout_bit = module.clrDOUTbit

    def write(self, bit: int, active: bool) -> None:
        """Set a DOUT bit logically on/off.

        ``active=True`` calls ``setDOUTbit`` and turns on the open-drain
        transistor. With a pull-up or high-side load, that pulls the DOUT
        terminal near 0 V. ``active=False`` calls ``clrDOUTbit`` and turns the
        transistor off, allowing the terminal to return high.
        """
        bit = self.validate_bit(bit)
        if not self.enabled:
            return
        if self._module is None:
            self.open()
        if self._set_dout_bit is None or self._clear_dout_bit is None:
            return
        if active:
            self._set_dout_bit(self.address, bit)
        else:
            self._clear_dout_bit(self.address, bit)

    def bind_bit(self, bit: int) -> tuple[Callable[[], None], Callable[[], None]]:
        """Return prevalidated on/off callables for a DOUT bit."""
        bit = self.validate_bit(bit)

        def set_bit() -> None:
            if not self.enabled:
                return
            if self._set_dout_bit is None:
                self.open()
            if self._set_dout_bit is not None:
                self._set_dout_bit(self.address, bit)

        def clear_bit() -> None:
            if not self.enabled:
                return
            if self._clear_dout_bit is None:
                self.open()
            if self._clear_dout_bit is not None:
                self._clear_dout_bit(self.address, bit)

        return set_bit, clear_bit

    @staticmethod
    def validate_address(value: int) -> int:
        address = int(value)
        if address < 0 or address > 7:
            raise ValueError("DAQC2 address must be in the range 0 through 7")
        return address

    @staticmethod
    def validate_bit(value: int) -> int:
        bit = int(value)
        if bit < 0 or bit > 7:
            raise ValueError("DAQC2 DOUT bit must be in the range 0 through 7")
        return bit
