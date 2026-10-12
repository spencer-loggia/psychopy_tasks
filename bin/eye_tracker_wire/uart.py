from __future__ import annotations

import errno
import fcntl
import os
import termios
from pathlib import Path


def open_exclusive_uart(path: str | Path) -> int:
    """Open a tty and reject any later opener until this descriptor closes."""
    fd = os.open(Path(path), os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        if not os.isatty(fd):
            raise OSError(errno.ENOTTY, "UART path is not a tty device", str(path))
        # flock coordinates our own processes (including privileged ones), while
        # TIOCEXCL also rejects unrelated later tty opens such as a serial getty.
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request = getattr(termios, "TIOCEXCL", None)
        if request is None:
            raise OSError(
                errno.ENOTSUP,
                "This platform cannot claim exclusive tty ownership",
                str(path),
            )
        fcntl.ioctl(fd, request)
    except BaseException:
        os.close(fd)
        raise
    return fd


def configure_raw_uart(fd: int, baud: int, *, read_timeout_deciseconds: int) -> None:
    """Configure a tty as raw 8-N-1 with no software or hardware flow control."""
    if not 0 <= read_timeout_deciseconds <= 255:
        raise ValueError("read_timeout_deciseconds must be in [0, 255]")
    speed = getattr(termios, f"B{baud}", None)
    if speed is None:
        raise ValueError(f"UART baud {baud} is not supported by this operating system")
    attributes = termios.tcgetattr(fd)
    attributes[0] = 0
    attributes[1] = 0
    clear_flags = termios.CSIZE | termios.PARENB | termios.CSTOPB
    clear_flags |= getattr(termios, "CRTSCTS", 0)
    attributes[2] = (
        (attributes[2] & ~clear_flags) | termios.CS8 | termios.CREAD | termios.CLOCAL
    )
    attributes[3] = 0
    attributes[4] = speed
    attributes[5] = speed
    attributes[6][termios.VMIN] = 0
    attributes[6][termios.VTIME] = read_timeout_deciseconds
    termios.tcsetattr(fd, termios.TCSANOW, attributes)
    termios.tcflush(fd, termios.TCIOFLUSH)
