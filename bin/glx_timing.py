"""Small native GLX queries used by the X11 timing diagnostic."""
from __future__ import annotations

import ctypes
import ctypes.util
import sys
from typing import Optional


def _error_text(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _current_context():
    if not sys.platform.startswith("linux"):
        raise RuntimeError("native GLX queries require Linux/X11")
    libgl = ctypes.CDLL(ctypes.util.find_library("GL") or "libGL.so.1")
    libx11 = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
    libgl.glXGetCurrentDisplay.restype = ctypes.c_void_p
    libgl.glXGetCurrentDrawable.restype = ctypes.c_ulong
    display = libgl.glXGetCurrentDisplay()
    drawable = libgl.glXGetCurrentDrawable()
    if not display or not drawable:
        raise RuntimeError("the PsychoPy window has no current GLX display/drawable")

    libx11.XDefaultScreen.argtypes = [ctypes.c_void_p]
    libx11.XDefaultScreen.restype = ctypes.c_int
    libgl.glXQueryExtensionsString.argtypes = [ctypes.c_void_p, ctypes.c_int]
    libgl.glXQueryExtensionsString.restype = ctypes.c_char_p
    raw_extensions = libgl.glXQueryExtensionsString(
        display,
        libx11.XDefaultScreen(display),
    )
    extensions = set((raw_extensions or b"").decode("ascii", "replace").split())
    return libgl, display, drawable, extensions


def _proc_address(libgl, name: bytes) -> Optional[int]:
    libgl.glXGetProcAddressARB.argtypes = [ctypes.c_char_p]
    libgl.glXGetProcAddressARB.restype = ctypes.c_void_p
    return libgl.glXGetProcAddressARB(name)


def query_glx_swap_interval() -> tuple[Optional[int], str]:
    """Read the swap interval stored for the current X11 drawable."""
    try:
        libgl, display, drawable, extensions = _current_context()
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
            address = _proc_address(libgl, b"glXGetSwapIntervalMESA")
            if address:
                get_interval = ctypes.CFUNCTYPE(ctypes.c_int)(address)
                return int(get_interval()), "GLX_MESA_swap_control"
            return None, "GLX_MESA_swap_control is advertised but not callable"

        return None, "the current GLX driver exposes no queryable swap-control extension"
    except Exception as exc:
        return None, f"GLX swap-interval query failed: {_error_text(exc)}"


def query_glx_sync_values() -> tuple[Optional[dict[str, int]], str]:
    """Read the GLX retrace and completed-swap counters for the drawable."""
    try:
        libgl, display, drawable, extensions = _current_context()
        if "GLX_OML_sync_control" not in extensions:
            return None, "GLX_OML_sync_control is not advertised"
        address = _proc_address(libgl, b"glXGetSyncValuesOML")
        if not address:
            return None, "GLX_OML_sync_control is advertised but not callable"
        get_values = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
        )(address)
        ust = ctypes.c_int64()
        msc = ctypes.c_int64()
        sbc = ctypes.c_int64()
        if not get_values(
            display,
            drawable,
            ctypes.byref(ust),
            ctypes.byref(msc),
            ctypes.byref(sbc),
        ):
            return None, "glXGetSyncValuesOML returned false"
        return {
            "ust": int(ust.value),
            "msc": int(msc.value),
            "sbc": int(sbc.value),
        }, "GLX_OML_sync_control"
    except Exception as exc:
        return None, f"GLX sync-counter query failed: {_error_text(exc)}"
