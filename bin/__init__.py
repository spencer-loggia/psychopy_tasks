"""
Lightweight package initializer for `bin`.

Avoid importing heavy submodule symbols at package-import time to prevent
import-time failures or circular imports. Consumers can import the submodules
directly (e.g. `from bin import utils`) or access `bin.utils`.
"""
from . import utils, logger, config

__all__ = ["utils", "logger", "config"]
