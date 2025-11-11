"""
Helpers to load and validate JSON configuration files for tasks.
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from json import JSONDecodeError


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        text = f.read()
        if not text.strip():
            raise ValueError(f"Config file is empty: {path}")
        # Accept files that may be wrapped in Markdown code fences (```json ... ```)
        s = text.strip()
        if s.startswith("```"):
            # remove first line (the fence) and last line if it's a fence
            lines = s.splitlines()
            if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
                s = "\n".join(lines[1:-1])
            else:
                # fallback: strip leading/trailing fences
                s = s.lstrip("`").rstrip("`")
        try:
            cfg = json.loads(s)
        except JSONDecodeError as e:
            # re-raise a clearer error that includes the filename
            raise ValueError(f"Error parsing JSON config '{path}': {e.msg} (line {e.lineno} col {e.colno})") from e
    if not isinstance(cfg, dict):
        raise ValueError("Config file must contain a top-level JSON object")
    return cfg


def _expect_key(cfg: Dict[str, Any], key: str):
    if key not in cfg:
        raise KeyError(f"Missing required config key: '{key}'")


def validate_config(cfg: Dict[str, Any], required: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Very small validator that ensures required keys exist and some basic checks.

    Parameters
    - cfg: loaded JSON dict
    - required: list of required top-level keys

    Returns cfg (unchanged) or raises an error.
    """
    if required is None:
        required = []
    for k in required:
        _expect_key(cfg, k)

    # Basic type checks for commonly expected keys (if present)
    if "n" in cfg:
        if not isinstance(cfg["n"], int) or cfg["n"] < 1:
            raise ValueError("Config 'n' must be an integer >= 1")
    if "duration" in cfg:
        if not (isinstance(cfg["duration"], int) or isinstance(cfg["duration"], float)) or cfg["duration"] <= 0:
            raise ValueError("Config 'duration' must be a positive number (seconds)")
    if "isi" in cfg:
        if not (isinstance(cfg["isi"], int) or isinstance(cfg["isi"], float)) or cfg["isi"] < 0:
            raise ValueError("Config 'isi' must be a non-negative number (seconds)")
    if "bg" in cfg:
        bg = cfg["bg"]
        if not (isinstance(bg, list) or isinstance(bg, tuple)) or len(bg) != 3:
            raise ValueError("Config 'bg' must be a list of three integers 0-255")
        for v in bg:
            if not isinstance(v, int) or v < 0 or v > 255:
                raise ValueError("Config 'bg' values must be integers in 0-255 range")

    # Validate paths if present
    if "images_dir" in cfg:
        p = Path(cfg["images_dir"])
        if not p.exists() or not p.is_dir():
            raise ValueError(f"images_dir does not exist or is not a directory: {cfg.get('images_dir')}")
    if "output_dir" in cfg:
        # Directory may not exist yet; that's okay, but parent should be writable.
        outp = Path(cfg["output_dir"]).resolve()
        try:
            outp.parent.exists()
        except Exception:
            raise ValueError(f"output_dir path seems invalid: {cfg.get('output_dir')}")

    return cfg
