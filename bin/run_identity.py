"""Stable UUIDs shared by experiment metadata, task blocks, and UART commands."""
from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import UUID, uuid4

EXPERIMENT_ID_ENV = "NEURO_EXPERIMENT_ID"
BLOCK_ID_ENV = "NEURO_BLOCK_ID"
BLOCK_IDENTITY_FILENAME = "block.json"


def new_run_id() -> str:
    return str(uuid4())


def prepare_block_identity(directory: Path) -> dict:
    """Persist standalone IDs, or verify the launcher's IDs before acquisition."""
    path = directory / BLOCK_IDENTITY_FILENAME
    identity = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(identity, dict):
        raise ValueError("Block identity must be a JSON object")
    experiment_id = os.environ.get(EXPERIMENT_ID_ENV, identity.get("experiment_id"))
    block_id = os.environ.get(BLOCK_ID_ENV) or identity.get("block_id") or new_run_id()
    for value in (experiment_id, block_id):
        if value is not None:
            parsed = UUID(value)
            if str(parsed) != value or parsed.int == 0:
                raise ValueError("Run IDs must be canonical nonzero UUIDs")
    for key, value in (("experiment_id", experiment_id), ("block_id", block_id)):
        if key in identity and identity[key] != value:
            raise ValueError(f"{key} does not match the block's saved identity")
        identity[key] = value
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return identity
