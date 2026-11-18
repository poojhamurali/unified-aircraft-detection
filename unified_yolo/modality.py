"""Modality bookkeeping shared by the dataset builder, trainer and model.

Every image in the unified dataset is a symlink named ``<prefix>__<original name>``,
so the sensor an image came from can be recovered from its file name alone. That
survives symlink resolution and needs no side table.
"""

from __future__ import annotations

import os

MODALITIES = ("optical", "ir", "sar")
PREFIX = {"optical": "opt", "ir": "ir", "sar": "sar"}
_FROM_PREFIX = {PREFIX[m]: i for i, m in enumerate(MODALITIES)}


def modality_of(path: str) -> int:
    """Index (0 optical, 1 ir, 2 sar) of an image, from its ``<prefix>__`` file name."""
    name = os.path.basename(str(path))
    prefix = name.split("__", 1)[0]
    if prefix not in _FROM_PREFIX:
        raise ValueError(
            f"{path!r} has no modality prefix ({'/'.join(_FROM_PREFIX)}); "
            "was it built by scripts/build_dataset.py?")
    return _FROM_PREFIX[prefix]
