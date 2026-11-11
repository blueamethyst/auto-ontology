# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read numeric tuning knobs from the environment."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def positive_int_env(name: str, default: int) -> int:
    """``int(os.environ[name])`` when it is a positive integer, else *default*.

    Unset, blank, non-numeric and non-positive values fall back to *default*
    (with a warning for the latter two), so a typo never disables a loop bound.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring non-integer %s=%r; using %s", name, raw, default)
        return default
    if value < 1:
        logger.warning("Ignoring non-positive %s=%r; using %s", name, raw, default)
        return default
    return value
