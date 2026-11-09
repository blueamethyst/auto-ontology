# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Row-limited query results: keep a preview, count the rest, stop at a cap.

Used when a caller wants the first ``preview_rows`` rows and the total row
count without holding the full result in memory. Reading stops once
``max_rows + 1`` rows have been seen, so a result larger than ``max_rows`` is
reported as ``over_limit`` rather than counted exactly.

The SQL is never rewritten (no ``COUNT(*)`` wrapper, no ``TOP``): rows are read
as the statement returns them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class LimitedResult:
    """Preview rows plus how many rows the statement returned.

    ``total_count`` is exact when ``over_limit`` is False. When it is True the
    statement returned more than ``max_rows`` rows and ``total_count`` is
    ``max_rows``.
    """

    frame: pd.DataFrame
    total_count: int
    over_limit: bool

    @property
    def sliced(self) -> bool:
        """True when ``frame`` holds fewer rows than the statement returned."""
        return self.over_limit or len(self.frame) < self.total_count


def validate_limits(preview_rows: int, max_rows: int) -> None:
    """Reject limits that cannot describe a result."""
    if preview_rows < 0:
        raise ValueError("preview_rows must be >= 0")
    if max_rows < 1:
        raise ValueError("max_rows must be >= 1")


def collect_limited(
    batches: Iterable[Sequence[Any]],
    columns: list[str],
    *,
    preview_rows: int,
    max_rows: int,
) -> LimitedResult:
    """Consume row *batches* until ``max_rows + 1`` rows have been seen.

    Only the first ``preview_rows`` rows are kept. The caller is responsible
    for cancelling whatever is left on the server once this returns early.
    """
    validate_limits(preview_rows, max_rows)
    kept: list[Any] = []
    seen = 0
    for batch in batches:
        if not batch:
            break
        room = preview_rows - len(kept)
        if room > 0:
            kept.extend(batch[:room])
        seen += len(batch)
        if seen > max_rows:
            break
    over_limit = seen > max_rows
    return LimitedResult(
        frame=pd.DataFrame.from_records(kept, columns=columns),
        total_count=max_rows if over_limit else seen,
        over_limit=over_limit,
    )


def limit_frame(df: pd.DataFrame, *, preview_rows: int, max_rows: int) -> LimitedResult:
    """Apply the same limits to an already-fetched frame (connectors without streaming)."""
    validate_limits(preview_rows, max_rows)
    over_limit = len(df) > max_rows
    return LimitedResult(
        frame=df.head(preview_rows).reset_index(drop=True),
        total_count=max_rows if over_limit else len(df),
        over_limit=over_limit,
    )
