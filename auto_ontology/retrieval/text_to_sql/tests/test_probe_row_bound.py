# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probes are row-bounded in the connector's own dialect (``TOP`` on T-SQL)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from auto_ontology.retrieval.text_to_sql.db_probe import executor as executor_mod
from auto_ontology.retrieval.text_to_sql.db_probe.executor import (
    ProbeExecutor,
    _ensure_limit,
)


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        (
            "SELECT [a] FROM [dbo].[t] WHERE [a] = 'x'",
            "SELECT TOP 20 [a] FROM [dbo].[t] WHERE [a] = 'x'",
        ),
        (
            "SELECT DISTINCT a FROM t ORDER BY a",
            "SELECT DISTINCT TOP 20 a FROM t ORDER BY a",
        ),
        (
            "WITH c AS (SELECT a FROM t) SELECT a FROM c",
            "WITH c AS (SELECT a AS a FROM t) SELECT TOP 20 a FROM c",
        ),
        ("SELECT TOP 5 a FROM t", "SELECT TOP 5 a FROM t"),  # already bounded
    ],
)
def test_tsql_probe_gets_top_not_limit(sql: str, expected: str) -> None:
    assert _ensure_limit(sql, 20, "tsql") == expected


def test_tsql_probe_that_does_not_parse_runs_as_written() -> None:
    sql = "SELECT a FROM t WHERE"
    assert _ensure_limit(sql, 20, "tsql") == sql


def test_other_dialects_keep_trailing_limit() -> None:
    assert (
        _ensure_limit("SELECT a FROM t;", 20, "postgres") == "SELECT a FROM t LIMIT 20"
    )
    assert (
        _ensure_limit("SELECT a FROM t LIMIT 3", 20, None) == "SELECT a FROM t LIMIT 3"
    )


def test_executor_bounds_with_the_connector_dialect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def _run(connector: Any, sql: str, **_: Any) -> pd.DataFrame:
        seen.append(sql)
        return pd.DataFrame({"a": [1]})

    monkeypatch.setattr(executor_mod, "execute_chat_sql", _run)
    with ProbeExecutor(SimpleNamespace(dialect="tsql"), max_rows=20) as ex:
        entry = ex.run("SELECT a FROM t", purpose="test")
    assert entry["ok"]
    assert seen == ["SELECT TOP 20 a FROM t"]
