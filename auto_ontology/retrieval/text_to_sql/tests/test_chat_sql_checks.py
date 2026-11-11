# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The chat path refuses statements the SQL execution API would refuse."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from auto_ontology.retrieval.text_to_sql.agents import sql_execution as mod


class _Connector:
    dialect = "tsql"
    database_name = "wow2rpt"


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM t",
        "SELECT 1; DROP TABLE t",
        "SELECT a INTO #tmp FROM t",
        "EXEC sp_who",
        "SELECT * FROM OPENQUERY(srv, 'SELECT 1')",
    ],
)
def test_unsafe_statements_are_not_executed(sql: str) -> None:
    ran: list[str] = []

    def fake_execute(connector: Any, statement: str) -> pd.DataFrame:
        ran.append(statement)
        return pd.DataFrame({"a": [1]})

    with patch.object(mod, "execute_chat_sql", fake_execute):
        response = mod._run_sql(sql, _Connector())
    assert ran == []
    assert response.error and response.error.startswith("Refused to run")


def test_read_only_query_runs() -> None:
    with patch.object(
        mod, "execute_chat_sql", lambda connector, statement: pd.DataFrame({"a": [1]})
    ):
        response = mod._run_sql(
            "WITH x AS (SELECT 1 AS a) SELECT REPLACE(CAST(a AS varchar), ' ', '') AS a FROM x",
            _Connector(),
        )
    assert response.error is None


def test_unparseable_but_read_only_query_still_runs() -> None:
    with (
        patch.object(
            mod,
            "execute_chat_sql",
            lambda connector, statement: pd.DataFrame({"a": [1]}),
        ),
        patch(
            "auto_ontology.server.sql_execute.checks.parse_single_query",
            side_effect=__import__(
                "auto_ontology.server.sql_execute.checks", fromlist=["SqlCheckError"]
            ).SqlCheckError("SQL could not be parsed: unsupported syntax"),
        ),
    ):
        response = mod._run_sql("SELECT a FROM t", _Connector())
    assert response.error is None
