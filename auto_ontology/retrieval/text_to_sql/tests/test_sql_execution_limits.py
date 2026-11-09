# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Row limits (preview + total count) through SQL execution and the answer."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pandas as pd

from auto_ontology.retrieval.text_to_sql.agents import sql_execution
from auto_ontology.retrieval.text_to_sql.chat_sql import execute_chat_sql_limited


class _PlainConnector:
    """A connector with no ``execute_limited``: limits are applied after fetching."""

    dialect = "sqlite"
    database_name = "db"
    supports_statement_timeout = False

    def __init__(self, rows: int) -> None:
        self.frame = pd.DataFrame({"n": range(rows)})
        self.calls: list[str] = []

    def execute(self, sql: str) -> pd.DataFrame:
        self.calls.append(sql)
        return self.frame


def test_limited_fallback_for_plain_connector() -> None:
    connector = _PlainConnector(250)
    result = execute_chat_sql_limited(
        connector, "SELECT n FROM t", preview_rows=100, max_rows=1000
    )
    assert connector.calls == ["SELECT n FROM t"]
    assert len(result.frame) == 100
    assert result.total_count == 250
    assert result.over_limit is False


def test_limited_uses_connector_streaming_when_available() -> None:
    seen: dict[str, Any] = {}

    class Streaming(_PlainConnector):
        supports_statement_timeout = True

        def execute_limited(self, sql: str, **kwargs: Any) -> Any:
            seen.update(kwargs, sql=sql)
            return "streamed"

    result = execute_chat_sql_limited(
        Streaming(1), "SELECT 1", preview_rows=5, max_rows=10, timeout_s=7
    )
    assert result == "streamed"
    assert seen == {
        "sql": "SELECT 1",
        "preview_rows": 5,
        "max_rows": 10,
        "timeout_s": 7,
    }


def test_run_sql_without_limits_returns_every_row() -> None:
    response = sql_execution._run_sql("SELECT n FROM t", _PlainConnector(250))
    assert response.error is None
    assert len(json.loads(response.result[0])) == 250
    assert response.total_count is None
    assert response.sliced is False


def test_run_sql_with_limits_returns_preview_and_count() -> None:
    response = sql_execution._run_sql(
        "SELECT n FROM t", _PlainConnector(250), limits=(100, 50_000)
    )
    assert response.error is None
    assert len(json.loads(response.result[0])) == 100
    assert response.total_count == 250
    assert response.over_limit is False
    assert response.sliced is True


def test_run_sql_over_limit() -> None:
    response = sql_execution._run_sql(
        "SELECT n FROM t", _PlainConnector(120), limits=(10, 100)
    )
    assert response.over_limit is True
    assert response.total_count == 100


def test_result_limits_from_state() -> None:
    assert sql_execution._result_limits({}) is None  # type: ignore[arg-type]
    assert sql_execution._result_limits(
        {"result_limits": {"preview_rows": 100, "max_rows": 50_000}}  # type: ignore[arg-type]
    ) == (100, 50_000)
    assert sql_execution._result_limits(
        {"result_limits": {"preview_rows": 100}}  # type: ignore[arg-type]
    ) == (100, 100)


def test_agent_records_count_in_path_state() -> None:
    connector = _PlainConnector(250)
    state = {
        "path_state": {
            "sql_code": "SELECT n FROM t",
            "relevant_tables": [{"name": "t", "database_name": "db"}],
        },
        "connectors": [connector],
        "result_limits": {"preview_rows": 100, "max_rows": 50_000},
    }
    out = sql_execution.SQLExecutionAgent().execute(state)  # type: ignore[arg-type]
    path_state = out["path_state"]
    assert out["decision"] == "valid_sql"
    assert len(json.loads(path_state["sql_response_from_db"][0])) == 100
    assert path_state["sql_result_total_count"] == 250
    assert path_state["sql_result_over_limit"] is False
    assert path_state["sql_result_sliced"] is True


def test_agent_without_limits_leaves_path_state_unchanged() -> None:
    state = {
        "path_state": {
            "sql_code": "SELECT n FROM t",
            "relevant_tables": [{"name": "t", "database_name": "db"}],
        },
        "connectors": [_PlainConnector(3)],
    }
    out = sql_execution.SQLExecutionAgent().execute(state)  # type: ignore[arg-type]
    assert "sql_result_total_count" not in out["path_state"]


def test_chat_request_result_limits() -> None:
    from auto_ontology.server.chat.helpers import ChatRequest

    assert ChatRequest(question="q").result_limits() is None
    assert ChatRequest(
        question="q", preview_rows=100, max_rows=50_000
    ).result_limits() == {
        "preview_rows": 100,
        "max_rows": 50_000,
    }


def _response_state(**extra: Any) -> dict[str, Any]:
    return {
        "messages": [],
        "path_state": {
            "sql_generation_result": SimpleNamespace(
                sql_code="SELECT n FROM t", response="ok"
            ),
            "sql_response_from_db": ["[]"],
            **extra,
        },
    }


def test_response_includes_count_when_limited() -> None:
    from auto_ontology.retrieval.text_to_sql.agents.response import ResponseAgent

    out = ResponseAgent().execute(
        _response_state(
            sql_result_total_count=250,
            sql_result_over_limit=False,
            sql_result_sliced=True,
        )  # type: ignore[arg-type]
    )
    final = out["path_state"]["final_response"]
    assert final["total_count"] == 250
    assert final["over_limit"] is False
    assert final["sliced"] is True


def test_response_without_limits_has_no_count() -> None:
    from auto_ontology.retrieval.text_to_sql.agents.response import ResponseAgent

    out = ResponseAgent().execute(_response_state())  # type: ignore[arg-type]
    assert "total_count" not in out["path_state"]["final_response"]
