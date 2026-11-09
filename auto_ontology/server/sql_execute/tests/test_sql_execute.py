# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``POST /api/sql/execute``: caller IP fence, read-only checks, row cap."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auto_ontology.connectors.limited_result import collect_limited
from auto_ontology.server.sql_execute import router as sql_router
from auto_ontology.server.sql_execute.checks import SqlCheckError, check_sql

CATALOG = {"bf_rctmst_02", "tb_ordmst", "lu_kind"}


def _check(sql: str) -> None:
    check_sql(sql, dialect="tsql", database_name="wow2rpt", catalog_tables=CATALOG)


class TestChecks:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT TOP 10 * FROM bf_rctmst_02",
            "SELECT kindno, SUM(amt10) AS amt FROM bf_rctmst_02 GROUP BY kindno ORDER BY amt DESC",
            "WITH k AS (SELECT kindno FROM lu_kind) SELECT * FROM k JOIN tb_ordmst o ON 1=1",
            "SELECT a FROM tb_ordmst UNION ALL SELECT a FROM bf_rctmst_02",
            "SELECT REPLACE(name, ' ', '') AS n FROM lu_kind",
            "SELECT * FROM dbo.lu_kind",
            "SELECT * FROM wow2rpt.dbo.lu_kind",
        ],
    )
    def test_allowed(self, sql: str) -> None:
        _check(sql)

    @pytest.mark.parametrize(
        "sql",
        [
            "UPDATE lu_kind SET name = 'x'",
            "DELETE FROM tb_ordmst",
            "INSERT INTO lu_kind (kindno) VALUES (1)",
            "DROP TABLE lu_kind",
            "SELECT * INTO tmp_copy FROM lu_kind",
            "SELECT 1 FROM lu_kind; DELETE FROM lu_kind",
            "SELECT * FROM lu_kind; SELECT * FROM tb_ordmst",
            "EXEC sp_mkbf_rctmst_02",
            "SELECT * FROM sys_users",
            "SELECT * FROM otherdb.dbo.lu_kind",
            "SELECT 1",
            "SELECT * FROM OPENQUERY(remote, 'SELECT 1')",
            "not sql at all (",
        ],
    )
    def test_rejected(self, sql: str) -> None:
        with pytest.raises(SqlCheckError):
            _check(sql)

    def test_empty_catalog_rejected(self) -> None:
        with pytest.raises(SqlCheckError):
            check_sql(
                "SELECT * FROM lu_kind",
                dialect="tsql",
                database_name="x",
                catalog_tables=[],
            )


class _Connector:
    dialect = "tsql"
    database_name = "wow2rpt"
    supports_statement_timeout = True

    def __init__(self, rows: int) -> None:
        self.rows = rows
        self.calls: list[dict[str, Any]] = []

    def execute_limited(self, sql: str, **kwargs: Any) -> Any:
        self.calls.append({"sql": sql, **kwargs})
        data = [(i, f"name{i}") for i in range(self.rows)]
        batches = [data[i : i + 100] for i in range(0, len(data), 100)]
        return collect_limited(
            batches,
            ["kindno", "name"],
            preview_rows=kwargs["preview_rows"],
            max_rows=kwargs["max_rows"],
        )


@pytest.fixture
def connector(monkeypatch: pytest.MonkeyPatch) -> _Connector:
    conn = _Connector(rows=250)
    monkeypatch.setattr(sql_router, "_resolve_connector", lambda _db: conn)
    monkeypatch.setattr(sql_router, "catalog_table_names", lambda _db: CATALOG)
    return conn


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # TestClient reports its peer as "testclient"; allow it like a loopback caller.
    monkeypatch.setenv("AO_SQL_EXECUTE_ALLOWED_IPS", "testclient")
    app = FastAPI()
    app.include_router(sql_router.router, prefix="/api")
    return TestClient(app)


def _post(client: TestClient, **body: Any) -> Any:
    payload = {
        "sql": "SELECT kindno, name FROM lu_kind",
        "target_db": "wow2rpt",
        **body,
    }
    return client.post("/api/sql/execute", json=payload)


def test_returns_every_row(client: TestClient, connector: _Connector) -> None:
    response = _post(client)
    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 250
    assert len(body["rows"]) == 250
    assert body["columns"] == ["kindno", "name"]
    assert body["rows"][0] == {"kindno": 0, "name": "name0"}
    # The submitted SQL runs unchanged, capped in rows and time.
    call = connector.calls[0]
    assert call["sql"] == "SELECT kindno, name FROM lu_kind"
    assert call["max_rows"] == 50_000
    assert call["timeout_s"] == 120


def test_over_cap_rejected(client: TestClient, connector: _Connector) -> None:
    response = _post(client, max_rows=100)
    assert response.status_code == 413


def test_request_cap_never_exceeds_server_cap(
    client: TestClient, connector: _Connector, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AO_SQL_EXECUTE_MAX_ROWS", "300")
    assert _post(client, max_rows=1_000_000).status_code == 200
    assert connector.calls[-1]["max_rows"] == 300


def test_write_statement_rejected(client: TestClient, connector: _Connector) -> None:
    response = _post(client, sql="DELETE FROM lu_kind")
    assert response.status_code == 422
    assert connector.calls == []


def test_unknown_table_rejected(client: TestClient, connector: _Connector) -> None:
    assert _post(client, sql="SELECT * FROM secret_table").status_code == 422
    assert connector.calls == []


def test_non_loopback_caller_refused(
    connector: _Connector, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AO_SQL_EXECUTE_ALLOWED_IPS", raising=False)
    app = FastAPI()
    app.include_router(sql_router.router, prefix="/api")
    response = _post(TestClient(app))
    assert response.status_code == 403
    assert connector.calls == []


def test_default_allowed_ips_are_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AO_SQL_EXECUTE_ALLOWED_IPS", raising=False)
    assert sql_router.allowed_ips() == {"127.0.0.1", "::1"}


def test_query_error_is_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken(_Connector):
        def execute_limited(self, sql: str, **kwargs: Any) -> Any:
            raise RuntimeError("timeout")

    monkeypatch.setattr(sql_router, "_resolve_connector", lambda _db: Broken(0))
    monkeypatch.setattr(sql_router, "catalog_table_names", lambda _db: CATALOG)
    assert _post(client).status_code == 502


def test_dates_serialised(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    class Dated(_Connector):
        def execute_limited(self, sql: str, **kwargs: Any) -> Any:
            from auto_ontology.connectors.limited_result import LimitedResult

            frame = pd.DataFrame({"dt": pd.to_datetime(["2026-11-09"])})
            return LimitedResult(frame=frame, total_count=1, over_limit=False)

    monkeypatch.setattr(sql_router, "_resolve_connector", lambda _db: Dated(0))
    monkeypatch.setattr(sql_router, "catalog_table_names", lambda _db: CATALOG)
    body = _post(client).json()
    assert body["rows"][0]["dt"].startswith("2026-11-09")
