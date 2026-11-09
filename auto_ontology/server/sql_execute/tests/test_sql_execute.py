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


def test_preview_mode_returns_preview_and_count(
    client: TestClient, connector: _Connector
) -> None:
    body = _post(client, preview_rows=100).json()
    assert body["total_count"] == 250
    assert len(body["rows"]) == 100
    assert body["over_limit"] is False
    assert connector.calls[-1]["preview_rows"] == 100


def test_preview_mode_reports_over_limit_instead_of_413(
    client: TestClient, connector: _Connector
) -> None:
    response = _post(client, preview_rows=10, max_rows=100)
    assert response.status_code == 200
    body = response.json()
    assert body["over_limit"] is True
    assert body["total_count"] == 100
    assert len(body["rows"]) == 10


def _fake_fp(target_db: str) -> dict[str, Any]:
    return {
        "content_hash": "c" * 64,
        "config_hash": "g" * 64,
        "structure_hash": "s" * 64,
        "structure_rows": 10,
        "counts": {},
        "validated_at": "2026-11-09T00:00:00+00:00",
    }


def test_fingerprint_response_and_memo(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from auto_ontology.dal import fingerprint as fp

    calls: list[str] = []

    def compute(target_db: str) -> dict[str, Any]:
        calls.append(target_db)
        return _fake_fp(target_db)

    monkeypatch.setattr(fp, "compute_fingerprint", compute)
    monkeypatch.setattr(sql_router, "_fingerprint_memo", {})
    monkeypatch.setattr(sql_router, "_state_and_generation", lambda db: ("ready", None))
    monkeypatch.setenv("AO_BUILD_VERSION", "abc123")
    body = client.get(
        "/api/catalog/fingerprint", params={"target_db": "wow2rpt"}
    ).json()
    assert body["state"] == "ready"
    assert body["build"] == "abc123"
    assert body["structure_hash"] == "s" * 64
    assert len(body["version"]) == 16
    assert body["warnings"] == []
    client.get("/api/catalog/fingerprint", params={"target_db": "wow2rpt"})
    assert len(calls) == 1  # memoised
    client.get(
        "/api/catalog/fingerprint", params={"target_db": "wow2rpt", "fresh": "true"}
    )
    assert len(calls) == 2


def test_fingerprint_unknown_build_warns(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from auto_ontology.dal import fingerprint as fp

    monkeypatch.setattr(fp, "compute_fingerprint", _fake_fp)
    monkeypatch.setattr(sql_router, "_fingerprint_memo", {})
    monkeypatch.setattr(sql_router, "_state_and_generation", lambda db: ("ready", None))
    monkeypatch.delenv("AO_BUILD_VERSION", raising=False)
    body = client.get(
        "/api/catalog/fingerprint", params={"target_db": "wow2rpt", "fresh": "true"}
    ).json()
    assert body["build"] == "unknown"
    assert body["state"] == "ready"
    assert body["warnings"]


def test_fingerprint_refuses_other_callers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AO_SQL_EXECUTE_ALLOWED_IPS", raising=False)
    app = FastAPI()
    app.include_router(sql_router.router, prefix="/api")
    response = TestClient(app).get(
        "/api/catalog/fingerprint", params={"target_db": "x"}
    )
    assert response.status_code == 403


def test_fingerprint_state_and_generation_change_version(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from auto_ontology.dal import fingerprint as fp

    monkeypatch.setattr(fp, "compute_fingerprint", _fake_fp)
    monkeypatch.setattr(sql_router, "_fingerprint_memo", {})
    monkeypatch.setattr(sql_router, "_state_and_generation", lambda db: ("ready", 4))
    a = client.get("/api/catalog/fingerprint", params={"target_db": "x"}).json()
    monkeypatch.setattr(sql_router, "_state_and_generation", lambda db: ("updating", 5))
    b = client.get("/api/catalog/fingerprint", params={"target_db": "x"}).json()
    assert a["generation"] == 4 and b["generation"] == 5
    assert b["state"] == "updating"
    assert a["version"] != b["version"]  # memo is bypassed when state/generation move
