# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preview + count reading (``limited_result``) and the MSSQL streaming path."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from auto_ontology.connectors import mssql
from auto_ontology.connectors.limited_result import collect_limited, limit_frame


def _batches(total: int, size: int) -> list[list[tuple[int]]]:
    rows = [(i,) for i in range(total)]
    return [rows[i : i + size] for i in range(0, total, size)]


class TestCollectLimited:
    def test_keeps_preview_and_counts_all(self) -> None:
        result = collect_limited(
            _batches(250, 40), ["n"], preview_rows=100, max_rows=1000
        )
        assert len(result.frame) == 100
        assert result.frame["n"].tolist() == list(range(100))
        assert result.total_count == 250
        assert result.over_limit is False
        assert result.sliced is True

    def test_small_result_not_sliced(self) -> None:
        result = collect_limited(
            _batches(3, 40), ["n"], preview_rows=100, max_rows=1000
        )
        assert result.total_count == 3
        assert result.sliced is False

    def test_stops_after_max_rows(self) -> None:
        consumed: list[int] = []

        def batches() -> Any:
            for batch in _batches(10_000, 10):
                consumed.append(len(batch))
                yield batch

        result = collect_limited(batches(), ["n"], preview_rows=5, max_rows=50)
        assert result.over_limit is True
        assert result.total_count == 50
        assert len(result.frame) == 5
        # Reading stops at the batch that crosses max_rows + 1.
        assert sum(consumed) == 60

    def test_exactly_max_rows_is_not_over(self) -> None:
        result = collect_limited(_batches(50, 10), ["n"], preview_rows=5, max_rows=50)
        assert result.over_limit is False
        assert result.total_count == 50

    def test_empty_result(self) -> None:
        result = collect_limited([], ["n"], preview_rows=5, max_rows=50)
        assert result.total_count == 0
        assert list(result.frame.columns) == ["n"]

    def test_invalid_limits(self) -> None:
        with pytest.raises(ValueError):
            collect_limited([], ["n"], preview_rows=5, max_rows=0)

    def test_limit_frame_matches(self) -> None:
        df = pd.DataFrame({"n": range(120)})
        result = limit_frame(df, preview_rows=100, max_rows=110)
        assert result.over_limit is True
        assert result.total_count == 110
        assert len(result.frame) == 100


class _FakeRawConn:
    def __init__(self) -> None:
        self.query_timeout = 0
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class _FakeCursor:
    def __init__(self, conn: "_FakeConn", total: int) -> None:
        self._conn = conn
        self._rows = [(i, f"r{i}") for i in range(total)]
        self._pos = 0
        self.description = None
        self.closed = False
        self.fetchall_called = False

    def execute(self, sql: str, *args: Any) -> None:
        self._conn.statements.append(sql)
        if not sql.startswith("SET "):
            self.description = (("id",), ("name",))

    def fetchmany(self, size: int) -> list[tuple[int, str]]:
        batch = self._rows[self._pos : self._pos + size]
        self._pos += len(batch)
        return batch

    def fetchall(self) -> list[tuple[int, str]]:
        self.fetchall_called = True
        return self._rows

    def close(self) -> None:
        self.closed = True


class _FakeConn:
    def __init__(self, total: int) -> None:
        self._conn = _FakeRawConn()
        self.statements: list[str] = []
        self.total = total
        self.cursors: list[_FakeCursor] = []

    def cursor(self, as_dict: bool = False) -> _FakeCursor:
        cursor = _FakeCursor(self, self.total)
        self.cursors.append(cursor)
        return cursor

    def commit(self) -> None:
        pass


def _db(total: int) -> tuple[mssql.MSSQLDatabase, _FakeConn]:
    db = mssql.MSSQLDatabase.__new__(mssql.MSSQLDatabase)
    db._config = {"server": "s", "port": 1433, "database": "wow2rpt"}
    conn = _FakeConn(total)
    db._conn = conn  # type: ignore[assignment]
    return db, conn


class TestMssqlExecuteLimited:
    def test_reads_in_batches_without_rewriting_sql(self) -> None:
        db, conn = _db(2_500)
        result = db.execute_limited(
            "SELECT id, name FROM t", preview_rows=100, max_rows=50_000, timeout_s=30
        )
        assert result.total_count == 2_500
        assert result.over_limit is False
        assert len(result.frame) == 100
        assert list(result.frame.columns) == ["id", "name"]
        assert "SELECT id, name FROM t" in conn.statements
        assert not conn.cursors[0].fetchall_called
        assert conn._conn.cancelled is False
        assert conn.cursors[0].closed

    def test_over_limit_cancels_rest(self) -> None:
        db, conn = _db(5_000)
        result = db.execute_limited(
            "SELECT id, name FROM t", preview_rows=10, max_rows=1_500
        )
        assert result.over_limit is True
        assert result.total_count == 1_500
        assert conn._conn.cancelled is True

    def test_timeout_sets_query_timeout_and_lock_timeout(self) -> None:
        db, conn = _db(1)
        db.execute_limited("SELECT 1", preview_rows=1, max_rows=1, timeout_s=45)
        assert conn._conn.query_timeout == 45
        assert "SET LOCK_TIMEOUT 45000" in conn.statements

    def test_execute_without_timeout_clears_query_timeout(self) -> None:
        db, conn = _db(1)
        conn._conn.query_timeout = 45
        db.execute("SELECT 1")
        assert conn._conn.query_timeout == 0


class TestMssqlDeadConnection:
    def _failing(self, *, connected: bool) -> tuple[mssql.MSSQLDatabase, _FakeConn]:
        db, conn = _db(1)
        conn._conn.connected = connected  # type: ignore[attr-defined]

        def boom(sql: str, *args: Any) -> None:
            if not sql.startswith("SET "):
                raise RuntimeError("TDS server connection timed out")

        original = conn.cursor

        def cursor(as_dict: bool = False) -> _FakeCursor:
            c = original(as_dict)
            c.execute = boom  # type: ignore[method-assign]
            return c

        conn.cursor = cursor  # type: ignore[method-assign]
        conn.close = lambda: None  # type: ignore[attr-defined]
        return db, conn

    def test_dead_connection_is_dropped_after_error(self) -> None:
        db, _ = self._failing(connected=False)
        with pytest.raises(RuntimeError):
            db.execute("SELECT 1", timeout_s=3)
        assert db._conn is None

    def test_dead_connection_dropped_in_limited_path(self) -> None:
        db, _ = self._failing(connected=False)
        with pytest.raises(RuntimeError):
            db.execute_limited("SELECT 1", preview_rows=1, max_rows=1, timeout_s=3)
        assert db._conn is None

    def test_live_connection_kept_after_sql_error(self) -> None:
        db, conn = self._failing(connected=True)
        with pytest.raises(RuntimeError):
            db.execute("SELECT broken")
        assert db._conn is conn


class TestMssqlQueryTimeout:
    def _timing_out(
        self, monkeypatch: pytest.MonkeyPatch, elapsed: float
    ) -> mssql.MSSQLDatabase:
        db, conn = _db(1)
        conn._conn.connected = True  # type: ignore[attr-defined]
        clock = iter([100.0, 100.0, 100.0 + elapsed, 100.0 + elapsed])
        monkeypatch.setattr(
            mssql.time, "monotonic", lambda: next(clock, 100.0 + elapsed)
        )

        def boom(sql: str, *args: Any) -> None:
            if not sql.startswith("SET "):
                raise RuntimeError(
                    "(20047, b'DB-Lib error message 20003, severity 6:\nTDS server connection timed out')"
                )

        original = conn.cursor

        def cursor(as_dict: bool = False) -> _FakeCursor:
            c = original(as_dict)
            c.execute = boom  # type: ignore[method-assign]
            return c

        conn.cursor = cursor  # type: ignore[method-assign]
        return db

    def test_timeout_after_cap_is_reported_as_query_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from auto_ontology.connectors.db_errors import is_infrastructure_error

        db = self._timing_out(monkeypatch, elapsed=30.2)
        with pytest.raises(mssql.MSSQLQueryTimeoutError) as exc:
            db.execute("SELECT 1", timeout_s=30)
        assert "30s" in str(exc.value)
        assert not is_infrastructure_error(exc.value, "SELECT 1")

    def test_limited_path_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db = self._timing_out(monkeypatch, elapsed=30.5)
        with pytest.raises(mssql.MSSQLQueryTimeoutError):
            db.execute_limited("SELECT 1", preview_rows=1, max_rows=1, timeout_s=30)

    def test_early_timeout_stays_a_connectivity_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = self._timing_out(monkeypatch, elapsed=2.0)
        with pytest.raises(RuntimeError) as exc:
            db.execute("SELECT 1", timeout_s=30)
        assert not isinstance(exc.value, mssql.MSSQLQueryTimeoutError)


class TestMssqlStaleConnectionRetry:
    """A connection that died while idle fails the next statement at once; reads retry."""

    def _db_failing_first(self, message: str) -> tuple[mssql.MSSQLDatabase, list[str]]:
        db, conn = _db(1)
        calls: list[str] = []
        original = conn.cursor

        def cursor(as_dict: bool = False) -> _FakeCursor:
            c = original(as_dict)
            real = c.execute

            def execute(sql: str, *args: Any) -> None:
                if not sql.startswith("SET "):
                    calls.append(sql)
                    if len(calls) == 1:
                        raise RuntimeError(message)
                real(sql, *args)

            c.execute = execute  # type: ignore[method-assign]
            return c

        conn.cursor = cursor  # type: ignore[method-assign]
        conn.close = lambda: None  # type: ignore[attr-defined]
        db._get_connection = lambda: conn  # type: ignore[method-assign]
        return db, calls

    def test_read_on_dead_connection_runs_once_more(self) -> None:
        db, calls = self._db_failing_first(
            "(20047, b'DB-Lib error message 20047, severity 9:\nDBPROCESS is dead or not enabled\n')"
        )
        frame = db.execute("SELECT 1")
        assert len(calls) == 2 and len(frame) == 1

    def test_limited_read_on_dead_connection_runs_once_more(self) -> None:
        db, calls = self._db_failing_first("DBPROCESS is dead or not enabled")
        result = db.execute_limited(
            "WITH x AS (SELECT 1 AS a) SELECT a FROM x", preview_rows=5, max_rows=5
        )
        assert len(calls) == 2 and result.total_count == 1

    def test_timeout_is_not_retried(self) -> None:
        db, calls = self._db_failing_first(
            "DB-Lib error message 20003: TDS server connection timed out; "
            "DB-Lib error message 20047: DBPROCESS is dead or not enabled"
        )
        with pytest.raises(RuntimeError):
            db.execute("SELECT 1")
        assert len(calls) == 1

    def test_write_is_not_retried(self) -> None:
        db, calls = self._db_failing_first("DBPROCESS is dead or not enabled")
        with pytest.raises(RuntimeError):
            db.execute("UPDATE t SET a = 1")
        assert len(calls) == 1
