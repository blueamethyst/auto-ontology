# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persisted state of the schema ingest handler (see ``dal/schema.py``).

Both the ingestion service (which writes) and the web API (which reports
``state`` through ``/api/catalog/fingerprint``) read these rows, so the state is
the same whichever process asks.
"""

from __future__ import annotations

import os
import socket
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, delete, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from auto_ontology.dal import schema as s
from auto_ontology.dal.session import store

RESULT_SUCCEEDED = "succeeded"
RESULT_FAILED = "failed"
RESULT_PARTIAL = "partial"
RESULT_SKIPPED = "skipped"

#: A run whose heartbeat is older than this is considered dead.
HEARTBEAT_TIMEOUT = timedelta(minutes=5)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def process_owner() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


# ---------------------------------------------------------------- runs


def start_run(database_name: str, trigger: str) -> str:
    rows = store().query_write(
        insert(s.schema_ingest_run)
        .values(
            database_name=database_name,
            trigger=trigger,
            owner=process_owner(),
            stage="start",
        )
        .returning(s.schema_ingest_run.c.id)
    )
    return rows[0]["id"]


def heartbeat(run_id: str, stage: str | None = None, connection: Any = None) -> None:
    values: dict[str, Any] = {"heartbeat_at": func.now()}
    if stage is not None:
        values["stage"] = stage
    statement = (
        update(s.schema_ingest_run)
        .where(s.schema_ingest_run.c.id == run_id)
        .values(**values)
    )
    if connection is not None:
        connection.execute(statement)
        connection.commit()
    else:
        store().query_write(statement)


def finish_run(
    run_id: str,
    result: str,
    verification: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    store().query_write(
        update(s.schema_ingest_run)
        .where(s.schema_ingest_run.c.id == run_id)
        .values(
            result=result,
            verification=verification,
            error=(error or "")[:4000] or None,
            finished_at=func.now(),
            stage="done",
        )
    )


def reap_stale_runs(timeout: timedelta = HEARTBEAT_TIMEOUT) -> list[dict[str, Any]]:
    """Close unfinished runs whose heartbeat stopped. Call while holding the write lock.

    Each closed run leaves an open failure (stage ``run``) so the next pass redoes it.
    """
    cutoff = _now() - timeout
    stale = store().query_read(
        select(
            s.schema_ingest_run.c.id,
            s.schema_ingest_run.c.database_name,
            s.schema_ingest_run.c.stage,
        )
        .where(s.schema_ingest_run.c.finished_at.is_(None))
        .where(s.schema_ingest_run.c.heartbeat_at < cutoff)
    )
    for run in stale:
        finish_run(
            run["id"],
            RESULT_FAILED,
            error=f"process stopped during stage {run['stage']}",
        )
        record_failure(
            run["database_name"], "run", "", run["id"], f"interrupted in {run['stage']}"
        )
    return stale


def run_in_progress(database_name: str | None = None) -> bool:
    cutoff = _now() - HEARTBEAT_TIMEOUT
    query = (
        select(func.count().label("n"))
        .select_from(s.schema_ingest_run)
        .where(s.schema_ingest_run.c.finished_at.is_(None))
        .where(s.schema_ingest_run.c.heartbeat_at >= cutoff)
    )
    if database_name is not None:
        query = query.where(s.schema_ingest_run.c.database_name == database_name)
    return bool(store().query_read(query)[0]["n"])


def recent_runs(
    database_name: str | None = None, limit: int = 20
) -> list[dict[str, Any]]:
    query = (
        select(s.schema_ingest_run)
        .order_by(s.schema_ingest_run.c.started_at.desc())
        .limit(limit)
    )
    if database_name:
        query = query.where(s.schema_ingest_run.c.database_name == database_name)
    return store().query_read(query)


# ---------------------------------------------------------------- failures


def record_failure(
    database_name: str,
    stage: str,
    target: str,
    run_id: str | None,
    error: str | None = None,
) -> None:
    table = s.schema_open_failure
    statement = pg_insert(table).values(
        database_name=database_name,
        stage=stage,
        target=target or "",
        first_run_id=run_id,
        last_run_id=run_id,
        error=(error or "")[:2000] or None,
    )
    statement = statement.on_conflict_do_update(
        constraint="uq_schema_open_failure_key",
        set_={
            "attempts": table.c.attempts + 1,
            "last_run_id": statement.excluded.last_run_id,
            "last_failed_at": func.now(),
            "error": statement.excluded.error,
        },
    )
    store().query_write(statement)


def clear_failure(database_name: str, stage: str, target: str | None = None) -> None:
    """Delete open failures a run just redid successfully (all targets when ``target`` is None)."""
    table = s.schema_open_failure
    condition = and_(table.c.database_name == database_name, table.c.stage == stage)
    if target is not None:
        condition = and_(condition, table.c.target == target)
    store().query_write(delete(table).where(condition))


def open_failures(database_name: str | None = None) -> list[dict[str, Any]]:
    query = select(s.schema_open_failure).order_by(
        s.schema_open_failure.c.first_failed_at
    )
    if database_name is not None:
        query = query.where(s.schema_open_failure.c.database_name == database_name)
    return store().query_read(query)


def retry_due(failure: dict[str, Any], now: datetime | None = None) -> bool:
    """10 minutes apart for the first 3 attempts, then hourly (plan 6.2)."""
    now = now or _now()
    gap = timedelta(minutes=10) if int(failure["attempts"]) <= 3 else timedelta(hours=1)
    return now - failure["last_failed_at"] >= gap


# ---------------------------------------------------------------- snapshot


def get_snapshot(database_name: str) -> dict[str, Any] | None:
    rows = store().query_read(
        select(s.schema_snapshot).where(
            s.schema_snapshot.c.database_name == database_name
        )
    )
    return rows[0] if rows else None


def upsert_snapshot(database_name: str, **values: Any) -> None:
    values = {**values, "updated_at": func.now()}
    statement = pg_insert(s.schema_snapshot).values(
        database_name=database_name, **values
    )
    statement = statement.on_conflict_do_update(
        index_elements=["database_name"], set_=values
    )
    store().query_write(statement)


def bump_generation(database_name: str) -> int:
    upsert_snapshot(database_name)
    rows = store().query_write(
        update(s.schema_snapshot)
        .where(s.schema_snapshot.c.database_name == database_name)
        .values(generation=s.schema_snapshot.c.generation + 1, updated_at=func.now())
        .returning(s.schema_snapshot.c.generation)
    )
    return int(rows[0]["generation"])


def current_state(database_name: str) -> tuple[str, int | None]:
    """``(state, generation)`` for the fingerprint API (plan 10.1).

    initializing: no successful run yet / updating: a run is in progress /
    failed: an open failure exists / ready: otherwise.
    """
    snapshot = get_snapshot(database_name)
    generation = int(snapshot["generation"]) if snapshot else None
    if snapshot is None or not snapshot.get("last_success_run_id"):
        if run_in_progress(database_name):
            return "updating", generation
        return "initializing", generation
    if run_in_progress(database_name):
        return "updating", generation
    if open_failures(database_name):
        return "failed", generation
    return "ready", generation


# ---------------------------------------------------------------- change log


def log_changes(run_id: str, database_name: str, entries: list[dict[str, Any]]) -> None:
    if not entries:
        return
    store().query_write(
        insert(s.schema_change_log),
        [
            {
                "run_id": run_id,
                "database_name": database_name,
                "kind": e["kind"],
                "object": e["object"],
                "detail": e.get("detail"),
            }
            for e in entries
        ],
    )


def recent_changes(
    database_name: str | None = None, limit: int = 200
) -> list[dict[str, Any]]:
    query = (
        select(s.schema_change_log)
        .order_by(s.schema_change_log.c.id.desc())
        .limit(limit)
    )
    if database_name:
        query = query.where(s.schema_change_log.c.database_name == database_name)
    return store().query_read(query)


def orphan_count(database_name: str) -> int:
    rows = store().query_read(
        select(func.count().label("n"))
        .select_from(s.schema_change_log)
        .where(s.schema_change_log.c.database_name == database_name)
        .where(s.schema_change_log.c.kind == "semantic_orphaned")
    )
    return int(rows[0]["n"])
