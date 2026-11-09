# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``POST /api/sql/execute`` — run one checked SELECT and return every row.

Built for a co-located caller (DABI) that showed a preview from the chat API
and now needs the full result for an Excel download. It is not a general SQL
console:

* only callers whose IP is in ``AO_SQL_EXECUTE_ALLOWED_IPS`` are served
  (default loopback only). Bind the server to ``127.0.0.1`` or firewall port
  3001 as well; the IP check is the second fence, not the first. With Docker,
  publish as ``-p 127.0.0.1:3001:3001`` and allow the bridge gateway address.
* the statement must pass :mod:`auto_ontology.server.sql_execute.checks`
  (single SELECT, no writes, catalog tables of the target database only);
* at most ``AO_SQL_EXECUTE_MAX_ROWS`` rows (default 50,000) — a larger result
  is rejected with 413 rather than returned partially;
* run time is capped by ``AO_SQL_EXECUTE_TIMEOUT_SECONDS`` (default 120).

The Next.js frontend has no route for this path on purpose: a same-host proxy
would make every browser request look like loopback.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from auto_ontology.server.sql_execute.checks import SqlCheckError, check_sql

logger = logging.getLogger(__name__)

router = APIRouter()

_DEFAULT_ALLOWED_IPS = "127.0.0.1,::1"
_DEFAULT_MAX_ROWS = 50_000
_DEFAULT_TIMEOUT_SECONDS = 120

# Cached connectors hold one driver connection each and are not safe to share
# between threads; downloads are rare, so run them one at a time.
_execute_lock = threading.Lock()


class SqlExecuteRequest(BaseModel):
    """One statement to run against one connected database."""

    sql: str = Field(..., min_length=1, max_length=100_000)
    target_db: str = Field(
        ..., min_length=1, description="Connected database name (as in the catalog)."
    )
    max_rows: int | None = Field(
        default=None,
        ge=1,
        description="Row cap for this call; never above the server cap.",
    )
    preview_rows: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Return only this many rows plus the full count (``total_count``, "
            "``over_limit``) instead of every row. A result over the cap is then "
            "reported with ``over_limit`` rather than rejected with 413."
        ),
    )


class SqlExecuteResponse(BaseModel):
    """Every row of the result, or a preview when ``preview_rows`` was sent."""

    columns: list[str]
    rows: list[dict[str, Any]]
    total_count: int
    over_limit: bool = False
    execution_time_ms: int


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring non-integer %s=%r", name, raw)
        return default


def allowed_ips() -> set[str]:
    """Caller IPs the API serves (``AO_SQL_EXECUTE_ALLOWED_IPS``, comma separated)."""
    raw = os.environ.get("AO_SQL_EXECUTE_ALLOWED_IPS") or _DEFAULT_ALLOWED_IPS
    return {ip.strip() for ip in raw.split(",") if ip.strip()}


def max_rows_cap() -> int:
    return max(1, _env_int("AO_SQL_EXECUTE_MAX_ROWS", _DEFAULT_MAX_ROWS))


def timeout_seconds() -> int:
    return max(1, _env_int("AO_SQL_EXECUTE_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT_SECONDS))


def _resolve_connector(target_db: str) -> Any:
    from auto_ontology.connectors import get_connectors
    from auto_ontology.retrieval.text_to_sql.connector_routing import (
        resolve_target_database_name,
    )

    connectors = get_connectors()
    try:
        database_name = resolve_target_database_name(target_db, connectors)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    for connector in connectors:
        if getattr(connector, "database_name", None) == database_name:
            return connector
    raise HTTPException(
        status_code=404, detail=f"Database {target_db!r} is not connected."
    )


def catalog_table_names(database_name: str) -> set[str]:
    """Table names the catalog holds for *database_name* (empty if none)."""
    from auto_ontology.dal.datasources import (
        fetch_schema_ids_for_database,
        fetch_schemas_by_ids,
    )

    schema_ids = fetch_schema_ids_for_database(database_name)
    if not schema_ids:
        # An empty id list means "every schema" to fetch_schemas_by_ids.
        return set()
    return {
        str(row["table_name"])
        for row in fetch_schemas_by_ids(schema_ids)
        if row.get("table_name")
    }


@router.post(
    "/sql/execute",
    response_model=SqlExecuteResponse,
    responses={
        403: {"description": "Caller IP is not allowed"},
        404: {"description": "Target database is not connected"},
        413: {"description": "Result has more rows than the cap"},
        422: {"description": "Statement failed the read-only / catalog checks"},
    },
)
def execute_sql(
    request: SqlExecuteRequest, http_request: Request
) -> SqlExecuteResponse:
    """Run one checked SELECT on a connected database and return every row.

    For a co-located service that needs the full result of a query it already
    previewed through chat (for example an Excel download). Only callers in
    ``AO_SQL_EXECUTE_ALLOWED_IPS`` (default loopback) are served. The statement
    must be a single SELECT that reads only catalog tables of ``target_db``.
    A result larger than the row cap (``AO_SQL_EXECUTE_MAX_ROWS``, default
    50,000, or ``max_rows`` when lower) is rejected with 413. With
    ``preview_rows`` only that many rows come back with the full count, and a
    result over the cap is reported as ``over_limit`` instead (used to refresh a
    cached query's preview without re-running text-to-SQL).
    """
    from auto_ontology.retrieval.text_to_sql.chat_sql import execute_chat_sql_limited

    client_ip = http_request.client.host if http_request.client else ""
    if client_ip not in allowed_ips():
        logger.warning("SQL execute API refused caller %s", client_ip or "<unknown>")
        raise HTTPException(status_code=403, detail="Caller is not allowed.")

    cap = max_rows_cap()
    max_rows = min(request.max_rows or cap, cap)

    connector = _resolve_connector(request.target_db)
    database_name = str(connector.database_name)
    try:
        check_sql(
            request.sql,
            dialect=getattr(connector, "dialect", None),
            database_name=database_name,
            catalog_tables=catalog_table_names(database_name),
        )
    except SqlCheckError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    started = time.perf_counter()
    try:
        with _execute_lock:
            result = execute_chat_sql_limited(
                connector,
                request.sql,
                preview_rows=(
                    max_rows
                    if request.preview_rows is None
                    else min(request.preview_rows, max_rows)
                ),
                max_rows=max_rows,
                kind="SQL execute API",
                timeout_s=timeout_seconds(),
            )
    except Exception as exc:
        logger.exception("SQL execute API query failed")
        raise HTTPException(status_code=502, detail=f"Query failed: {exc}") from exc
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if result.over_limit and request.preview_rows is None:
        raise HTTPException(
            status_code=413,
            detail=f"Result has more than {max_rows} rows; narrow the query.",
        )

    from auto_ontology.retrieval.text_to_sql.agents.sql_execution import (
        _deduplicate_result_columns,
    )

    frame = _deduplicate_result_columns(result.frame)
    rows = (
        json.loads(
            frame.to_json(orient="records", date_format="iso", default_handler=str)
        )
        if len(frame)
        else []
    )
    return SqlExecuteResponse(
        columns=[str(c) for c in frame.columns],
        rows=rows,
        total_count=result.total_count,
        over_limit=result.over_limit,
        execution_time_ms=elapsed_ms,
    )
