# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source schema signature, computed the same way the catalog stores it.

The watch pass and the ingest handler both reduce a schema to the same rows:

* tables  ``(schema, table, type)``
* columns ``(schema, table, column, type name, nullable)``
* PKs     ``(schema, table, position, column)`` -- only for tables kept by the filter
* FKs     ``(schema, table, column, ref schema, ref table, ref column)`` -- only when
  both ends are kept

The watch pass reads them with the connector's own metadata functions and the
same table filter the ingest uses (no separate SQL to keep in step), skipping
``get_views`` (view text is not stored) and ``get_queries`` (not structure, and
costly). It never reads data tables. ``object_id``/``modify_date`` are not part
of the connector output, so recreating a table with the same shape does not
change the signature.

:func:`catalog_rows` reads the same rows back from the AO catalog, which the
handler uses to verify a write and to diff the store against a fresh extract.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd
from sqlalchemy import text

from auto_ontology.catalog import table_filter
from auto_ontology.dal.session import store

Rows = dict[str, list[tuple]]

_KINDS = ("tables", "columns", "pks", "fks")


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().upper() in ("YES", "Y", "TRUE", "T", "1")
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    return bool(value)


def _s(value: Any) -> str:
    return (
        ""
        if value is None or (isinstance(value, float) and pd.isna(value))
        else str(value)
    )


def normalize_rows(
    tables: pd.DataFrame | None,
    columns: pd.DataFrame | None,
    pks: pd.DataFrame | None,
    fks: pd.DataFrame | None,
) -> Rows:
    """Reduce frames (raw connector output or the ingest's ``data`` dict) to rows."""
    tables = tables if tables is not None else pd.DataFrame()
    columns = columns if columns is not None else pd.DataFrame()
    table_rows = sorted(
        {
            (
                _s(r.get("table_schema")),
                _s(r.get("table_name")),
                _s(r.get("table_type")).lower(),
            )
            for r in tables.to_dict(orient="records")
        }
    )
    kept = {(t[0], t[1]) for t in table_rows}
    column_rows = sorted(
        {
            (
                _s(r.get("table_schema")),
                _s(r.get("table_name")),
                _s(r.get("column_name")),
                _s(r.get("data_type")).lower(),
                _truthy(r.get("is_nullable")),
            )
            for r in columns.to_dict(orient="records")
            if (_s(r.get("table_schema")), _s(r.get("table_name"))) in kept
        }
    )
    pk_rows: list[tuple] = []
    if pks is not None and len(pks):
        for r in pks.to_dict(orient="records"):
            key = (_s(r.get("table_schema")), _s(r.get("table_name")))
            if key in kept:
                pk_rows.append(
                    (
                        *key,
                        int(r.get("ordinal_position") or 0),
                        _s(r.get("column_name")),
                    )
                )
    fk_rows: list[tuple] = []
    if fks is not None and len(fks):
        for r in fks.to_dict(orient="records"):
            src = (_s(r.get("table_schema")), _s(r.get("table_name")))
            dst = (_s(r.get("referenced_schema")), _s(r.get("referenced_table")))
            if src in kept and dst in kept:
                fk_rows.append(
                    (
                        *src,
                        _s(r.get("column_name")),
                        *dst,
                        _s(r.get("referenced_column")),
                    )
                )
    return {
        "tables": table_rows,
        "columns": column_rows,
        "pks": sorted(set(pk_rows)),
        "fks": sorted(set(fk_rows)),
    }


def signature(rows: Rows) -> str:
    payload = json.dumps(
        {k: [list(r) for r in rows.get(k, [])] for k in _KINDS}, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def rows_to_json(rows: Rows) -> dict[str, list[list]]:
    return {k: [list(r) for r in rows.get(k, [])] for k in _KINDS}


def rows_from_json(data: dict[str, Any] | None) -> Rows:
    data = data or {}
    return {k: [tuple(r) for r in data.get(k, [])] for k in _KINDS}


def observe(connector: Any) -> Rows:
    """Read the source structure with the ingest's metadata calls and filter."""
    tables = connector.get_tables()
    columns = connector.get_columns()
    pks = connector.get_pks()
    fks = connector.get_fks()
    tables, columns, _ = table_filter.apply(
        tables,
        columns,
        None,
        getattr(connector, "table_filter", None) or table_filter.TableFilter(),
    )
    return normalize_rows(tables, columns, pks, fks)


def rows_from_extract(data: dict[str, Any]) -> Rows:
    """Rows of an ingest extract (``extract_tabular_db_data`` output)."""
    return normalize_rows(
        data.get("tables"), data.get("columns"), data.get("pks"), data.get("fks")
    )


_CATALOG_TABLES_SQL = """
SELECT s.name AS schema_name, t.name AS table_name, coalesce(t.table_type, '') AS table_type,
       t.pk AS pk, t.id AS table_id
  FROM catalog_database d
  JOIN catalog_schema s ON s.database_id = d.id
  JOIN catalog_table t ON t.schema_id = s.id
 WHERE d.name = :db
"""

_CATALOG_COLUMNS_SQL = """
SELECT s.name AS schema_name, t.name AS table_name, c.name AS column_name,
       coalesce(c.data_type, '') AS data_type, c.is_nullable AS is_nullable, c.id AS column_id
  FROM catalog_database d
  JOIN catalog_schema s ON s.database_id = d.id
  JOIN catalog_table t ON t.schema_id = s.id
  JOIN catalog_column c ON c.table_id = t.id
 WHERE d.name = :db
"""


def catalog_rows(database_name: str) -> tuple[Rows, dict[tuple, str], dict[tuple, str]]:
    """The AO catalog for *database_name* as rows, plus table/column id maps.

    FKs are not compared (AO skips FKs whose target it did not map), so ``fks`` is
    left empty here and verification compares tables, columns and PKs.
    """
    tables = store().query_read(text(_CATALOG_TABLES_SQL), {"db": database_name})
    columns = store().query_read(text(_CATALOG_COLUMNS_SQL), {"db": database_name})
    table_ids = {(t["schema_name"], t["table_name"]): t["table_id"] for t in tables}
    column_ids = {
        (c["schema_name"], c["table_name"], c["column_name"]): c["column_id"]
        for c in columns
    }
    pk_rows = []
    for t in tables:
        for position, column in enumerate(t["pk"] or [], start=1):
            pk_rows.append((t["schema_name"], t["table_name"], position, column))
    rows: Rows = {
        "tables": sorted(
            {
                (t["schema_name"], t["table_name"], str(t["table_type"]).lower())
                for t in tables
            }
        ),
        "columns": sorted(
            {
                (
                    c["schema_name"],
                    c["table_name"],
                    c["column_name"],
                    str(c["data_type"]).lower(),
                    _truthy(c["is_nullable"]),
                )
                for c in columns
            }
        ),
        "pks": sorted(set(pk_rows)),
        "fks": [],
    }
    return rows, table_ids, column_ids


def diff(old: Rows, new: Rows) -> dict[str, Any]:
    """Table/column level difference between two row sets."""
    old_tables = {(r[0], r[1]) for r in old.get("tables", [])}
    new_tables = {(r[0], r[1]) for r in new.get("tables", [])}
    old_cols = {(r[0], r[1], r[2]): r for r in old.get("columns", [])}
    new_cols = {(r[0], r[1], r[2]): r for r in new.get("columns", [])}
    added_tables = sorted(new_tables - old_tables)
    removed_tables = sorted(old_tables - new_tables)
    kept_tables = old_tables & new_tables
    added_columns = sorted(
        k for k in new_cols.keys() - old_cols.keys() if (k[0], k[1]) in kept_tables
    )
    removed_columns = sorted(
        k for k in old_cols.keys() - new_cols.keys() if (k[0], k[1]) in kept_tables
    )
    changed_columns = sorted(
        k
        for k in old_cols.keys() & new_cols.keys()
        if old_cols[k][3:] != new_cols[k][3:]
    )
    old_pk = {(r[0], r[1]): [] for r in old.get("pks", [])}
    for r in sorted(old.get("pks", [])):
        old_pk[(r[0], r[1])].append(r[3])
    new_pk = {(r[0], r[1]): [] for r in new.get("pks", [])}
    for r in sorted(new.get("pks", [])):
        new_pk[(r[0], r[1])].append(r[3])
    changed_pk_tables = sorted(
        t for t in kept_tables if old_pk.get(t, []) != new_pk.get(t, [])
    )
    changed_tables = sorted(
        {(k[0], k[1]) for k in added_columns + removed_columns + changed_columns}
        | set(changed_pk_tables)
    )
    return {
        "added_tables": added_tables,
        "removed_tables": removed_tables,
        "added_columns": added_columns,
        "removed_columns": removed_columns,
        "changed_columns": changed_columns,
        "changed_tables": changed_tables,
        "total_tables_before": len(old_tables),
    }


def auth_signal(
    connector: Any, schemas: list[str] | None = None
) -> dict[str, Any] | None:
    """Account and permission signal for delete judgement (MSSQL only, plan 7.3).

    ``OBJECT_ID`` cannot tell "dropped" from "not permitted" (both NULL), so
    instead the account and its DB/schema permissions are recorded and compared
    between runs.
    """
    if (getattr(connector, "dialect", "") or "").lower() != "tsql":
        return None
    try:
        frame = connector.execute(
            "SELECT SUSER_SNAME() AS login_name, USER_NAME() AS user_name, "
            "IS_SRVROLEMEMBER('sysadmin') AS is_sysadmin, "
            "HAS_PERMS_BY_NAME(DB_NAME(), 'DATABASE', 'VIEW DEFINITION') AS db_view_definition"
        )
        signal: dict[str, Any] = {
            k: (
                None
                if pd.isna(v)
                else (int(v) if isinstance(v, (int, float)) else str(v))
            )
            for k, v in frame.iloc[0].to_dict().items()
        }
        for schema in sorted(set(schemas or [])):
            quoted = schema.replace("'", "''")
            perms = connector.execute(
                f"SELECT HAS_PERMS_BY_NAME('{quoted}', 'SCHEMA', 'VIEW DEFINITION') AS vd, "
                f"HAS_PERMS_BY_NAME('{quoted}', 'SCHEMA', 'SELECT') AS sel"
            ).iloc[0]
            signal[f"schema:{schema}"] = [
                None if pd.isna(perms["vd"]) else int(perms["vd"]),
                None if pd.isna(perms["sel"]) else int(perms["sel"]),
            ]
        return signal
    except Exception:  # noqa: BLE001 - absence of a signal never blocks
        return None
