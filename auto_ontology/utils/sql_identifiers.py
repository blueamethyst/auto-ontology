# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dialect-aware quoting of SQL identifiers.

Shared by the connectors (which build fully-qualified references for metadata
queries) and by the semantic layer's profiling probes, so both quote names the
same way.
"""

from __future__ import annotations

from sqlglot import exp


def quoted_identifier(name: str, dialect: str | None = None) -> str:
    """Quote a catalog, schema, table or column name for *dialect*.

    Names carrying a space or a reserved word have to be quoted or the probe
    silently loses the table: ``SELECT * FROM main.Sales Orders`` parses as
    table ``main.Sales``, raises "no such table", and the caller drops every
    column of that table from profiling. The quote character is dialect-specific
    (backticks on MySQL and Spark, double quotes elsewhere), so the naive
    hard-coded ``"`` would trade a SQLite bug for a MySQL one.
    """
    try:
        return exp.to_identifier(name, quoted=True).sql(dialect=dialect or None)
    except Exception:
        # Unknown dialect: fall back to the SQL-standard quote rather than
        # emitting a bare identifier, since bare is what breaks on spaces.
        return '"' + name.replace('"', '""') + '"'


def limited_select(
    columns: str,
    rest: str,
    limit: int,
    dialect: str | None = None,
    *,
    distinct: bool = False,
) -> str:
    """Build ``SELECT [DISTINCT] <columns> <rest>`` capped at *limit* rows.

    T-SQL has no ``LIMIT``: a probe written with it fails to parse on SQL
    Server, so profiling silently returns nothing there. SQL Server takes
    ``TOP (n)`` after ``DISTINCT``; every other supported engine takes a
    trailing ``LIMIT n``.
    """
    keyword = "SELECT DISTINCT" if distinct else "SELECT"
    if dialect == "tsql":
        return f"{keyword} TOP ({int(limit)}) {columns} {rest}"
    return f"{keyword} {columns} {rest} LIMIT {int(limit)}"


def qualified_name(*parts: str | None, dialect: str | None = None) -> str:
    """Join *parts* into a dotted, quoted reference, skipping empty parts.

    ``qualified_name("nvapp", "raw", "t", dialect="spark")`` ->
    ```nvapp`.`raw`.`t```.
    """
    return ".".join(quoted_identifier(p, dialect) for p in parts if p)
