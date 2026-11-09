# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks a statement must pass before the SQL execution API runs it.

* exactly one statement, and it is a query (``SELECT`` / ``WITH`` / set ops);
* nothing inside it writes or runs other code (DML, DDL, ``SELECT ... INTO``,
  ``EXEC``, ``OPENROWSET``/``OPENQUERY``...);
* every table it reads is in the catalog of the target database.

The checks work on the sqlglot AST rather than keywords, so harmless SQL such
as ``REPLACE(name, ' ', '')`` is not rejected for containing a verb.
"""

from __future__ import annotations

from collections.abc import Iterable

import sqlglot
from sqlglot import exp


class SqlCheckError(ValueError):
    """The statement must not be executed; the message says why."""


# Node types that must not appear anywhere in the statement.
_FORBIDDEN_NODE_NAMES = (
    "Insert",
    "Update",
    "Delete",
    "Merge",
    "Create",
    "Drop",
    "Alter",
    "AlterTable",
    "TruncateTable",
    "Command",
    "Into",
    "Use",
    "Set",
    "Grant",
    "Revoke",
    "Transaction",
    "Commit",
    "Rollback",
)
_FORBIDDEN_NODES = tuple(
    node for name in _FORBIDDEN_NODE_NAMES if (node := getattr(exp, name, None))
)

# Table-valued functions that reach outside the connected database.
_FORBIDDEN_FUNCTIONS = {"OPENROWSET", "OPENQUERY", "OPENDATASOURCE", "OPENXML"}


def parse_single_query(sql: str, dialect: str | None) -> exp.Expression:
    """Parse *sql* as exactly one read-only query, or raise :class:`SqlCheckError`."""
    try:
        statements = [
            s for s in sqlglot.parse(sql, read=dialect or None) if s is not None
        ]
    except sqlglot.errors.ParseError as exc:
        raise SqlCheckError(f"SQL could not be parsed: {exc}") from exc

    if len(statements) != 1:
        raise SqlCheckError("Exactly one SQL statement is allowed.")
    tree = statements[0]

    if not isinstance(tree, exp.Query):
        raise SqlCheckError("Only SELECT queries are allowed.")

    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise SqlCheckError(
                f"Statement contains a disallowed operation: {type(node).__name__}."
            )
        if isinstance(node, exp.Func):
            name = (
                node.name if isinstance(node, exp.Anonymous) else node.sql_name()
            ) or ""
            if name.upper() in _FORBIDDEN_FUNCTIONS:
                raise SqlCheckError(f"Function {name.upper()} is not allowed.")
    return tree


def referenced_tables(tree: exp.Expression) -> list[exp.Table]:
    """Physical tables the query reads (CTE names excluded)."""
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    return [
        table
        for table in tree.find_all(exp.Table)
        if table.name and table.name.lower() not in cte_names
    ]


def check_tables_in_catalog(
    tree: exp.Expression,
    database_name: str,
    catalog_tables: Iterable[str],
) -> None:
    """Raise unless every table *tree* reads is a catalog table of *database_name*."""
    known = {name.lower() for name in catalog_tables if name}
    if not known:
        raise SqlCheckError(
            f"No catalog tables found for database {database_name!r}; ingest it first."
        )

    tables = referenced_tables(tree)
    if not tables:
        raise SqlCheckError("The query does not read any table.")

    unknown: list[str] = []
    for table in tables:
        catalog = table.catalog
        if catalog and catalog.lower() != database_name.lower():
            raise SqlCheckError(
                f"Table {table.sql()} is in another database; only "
                f"{database_name!r} can be queried."
            )
        if table.name.lower() not in known:
            unknown.append(table.name)
    if unknown:
        raise SqlCheckError(
            "Tables not in the catalog: " + ", ".join(sorted(set(unknown)))
        )


def check_sql(
    sql: str,
    *,
    dialect: str | None,
    database_name: str,
    catalog_tables: Iterable[str],
) -> exp.Expression:
    """Run every check; return the parsed query."""
    tree = parse_single_query(sql, dialect)
    check_tables_in_catalog(tree, database_name, catalog_tables)
    return tree
