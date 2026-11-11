# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks a statement must pass before the SQL execution API runs it.

* exactly one statement, and it is a query (``SELECT`` / ``WITH`` / set ops);
* nothing inside it writes or runs other code (DML, DDL, ``SELECT ... INTO``,
  ``EXEC``, ``OPENROWSET``/``OPENQUERY``...);
* every function is one sqlglot recognises (aggregates, string, date, math,
  window...) or a known-harmless built-in it leaves unparsed; any other name,
  including user-defined and external-access functions, is rejected;
* every table it reads is in the catalog of the target database, compared by
  schema and name (unqualified names resolve to the connector's default schema).

These checks are a second line of defence: the database login should also be
read-only (``db_datareader``) so the server itself refuses writes.

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

# Built-ins sqlglot parses as ``exp.Anonymous`` (no typed node) that only compute
# a value from their arguments. Any other anonymous call is rejected, so external
# access (OPENROWSET, OPENQUERY, OPENDATASOURCE...), server/metadata probes
# (HOST_NAME, FILEPROPERTY...) and user-defined functions never run.
_ALLOWED_ANONYMOUS_FUNCTIONS = {
    # string
    "STR",
    "PATINDEX",
    "QUOTENAME",
    "NCHAR",
    "SOUNDEX",
    "DIFFERENCE",
    "TRANSLATE",
    "FORMATMESSAGE",
    "STRING_ESCAPE",
    # date / time
    "ISDATE",
    "GETUTCDATE",
    "SYSUTCDATETIME",
    "SYSDATETIMEOFFSET",
    "DATETIMEFROMPARTS",
    "DATETIME2FROMPARTS",
    "SMALLDATETIMEFROMPARTS",
    "TIMEFROMPARTS",
    "DATEDIFF_BIG",
    "SWITCHOFFSET",
    "TODATETIMEOFFSET",
    # conversion / logic
    "CHOOSE",
    "ISNUMERIC",
    "TRY_PARSE",
    "PARSE",
    # math
    "SQUARE",
    "PI",
    "LOG10",
    "ATN2",
    "COT",
    "DEGREES",
    "RADIANS",
    # aggregates / window / grouping
    "VAR",
    "VARP",
    "STDEVP",
    "GROUPING",
    "GROUPING_ID",
    "CHECKSUM_AGG",
    "APPROX_COUNT_DISTINCT",
    "CUME_DIST",
    "PERCENTILE_CONT",
    "PERCENTILE_DISC",
}


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
        if isinstance(node, exp.Anonymous):
            name = str(node.name or "").upper()
            if name not in _ALLOWED_ANONYMOUS_FUNCTIONS:
                raise SqlCheckError(f"Function {name or node.sql()} is not allowed.")
        elif isinstance(node, exp.Dot) and isinstance(node.expression, exp.Func):
            # schema.fn(...): a user-defined function, never a built-in
            raise SqlCheckError(f"Function {node.sql()} is not allowed.")
    return tree


# Schema an unqualified table name resolves to, per dialect.
_DEFAULT_SCHEMAS = {"tsql": "dbo", "postgres": "public"}


def default_schema_for(dialect: str | None) -> str | None:
    """The schema an unqualified name means for *dialect* (``None`` if unknown)."""
    return _DEFAULT_SCHEMAS.get((dialect or "").lower())


def referenced_tables(tree: exp.Expression) -> list[exp.Table]:
    """Physical tables the query reads.

    A table node is dropped only when it really refers to a CTE: it has no
    schema/database qualifier and a CTE of that name is visible in its scope.
    Qualified names and tables read inside CTE bodies are always kept, so a CTE
    named like a real table cannot hide that table from the catalog check.
    """
    from sqlglot.errors import OptimizeError
    from sqlglot.optimizer.scope import traverse_scope

    try:
        scopes = traverse_scope(tree)
    except OptimizeError as exc:
        raise SqlCheckError(f"SQL could not be analysed: {exc}") from exc

    tables: list[exp.Table] = []
    seen: set[int] = set()
    for scope in scopes:
        cte_names = {name.lower() for name in scope.cte_sources}
        for table in scope.tables:
            if id(table) in seen or not table.name:
                continue
            seen.add(id(table))
            is_cte_ref = (
                not table.db
                and not table.catalog
                and (
                    table.name.lower() in cte_names
                    or table.name.lower() in _enclosing_cte_names(table)
                )
            )
            if not is_cte_ref:
                tables.append(table)
    # A FROM/JOIN table no scope claimed (should not happen) is kept as physical:
    # failing closed beats letting an unscoped name skip the catalog check.
    for table in tree.find_all(exp.Table):
        if (
            table.name
            and id(table) not in seen
            and isinstance(table.parent, (exp.From, exp.Join))
        ):
            tables.append(table)
    return tables


def _enclosing_cte_names(node: exp.Expression) -> set[str]:
    """Names of the CTEs whose body contains *node* (T-SQL recursive self-reference)."""
    names: set[str] = set()
    parent = node.parent
    while parent is not None:
        if isinstance(parent, exp.CTE):
            names.add(parent.alias_or_name.lower())
        parent = parent.parent
    return names


CatalogTable = tuple[str, str]
"""``(schema, table)`` of a catalog table."""


def check_tables_in_catalog(
    tree: exp.Expression,
    database_name: str,
    catalog_tables: Iterable[CatalogTable],
    default_schema: str | None = None,
) -> None:
    """Raise unless every table *tree* reads is a catalog table of *database_name*.

    Tables are compared as ``(schema, name)``. An unqualified name resolves to
    *default_schema*; when that is unknown it must name exactly one catalog
    table, so it cannot pick up a same-named table in another schema.
    """
    known = {(schema.lower(), name.lower()) for schema, name in catalog_tables if name}
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
        name = table.name.lower()
        schema = (table.db or default_schema or "").lower()
        if schema:
            ok = (schema, name) in known
        else:
            ok = sum(1 for _, n in known if n == name) == 1
        if not ok:
            unknown.append(table.sql())
    if unknown:
        raise SqlCheckError(
            "Tables not in the catalog: " + ", ".join(sorted(set(unknown)))
        )


def check_sql(
    sql: str,
    *,
    dialect: str | None,
    database_name: str,
    catalog_tables: Iterable[CatalogTable],
) -> exp.Expression:
    """Run every check; return the parsed query."""
    tree = parse_single_query(sql, dialect)
    check_tables_in_catalog(
        tree, database_name, catalog_tables, default_schema_for(dialect)
    )
    return tree
