# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dialect-aware SQL identifier quoting."""

from __future__ import annotations

import pytest

from auto_ontology.utils.sql_identifiers import (
    limited_select,
    qualified_name,
    quoted_identifier,
)


@pytest.mark.parametrize(
    ("dialect", "expected"),
    [
        ("sqlite", '"Sales Orders"'),
        ("postgres", '"Sales Orders"'),
        ("mysql", "`Sales Orders`"),
        ("spark", "`Sales Orders`"),
        (None, '"Sales Orders"'),
        ("not-a-dialect", '"Sales Orders"'),
    ],
)
def test_quoted_identifier_per_dialect(dialect: str | None, expected: str) -> None:
    # Backticks on MySQL/Spark, double quotes elsewhere; an unknown dialect
    # still quotes, since a bare identifier is what loses the table.
    assert quoted_identifier("Sales Orders", dialect) == expected


def test_qualified_name_joins_parts() -> None:
    assert qualified_name("nvapp", "raw", "t", dialect="spark") == "`nvapp`.`raw`.`t`"


def test_qualified_name_skips_empty_parts() -> None:
    # A two-level engine passes no catalog; an unqualified table passes no
    # schema. Neither may leave a stray leading dot.
    assert qualified_name(None, "raw", "t", dialect="spark") == "`raw`.`t`"
    assert qualified_name(None, None, "t", dialect="spark") == "`t`"
    assert qualified_name("", "raw", "t", dialect="spark") == "`raw`.`t`"


def test_limited_select_uses_top_on_tsql() -> None:
    # SQL Server has no LIMIT; a probe written with it fails to parse there.
    assert (
        limited_select("*", "FROM [dbo].[t]", 1000, "tsql")
        == "SELECT TOP (1000) * FROM [dbo].[t]"
    )
    assert (
        limited_select(
            "[a]", "FROM [t] WHERE [a] IS NOT NULL", 21, "tsql", distinct=True
        )
        == "SELECT DISTINCT TOP (21) [a] FROM [t] WHERE [a] IS NOT NULL"
    )


def test_limited_select_appends_limit_elsewhere() -> None:
    assert limited_select("*", 'FROM "t"', 5, "postgres") == 'SELECT * FROM "t" LIMIT 5'
    assert (
        limited_select('"a"', 'FROM "t"', 2, None, distinct=True)
        == 'SELECT DISTINCT "a" FROM "t" LIMIT 2'
    )
