# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Store fingerprint: changes exactly when generated SQL may change."""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import func, update  # noqa: E402

from auto_ontology.dal import fingerprint as fp  # noqa: E402
from auto_ontology.dal import schema as s  # noqa: E402
from auto_ontology.dal.session import store  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def require_schema():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read("SELECT 1 FROM term LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"auto_ontology schema unavailable (alembic upgrade head): {exc}")


def _add(table, **values) -> str:
    return store().query_write(table.insert().values(**values).returning(table.c.id))[
        0
    ]["id"]


@pytest.fixture
def world():
    db_name = f"fp_{uuid.uuid4().hex[:8]}"
    database = _add(s.catalog_database, name=db_name)
    schema = _add(s.catalog_schema, database_id=database, name="dbo")
    table = _add(s.catalog_table, schema_id=schema, name="orders")
    column = _add(
        s.catalog_column,
        table_id=table,
        name="amount",
        ordinal_position=1,
        data_type="int",
        is_nullable=True,
    )
    yield {"db": db_name, "database": database, "table": table, "column": column}
    store().query_write(
        s.catalog_database.delete().where(s.catalog_database.c.id == database)
    )


def test_stable_when_nothing_changes(world):
    a = fp.compute_fingerprint(world["db"])
    b = fp.compute_fingerprint(world["db"])
    assert (a["content_hash"], a["config_hash"], a["structure_hash"]) == (
        b["content_hash"],
        b["config_hash"],
        b["structure_hash"],
    )
    assert a["structure_rows"] == 2  # one column row + one table row


def test_ingest_timestamp_alone_does_not_change_anything(world):
    before = fp.compute_fingerprint(world["db"])
    store().query_write(
        update(s.catalog_database)
        .where(s.catalog_database.c.id == world["database"])
        .values(pulled=func.now())
    )
    after = fp.compute_fingerprint(world["db"])
    assert before["content_hash"] == after["content_hash"]
    assert before["structure_hash"] == after["structure_hash"]


def test_column_type_change_moves_structure_and_content(world):
    before = fp.compute_fingerprint(world["db"])
    store().query_write(
        update(s.catalog_column)
        .where(s.catalog_column.c.id == world["column"])
        .values(data_type="bigint")
    )
    after = fp.compute_fingerprint(world["db"])
    assert before["structure_hash"] != after["structure_hash"]
    assert before["content_hash"] != after["content_hash"]


def test_description_edit_moves_content_not_structure(world):
    before = fp.compute_fingerprint(world["db"])
    store().query_write(
        update(s.catalog_table)
        .where(s.catalog_table.c.id == world["table"])
        .values(description="주문 금액 테이블")
    )
    after = fp.compute_fingerprint(world["db"])
    assert before["content_hash"] != after["content_hash"]
    assert before["structure_hash"] == after["structure_hash"]


def test_semantic_object_add_and_delete_move_content(world):
    before = fp.compute_fingerprint(world["db"])
    term = _add(s.term, name=f"매출-{uuid.uuid4().hex[:6]}", description="x")
    added = fp.compute_fingerprint(world["db"])
    store().query_write(s.term.delete().where(s.term.c.id == term))
    removed = fp.compute_fingerprint(world["db"])
    assert before["content_hash"] != added["content_hash"]
    assert removed["content_hash"] == before["content_hash"]


def test_other_database_structure_is_ignored(world):
    before = fp.compute_fingerprint(world["db"])
    other = _add(s.catalog_database, name=f"fp_other_{uuid.uuid4().hex[:6]}")
    try:
        after = fp.compute_fingerprint(world["db"])
        assert before["structure_hash"] == after["structure_hash"]
    finally:
        store().query_write(
            s.catalog_database.delete().where(s.catalog_database.c.id == other)
        )


def test_config_env_hash_ignores_secrets():
    base = {"REASONING_MODEL": "openai/gpt-4o", "PATH": "/bin"}
    with_secret = {**base, "REASONING_API_KEY": "sk-1", "SQL_PASSWORD": "p"}
    assert fp.config_env_hash(base) == fp.config_env_hash(with_secret)
    assert fp.config_env_hash(base) != fp.config_env_hash(
        {**base, "REASONING_MODEL": "openai/gpt-5"}
    )
    assert fp.config_env_hash(base) == fp.config_env_hash({**base, "UNRELATED": "x"})


def test_version_parts():
    v = fp.fingerprint_version("abc", "c1", "h1", None)
    assert v != fp.fingerprint_version("abd", "c1", "h1", None)
    assert v != fp.fingerprint_version("abc", "c1", "h1", 3)
    assert len(v) == 16
