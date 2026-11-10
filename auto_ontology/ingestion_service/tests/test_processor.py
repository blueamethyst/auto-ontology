# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema ingest handler and source-schema watch (against a real store).

Embedding, compilation, PII and rules are stubbed; the catalog write, its
verification, deletion judgement, semantic cleanup and the persisted state are
real, because those are what this handler is for.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import delete, insert, select, text  # noqa: E402

from auto_ontology.dal import schema as s  # noqa: E402
from auto_ontology.dal import schema_state as st  # noqa: E402
from auto_ontology.dal.session import store  # noqa: E402
from auto_ontology.ingestion_service import processor  # noqa: E402
from auto_ontology.ingestion_service import schema_signature as sig  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def require_schema():
    if not os.environ.get("POSTGRES_USER"):
        pytest.skip("POSTGRES_* not set")
    try:
        store().query_read("SELECT 1 FROM schema_snapshot LIMIT 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"schema ingest tables unavailable (alembic upgrade head): {exc}")


class FakeConnector:
    dialect = "postgres"
    table_filter = None

    def __init__(self, name: str) -> None:
        self.database_name = name
        self.tables: dict[str, list[tuple[str, str, bool]]] = {}
        self.pks: dict[str, list[str]] = {}

    def add(
        self, table: str, *columns: tuple[str, str, bool], pk: list[str] | None = None
    ) -> None:
        self.tables[table] = list(columns)
        if pk:
            self.pks[table] = pk

    def get_tables(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"table_schema": "dbo", "table_name": t, "table_type": "BASE TABLE"}
                for t in self.tables
            ],
            columns=["table_schema", "table_name", "table_type"],
        )

    def get_columns(self) -> pd.DataFrame:
        rows = [
            {
                "table_schema": "dbo",
                "table_name": t,
                "column_name": c,
                "data_type": ty,
                "is_nullable": "YES" if nullable else "NO",
                "ordinal_position": i + 1,
            }
            for t, cols in self.tables.items()
            for i, (c, ty, nullable) in enumerate(cols)
        ]
        return pd.DataFrame(
            rows,
            columns=[
                "table_schema",
                "table_name",
                "column_name",
                "data_type",
                "is_nullable",
                "ordinal_position",
            ],
        )

    def get_views(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["table_schema", "table_name", "view_definition"])

    def get_queries(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["end_time", "query_text"])

    def get_pks(self) -> pd.DataFrame:
        rows = [
            {
                "table_schema": "dbo",
                "table_name": t,
                "column_name": c,
                "ordinal_position": i + 1,
            }
            for t, cols in self.pks.items()
            for i, c in enumerate(cols)
        ]
        return pd.DataFrame(
            rows,
            columns=["table_schema", "table_name", "column_name", "ordinal_position"],
        )

    def get_fks(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "table_schema",
                "table_name",
                "column_name",
                "referenced_schema",
                "referenced_table",
                "referenced_column",
            ]
        )


class FakeVdb:
    def __init__(self) -> None:
        self.written = 0

    def run(self, records: list) -> int:
        n = sum(len(r) for r in records)
        self.written += n
        return n

    def delete_by_database(self, name: str) -> list[str]:
        return []

    def delete_by_id(self, node_id: str) -> int:
        return 1


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch):
    import auto_ontology.infra.feature_flags as flags
    import auto_ontology.ingestion_service.config as config
    import auto_ontology.ingestion_service.pii as pii
    import auto_ontology.ingestion_service.rules as rules
    import auto_ontology.utils as utils
    import auto_ontology.utils.embedding as embedding
    import auto_ontology.vdb as vdb
    import auto_ontology.vdb.records as records

    state: dict[str, Any] = {"lose": 0, "vdb": FakeVdb()}

    def chunks(rows: Any, params: Any, label: str = "") -> Any:
        yield rows

    def to_records(chunk: Any) -> list:
        n = len(chunk) - state["lose"]
        return [[{"id": i} for i in range(max(n, 0))]] if n > 0 else []

    monkeypatch.setattr(embedding, "batch_embed_chunks", chunks)
    monkeypatch.setattr(records, "to_vdb_records", to_records)
    monkeypatch.setattr(vdb, "get_data_vdb", lambda **_: state["vdb"])
    monkeypatch.setattr(vdb, "get_semantic_vdb", lambda **_: state["vdb"])
    monkeypatch.setattr(utils, "get_embed_params", lambda: {})
    monkeypatch.setattr(flags, "is_pii_detection_enabled", lambda: False)
    monkeypatch.setattr(config, "is_semantic_compilation_enabled", lambda: False)
    monkeypatch.setattr(pii, "run_pii_propagation", lambda label: None)
    monkeypatch.setattr(rules, "replay_rules", lambda label: None)
    monkeypatch.setattr(
        sig, "auth_signal", lambda connector, schemas=None: state.get("auth")
    )
    created: list[str] = []
    state["created"] = created

    def make(prefix: str = "proc") -> FakeConnector:
        name = f"{prefix}_{uuid.uuid4().hex[:8]}"
        created.append(name)
        return FakeConnector(name)

    state["make"] = make
    yield state
    for name in created:
        for table in (
            s.schema_snapshot,
            s.schema_open_failure,
            s.schema_change_log,
            s.schema_ingest_run,
        ):
            store().query_write(delete(table).where(table.c.database_name == name))
        store().query_write(
            delete(s.catalog_database).where(s.catalog_database.c.name == name)
        )


def _connector(env: dict, tables: int = 3) -> FakeConnector:
    c = env["make"]()
    for i in range(tables):
        c.add(f"t{i}", ("id", "int", False), ("amount", "int", True), pk=["id"])
    return c


def _changes(name: str) -> list[tuple[str, str]]:
    return [(r["kind"], r["object"]) for r in st.recent_changes(name)]


def test_first_run_writes_verifies_and_publishes(env):
    c = _connector(env)
    report = processor.process_database(c, "new")
    assert report.result == "succeeded", report.failures
    assert report.stages["catalog"]["problems"] == []
    assert (
        report.stages["embedding"]["expected"]
        == report.stages["embedding"]["written"]
        > 0
    )
    snapshot = st.get_snapshot(c.database_name)
    assert snapshot["applied_signature"] == sig.signature(sig.observe(c))
    assert snapshot["generation"] == 1
    assert st.current_state(c.database_name) == ("ready", 1)
    assert ("table_added", "dbo.t0") in _changes(c.database_name)


def test_unchanged_rerun_keeps_generation(env):
    c = _connector(env)
    processor.process_database(c, "new")
    report = processor.process_database(c, "force")
    assert report.result == "succeeded"
    assert report.generation is None
    assert st.get_snapshot(c.database_name)["generation"] == 1


def test_added_column_changes_generation_and_logs(env):
    c = _connector(env)
    processor.process_database(c, "new")
    c.tables["t1"].append(("note", "nvarchar", True))
    report = processor.process_database(c, "change")
    assert report.result == "succeeded"
    assert report.generation == 2
    assert ("column_added", "dbo.t1.note") in _changes(c.database_name)
    catalog, _, _ = sig.catalog_rows(c.database_name)
    assert ("dbo", "t1", "note", "nvarchar", True) in catalog["columns"]


def _add_semantics(
    name: str, table: str, column: str, *, human: bool
) -> dict[str, str]:
    _, table_ids, column_ids = sig.catalog_rows(name)
    tid = table_ids[("dbo", table)]
    cid = column_ids[("dbo", table, column)]
    term = store().query_write(
        insert(s.term)
        .values(
            name=f"{table}-{uuid.uuid4().hex[:6]}",
            source="user" if human else "semantic",
        )
        .returning(s.term.c.id)
    )[0]["id"]
    store().query_write(insert(s.table__term).values(table_id=tid, term_id=term))
    attr = store().query_write(
        insert(s.column_attribute)
        .values(
            name=f"{column}-attr",
            source="semantic",
            source_column=column,
            term_name="x",
            table_id=tid,
        )
        .returning(s.column_attribute.c.id)
    )[0]["id"]
    store().query_write(
        insert(s.column__has_attribute).values(column_id=cid, attribute_id=attr)
    )
    return {"term": term, "attr": attr, "table_id": tid, "column_id": cid}


def test_dropped_table_cleans_generated_semantics_and_keeps_human_ones(env):
    c = env["make"]()
    for i in range(12):  # 1 of 12 < 10% bulk threshold
        c.add(f"t{i}", ("id", "int", False), pk=["id"])
    processor.process_database(c, "new")
    auto = _add_semantics(c.database_name, "t0", "id", human=False)
    human = _add_semantics(c.database_name, "t1", "id", human=True)
    # a live column whose semantic FK points at t0's (generated) attribute
    _, _, column_ids = sig.catalog_rows(c.database_name)
    live = column_ids[("dbo", "t2", "id")]
    store().query_write(
        insert(s.column__semantic_fk).values(column_id=live, attribute_id=auto["attr"])
    )

    del c.tables["t0"], c.pks["t0"]
    report = processor.process_database(c, "change")
    assert report.result == "succeeded", report.failures
    assert not store().query_read(select(s.term).where(s.term.c.id == auto["term"]))
    assert not store().query_read(
        select(s.column_attribute).where(s.column_attribute.c.id == auto["attr"])
    )
    assert not store().query_read(
        select(s.column__semantic_fk).where(s.column__semantic_fk.c.column_id == live)
    )
    changes = _changes(c.database_name)
    assert ("table_removed", "dbo.t0") in changes
    assert any(k == "semantic_fk_unlinked" for k, _ in changes)

    del c.tables["t1"], c.pks["t1"]
    processor.process_database(c, "change")
    assert store().query_read(
        select(s.term).where(s.term.c.id == human["term"])
    )  # human term kept
    assert any(
        k == "semantic_orphaned" and o.startswith("term:")
        for k, o in _changes(c.database_name)
    )
    assert st.orphan_count(c.database_name) >= 1


def test_bulk_delete_is_held_then_applied_after_delay(
    env, monkeypatch: pytest.MonkeyPatch
):
    c = _connector(env, tables=4)
    processor.process_database(c, "new")
    del c.tables["t0"], c.tables["t1"], c.pks["t0"], c.pks["t1"]  # 50%
    c.add("t9", ("id", "int", False))  # an addition still lands
    held = processor.process_database(c, "change")
    assert held.held == "bulk_delete"
    catalog, _, _ = sig.catalog_rows(c.database_name)
    names = {r[1] for r in catalog["tables"]}
    assert {"t0", "t1", "t9"} <= names
    snapshot = st.get_snapshot(c.database_name)
    assert snapshot["pending_reason"] == "bulk_delete"

    # still inside the delay: still held
    assert processor.process_database(c, "change").held == "bulk_delete"
    # after the delay with the same result: applied
    store().query_write(
        s.schema_snapshot.update()
        .where(s.schema_snapshot.c.database_name == c.database_name)
        .values(pending_since=datetime.now(timezone.utc) - timedelta(hours=2))
    )
    applied = processor.process_database(c, "change")
    assert applied.held is None and applied.result == "succeeded"
    catalog, _, _ = sig.catalog_rows(c.database_name)
    assert {r[1] for r in catalog["tables"]} == {"t2", "t3", "t9"}
    assert st.get_snapshot(c.database_name)["pending_reason"] is None


def test_permission_signal_change_holds_deletions_until_it_repeats(env):
    env["auth"] = {"login_name": "sasa", "db_view_definition": 1}
    c = env["make"]()
    for i in range(12):
        c.add(f"t{i}", ("id", "int", False))
    processor.process_database(c, "new")
    env["auth"] = {"login_name": "reader", "db_view_definition": 1}
    del c.tables["t0"]
    first = processor.process_database(c, "change")
    assert first.held == "auth_signal_changed"
    assert ("dbo", "t0") in {
        (r[0], r[1]) for r in sig.catalog_rows(c.database_name)[0]["tables"]
    }
    second = processor.process_database(c, "change")  # same new signal again
    assert second.held is None
    assert ("dbo", "t0") not in {
        (r[0], r[1]) for r in sig.catalog_rows(c.database_name)[0]["tables"]
    }


def test_partial_embedding_is_not_published_and_retries(env):
    c = _connector(env)
    processor.process_database(c, "new")
    c.add("t5", ("id", "int", False))
    env["lose"] = 1  # one row comes back without an embedding
    report = processor.process_database(c, "change")
    assert report.result == "partial"
    assert any(f["stage"] == "embedding" for f in report.failures)
    assert st.current_state(c.database_name)[0] == "failed"
    assert st.get_snapshot(c.database_name)["generation"] == 1  # not published
    env["lose"] = 0
    retry = processor.process_database(c, "retry")
    assert retry.result == "succeeded"
    assert st.open_failures(c.database_name) == []
    assert st.current_state(c.database_name) == ("ready", 2)


def test_empty_extract_is_held_then_empties_catalog(env):
    c = _connector(env, tables=2)
    processor.process_database(c, "new")
    c.tables.clear()
    c.pks.clear()
    assert processor.process_database(c, "change").held == "bulk_delete"
    assert sig.catalog_rows(c.database_name)[0]["tables"]
    store().query_write(
        s.schema_snapshot.update()
        .where(s.schema_snapshot.c.database_name == c.database_name)
        .values(pending_since=datetime.now(timezone.utc) - timedelta(hours=2))
    )
    report = processor.process_database(c, "change")
    assert report.stages["catalog"] == {"emptied": True}
    assert not sig.catalog_rows(c.database_name)[0]["tables"]


def test_stale_run_is_reaped_into_an_open_failure(env):
    c = _connector(env)
    run_id = st.start_run(c.database_name, "scheduled")
    store().query_write(
        s.schema_ingest_run.update()
        .where(s.schema_ingest_run.c.id == run_id)
        .values(heartbeat_at=datetime.now(timezone.utc) - timedelta(minutes=10))
    )
    reaped = st.reap_stale_runs()
    assert run_id in {r["id"] for r in reaped}
    assert any(f["stage"] == "run" for f in st.open_failures(c.database_name))


def test_write_lock_is_exclusive():
    from auto_ontology.ingestion_service.write_lock import (
        LockBusy,
        lock_held_elsewhere,
        write_lock,
    )

    with write_lock():
        with pytest.raises(LockBusy):
            with write_lock(wait=False):
                pass
        assert lock_held_elsewhere()
    assert not lock_held_elsewhere()


# ------------------------------------------------------------ watch pass


@pytest.fixture
def watcher(env, monkeypatch: pytest.MonkeyPatch):
    from auto_ontology.ingestion_service import data_scheduler

    calls: list[str] = []
    real = processor.process_database

    def spy(connector: Any, trigger: str, **kw: Any) -> Any:
        calls.append(trigger)
        return real(connector, trigger, **kw)

    monkeypatch.setattr(processor, "process_database", spy)
    monkeypatch.setattr(data_scheduler, "force_due", lambda snapshot, now=None: False)
    return data_scheduler.DataScheduler(), calls


def test_watch_existing_connection_initialises_without_ingest(env, watcher):
    scheduler, calls = watcher
    c = _connector(env)
    processor.process_database(c, "new")  # catalog exists
    store().query_write(
        delete(s.schema_snapshot).where(
            s.schema_snapshot.c.database_name == c.database_name
        )
    )
    scheduler._watch_one(c)
    assert calls == ["new"]  # only the setup call above
    assert st.get_snapshot(c.database_name)["applied_signature"] == sig.signature(
        sig.observe(c)
    )


def test_watch_new_connection_ingests_immediately(env, watcher):
    scheduler, calls = watcher
    c = _connector(env)
    scheduler._watch_one(c)
    assert calls == ["new"]


def test_watch_needs_two_matching_passes(env, watcher, monkeypatch: pytest.MonkeyPatch):
    from auto_ontology.ingestion_service import data_scheduler

    scheduler, calls = watcher
    c = _connector(env)
    scheduler._watch_one(c)  # new
    c.add("t7", ("id", "int", False))
    scheduler._watch_one(c)  # first sighting: pending only
    assert calls == ["new"]
    assert st.get_snapshot(c.database_name)["pending_signature"]
    monkeypatch.setattr(data_scheduler, "min_ingest_gap", lambda: timedelta(0))
    scheduler._watch_one(c)  # confirmed
    assert calls == ["new", "change"]
    scheduler._watch_one(c)  # applied: nothing
    assert calls == ["new", "change"]


def test_watch_change_respects_ingest_gap(env, watcher):
    scheduler, calls = watcher
    c = _connector(env)
    scheduler._watch_one(c)
    st.upsert_snapshot(
        c.database_name, last_change_ingest_at=datetime.now(timezone.utc)
    )
    c.add("t7", ("id", "int", False))
    scheduler._watch_one(c)
    scheduler._watch_one(c)
    assert calls == ["new"]  # within the hour since the last change ingest


def test_watch_open_failure_takes_the_retry_path_only(env, watcher):
    scheduler, calls = watcher
    c = _connector(env)
    scheduler._watch_one(c)
    st.record_failure(c.database_name, "embedding", "", None, "x")
    c.add("t7", ("id", "int", False))
    scheduler._watch_one(c)  # failure just recorded: not due, no signature path
    assert calls == ["new"]
    store().query_write(
        s.schema_open_failure.update()
        .where(s.schema_open_failure.c.database_name == c.database_name)
        .values(last_failed_at=datetime.now(timezone.utc) - timedelta(minutes=11))
    )
    scheduler._watch_one(c)
    assert calls == ["new", "retry"]
    assert st.open_failures(c.database_name) == []


def test_watch_pass_does_not_invalidate_connectors(
    env, monkeypatch: pytest.MonkeyPatch
):
    import asyncio

    from auto_ontology.ingestion_service import data_scheduler

    monkeypatch.setattr(data_scheduler, "watch_enabled", lambda: True)
    monkeypatch.setattr(data_scheduler, "get_connectors", lambda: [])

    def boom() -> None:
        raise AssertionError("watch pass must not invalidate the connector cache")

    monkeypatch.setattr(data_scheduler, "invalidate_connectors_cache", boom)
    asyncio.run(data_scheduler.DataScheduler()._run_once())


def test_force_due_by_date():
    from auto_ontology.ingestion_service.data_scheduler import KST, force_due

    at_0500 = datetime(2026, 11, 10, 5, 0, tzinfo=KST)
    at_0400 = datetime(2026, 11, 10, 4, 0, tzinfo=KST)
    assert force_due({"last_force_date": "2026-11-09"}, at_0500)
    assert not force_due({"last_force_date": "2026-11-10"}, at_0500)
    assert not force_due({"last_force_date": "2026-11-09"}, at_0400)
    assert not force_due(
        {"last_force_date": None}, at_0500
    )  # first run records today instead
    assert not force_due(None, at_0500)


def test_signature_ignores_recreated_identical_table(env):
    c = _connector(env)
    a = sig.signature(sig.observe(c))
    c.tables["t0"] = list(c.tables["t0"])  # same shape, "recreated"
    assert sig.signature(sig.observe(c)) == a
    c.tables["t0"][1] = ("amount", "bigint", True)
    assert sig.signature(sig.observe(c)) != a


def test_protected_table_detection(env):
    c = _connector(env)
    processor.process_database(c, "new")
    auto = _add_semantics(c.database_name, "t0", "id", human=False)
    human = _add_semantics(c.database_name, "t1", "id", human=True)
    assert not processor.protected_table({"id": auto["table_id"]})
    assert processor.protected_table({"id": human["table_id"]})
    store().query_write(
        s.column_attribute.update()
        .where(s.column_attribute.c.id == auto["attr"])
        .values(modified=text("now()"))
    )
    assert processor.protected_table({"id": auto["table_id"]})


def test_changed_tables_are_recompiled_except_protected(
    env, monkeypatch: pytest.MonkeyPatch
):
    import auto_ontology.ingestion_service.config as config
    import auto_ontology.semantic.compile as compile_mod
    import auto_ontology.semantic.pipeline as pipeline

    c = _connector(env)
    processor.process_database(c, "new")
    auto = _add_semantics(c.database_name, "t1", "id", human=False)
    human = _add_semantics(c.database_name, "t2", "id", human=True)

    monkeypatch.setattr(config, "is_semantic_compilation_enabled", lambda: True)
    monkeypatch.setattr(compile_mod, "build_semantic_embedder", lambda *a, **k: None)
    monkeypatch.setattr(
        compile_mod, "run_semantic_compilation", lambda db, failures=None: 0
    )
    seen: dict[str, Any] = {}

    def fake_compile(
        database_name: str,
        *,
        embedder: Any,
        failures: list,
        table_ids: list,
        protect: Any,
    ) -> int:
        seen["table_ids"] = sorted(table_ids)
        seen["kept"] = sorted(t for t in table_ids if not protect({"id": t}))
        failures.append(
            {"stage": "compile_table", "target": auto["table_id"], "error": "boom"}
        )
        return len(seen["kept"])

    monkeypatch.setattr(pipeline, "compile_semantic_layer", fake_compile)
    c.tables["t1"].append(("x", "int", True))
    c.tables["t2"].append(("y", "int", True))
    report = processor.process_database(c, "change")
    assert seen["table_ids"] == sorted([auto["table_id"], human["table_id"]])
    assert seen["kept"] == [auto["table_id"]]  # human-edited table is not recompiled
    assert report.stages["compile"]["protected"] == [human["table_id"]]
    assert report.result == "partial"
    assert [
        f["target"]
        for f in st.open_failures(c.database_name)
        if f["stage"] == "compile_table"
    ] == [auto["table_id"]]

    # next run: the failed table is retried by id even though nothing changed
    def ok_compile(
        database_name: str,
        *,
        embedder: Any,
        failures: list,
        table_ids: list,
        protect: Any,
    ) -> int:
        seen["retry"] = sorted(table_ids)
        return len(table_ids)

    monkeypatch.setattr(pipeline, "compile_semantic_layer", ok_compile)
    retry = processor.process_database(c, "retry")
    assert seen["retry"] == [auto["table_id"]]
    assert retry.result == "succeeded"
    assert st.open_failures(c.database_name) == []


def test_signal_with_new_schema_key_is_not_a_change():
    before = {"login_name": "sasa", "db_view_definition": 1}
    after = {"login_name": "sasa", "db_view_definition": 1, "schema:dbo": [1, 1]}
    assert not processor._signal_changed(before, after)
    assert processor._signal_changed(before, {**after, "login_name": "reader"})
    assert processor._signal_changed({"schema:dbo": [1, 1]}, {"schema:dbo": [0, 1]})


def test_recompile_replaces_generated_semantics_of_a_changed_table(
    env, monkeypatch: pytest.MonkeyPatch
):
    """A changed table's old generated Term/attributes do not survive its recompile.

    Compilation upserts by name, so a recompile that names the table differently
    used to leave both sets on the table (seen live: "Test Record" beside
    "Test Watch Record"). A Term another table shares is only unlinked.
    """
    import auto_ontology.ingestion_service.config as config
    import auto_ontology.semantic.compile as compile_mod
    import auto_ontology.semantic.pipeline as pipeline

    c = _connector(env)
    processor.process_database(c, "new")
    old = _add_semantics(c.database_name, "t1", "id", human=False)
    other = _add_semantics(c.database_name, "t0", "id", human=False)
    # a generated Term linked to both t1 and t0: kept, only unlinked from t1
    shared_term = store().query_write(
        insert(s.term)
        .values(name=f"shared-{uuid.uuid4().hex[:6]}", source="semantic")
        .returning(s.term.c.id)
    )[0]["id"]
    for table_id in (old["table_id"], other["table_id"]):
        store().query_write(
            insert(s.table__term).values(table_id=table_id, term_id=shared_term)
        )
    live = sig.catalog_rows(c.database_name)[2][("dbo", "t0", "id")]
    store().query_write(
        insert(s.column__semantic_fk).values(column_id=live, attribute_id=old["attr"])
    )

    monkeypatch.setattr(config, "is_semantic_compilation_enabled", lambda: True)
    monkeypatch.setattr(compile_mod, "build_semantic_embedder", lambda *a, **k: None)
    monkeypatch.setattr(
        compile_mod, "run_semantic_compilation", lambda db, failures=None: 0
    )

    def fake_compile(database_name: str, *, table_ids: list, protect: Any, **_: Any):
        return sum(not protect({"id": t}) for t in table_ids)

    monkeypatch.setattr(pipeline, "compile_semantic_layer", fake_compile)
    c.tables["t1"].append(("memo", "nvarchar", True))
    report = processor.process_database(c, "change")
    assert report.result == "succeeded", report.failures

    def exists(table: Any, row_id: str) -> bool:
        return bool(store().query_read(select(table).where(table.c.id == row_id)))

    assert not exists(s.term, old["term"])
    assert not exists(s.column_attribute, old["attr"])
    assert exists(s.term, shared_term)  # shared with t0: kept, unlinked from t1
    links = store().query_read(
        select(s.table__term.c.table_id).where(s.table__term.c.term_id == shared_term)
    )
    assert [r["table_id"] for r in links] == [other["table_id"]]
    assert exists(s.term, other["term"])  # unchanged table untouched
    assert exists(s.column_attribute, other["attr"])
    assert not store().query_read(
        select(s.column__semantic_fk).where(s.column__semantic_fk.c.column_id == live)
    )
    assert (
        report.stages["compile"]["replaced_objects"] == 2
    )  # t1's own term + attribute
    assert sum(k == "semantic_replaced" for k, _ in _changes(c.database_name)) == 2


def test_recompile_keeps_human_edited_table_objects(
    env, monkeypatch: pytest.MonkeyPatch
):
    import auto_ontology.ingestion_service.config as config
    import auto_ontology.semantic.compile as compile_mod
    import auto_ontology.semantic.pipeline as pipeline

    c = _connector(env)
    processor.process_database(c, "new")
    human = _add_semantics(c.database_name, "t2", "id", human=True)
    monkeypatch.setattr(config, "is_semantic_compilation_enabled", lambda: True)
    monkeypatch.setattr(compile_mod, "build_semantic_embedder", lambda *a, **k: None)
    monkeypatch.setattr(
        compile_mod, "run_semantic_compilation", lambda db, failures=None: 0
    )
    monkeypatch.setattr(
        pipeline,
        "compile_semantic_layer",
        lambda db, *, table_ids, protect, **_: [protect({"id": t}) for t in table_ids],
    )
    c.tables["t2"].append(("y", "int", True))
    report = processor.process_database(c, "change")
    assert report.stages["compile"]["protected"] == [human["table_id"]]
    assert report.stages["compile"]["replaced_objects"] == 0
    for table, row_id in ((s.term, human["term"]), (s.column_attribute, human["attr"])):
        assert store().query_read(select(table).where(table.c.id == row_id))
