# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one handler every ingest goes through.

Scheduled, forced, change-triggered, manual (``POST /ingest``), connection-create
and retry ingests all call :func:`process_database`, under the global write lock
(:mod:`write_lock`) and with a persisted run record (:mod:`dal.schema_state`).

Stages, in order (``plan 6.1``)::

    extract  -> judge deletions -> impact analysis -> semantic cleanup
             -> catalog write (+ verify) -> PII detection -> data embedding (+ verify)
             -> compile (new tables, changed tables, failed tables) -> PII/rules
             -> publish (snapshot, generation)

Every stage records its own failure in ``schema_open_failure`` instead of being
swallowed, and a run with any failure is ``partial``: the applied snapshot and
the generation stay where they were, and the next pass retries. A stage that
succeeds clears the open failures it covers.

Deletion is judged before anything is written, because the catalog diff deletes
whatever is missing from the extract (``catalog/diff.py``) and semantic
ownership can only be resolved while the catalog rows still exist:

* an account/permission signal that differs from the last run holds deletions
  (they look like deletions but may be visibility) until it repeats;
* a bulk deletion (>= ratio of tables, a whole schema, an empty extract) is held
  for ``SCHEMA_WATCH_BULK_DELETE_DELAY_MINUTES`` and logged as a notice;
* held rows are put back into the extract, so additions and changes still land.

Semantic objects of deleted tables/columns are removed only when generated
(``source`` semantic/sql and never edited); human-written or edited ones and
every CustomAnalysis are kept and logged as orphans. Changed tables are
recompiled by table id, except tables that carry human-written semantic
objects, which are kept as they are and logged for review.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from sqlalchemy import and_, delete, or_, select

from auto_ontology.dal import schema as s
from auto_ontology.dal import schema_state as st
from auto_ontology.dal.session import store
from auto_ontology.ingestion_service import schema_signature as sig
from auto_ontology.ingestion_service.write_lock import WriteLock, write_lock
from auto_ontology.semantic.constants import SEMANTIC_SOURCE, SQL_ATTR_SOURCE_SQL

logger = logging.getLogger("ingestion_service.processor")

AUTO_SOURCES = (SEMANTIC_SOURCE, SQL_ATTR_SOURCE_SQL)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def bulk_delete_ratio() -> float:
    return _env_float("SCHEMA_WATCH_BULK_DELETE_RATIO", 0.10)


def bulk_delete_delay() -> timedelta:
    return timedelta(minutes=_env_float("SCHEMA_WATCH_BULK_DELETE_DELAY_MINUTES", 60))


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class RunReport:
    run_id: str
    database_name: str
    trigger: str
    result: str = st.RESULT_SUCCEEDED
    stages: dict[str, Any] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    diff: dict[str, Any] = field(default_factory=dict)
    held: str | None = None
    generation: int | None = None

    def fail(self, stage: str, target: str = "", error: str = "") -> None:
        self.failures.append({"stage": stage, "target": target, "error": error})


# ---------------------------------------------------------------- deletion judgement


def _removed_schemas(d: dict[str, Any], old: sig.Rows, new: sig.Rows) -> list[str]:
    old_schemas = {r[0] for r in old.get("tables", [])}
    new_schemas = {r[0] for r in new.get("tables", [])}
    return sorted(old_schemas - new_schemas)


def _signal_changed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Compare only keys both signals carry (schemas come and go legitimately)."""
    return any(before[k] != after[k] for k in before.keys() & after.keys())


def judge_deletions(
    database_name: str,
    d: dict[str, Any],
    old: sig.Rows,
    new: sig.Rows,
    snapshot: dict[str, Any] | None,
    auth: dict[str, Any] | None,
    now: datetime | None = None,
) -> tuple[str, str | None]:
    """``("apply", None)``, ``("hold", reason)`` or ``("apply_empty", None)``.

    ``hold`` keeps the deletions out of this run (additions and changes still
    apply). The caller records the pending state.
    """
    now = now or _now()
    removed = d["removed_tables"] or d["removed_columns"]
    if not removed:
        return "apply", None

    previous_auth = (snapshot or {}).get("auth_signal")
    if (
        previous_auth is not None
        and auth is not None
        and _signal_changed(previous_auth, auth)
    ):
        pending_auth = (snapshot or {}).get("pending_auth_signal")
        if pending_auth is not None and not _signal_changed(pending_auth, auth):
            logger.warning(
                "schema-watch %s: account/permission signal changed and repeated; "
                "accepting the new signal and applying deletions",
                database_name,
            )
        else:
            return "hold", "auth_signal_changed"

    total = d["total_tables_before"]
    empty = not new.get("tables")
    bulk = (
        empty
        or (
            total > 0
            and len(d["removed_tables"]) >= max(1.0, bulk_delete_ratio() * total)
        )
        or bool(_removed_schemas(d, old, new))
    )
    if not bulk:
        return "apply", None

    pending_since = (snapshot or {}).get("pending_since")
    same_pending = (
        (snapshot or {}).get("pending_reason") == "bulk_delete"
        and (snapshot or {}).get("pending_signature") == sig.signature(new)
        and pending_since is not None
    )
    if same_pending and now - pending_since >= bulk_delete_delay():
        return ("apply_empty" if empty else "apply"), None
    return "hold", "bulk_delete"


def _restore_held(data: dict[str, Any], old: sig.Rows, d: dict[str, Any]) -> None:
    """Put held (removed) tables/columns back into the extract so they are kept."""
    removed_tables = set(d["removed_tables"])
    removed_columns = set(d["removed_columns"])
    table_rows = [
        {"table_schema": r[0], "table_name": r[1], "table_type": r[2].upper()}
        for r in old["tables"]
        if (r[0], r[1]) in removed_tables
    ]
    column_rows = [
        {
            "table_schema": r[0],
            "table_name": r[1],
            "column_name": r[2],
            "data_type": r[3],
            "is_nullable": bool(r[4]),
        }
        for r in old["columns"]
        if (r[0], r[1]) in removed_tables or (r[0], r[1], r[2]) in removed_columns
    ]
    pk_rows = [
        {
            "table_schema": r[0],
            "table_name": r[1],
            "ordinal_position": r[2],
            "column_name": r[3],
        }
        for r in old["pks"]
        if (r[0], r[1]) in removed_tables
    ]
    for key, rows in (
        ("tables", table_rows),
        ("columns", column_rows),
        ("pks", pk_rows),
    ):
        if not rows:
            continue
        frame = data.get(key)
        extra = pd.DataFrame(rows)
        data[key] = (
            extra
            if frame is None or frame.empty
            else pd.concat([frame, extra], ignore_index=True)
        )
    if table_rows or column_rows:
        from auto_ontology.catalog.normalize import normalize_columns, normalize_tables

        data["tables"] = normalize_tables(data["tables"])
        data["columns"] = normalize_columns(data["columns"])


# ---------------------------------------------------------------- semantic impact & cleanup


def _is_auto(row: dict[str, Any]) -> bool:
    return row.get("source") in AUTO_SOURCES and row.get("modified") is None


def impact_and_cleanup(
    report: RunReport,
    d: dict[str, Any],
    table_ids: dict[tuple, str],
    column_ids: dict[tuple, str],
    apply_cleanup: bool,
) -> list[dict[str, Any]]:
    """Find semantic objects attached to removed tables/columns (plan 8.1/8.2).

    Must run before the catalog write, while the catalog rows still exist.
    Returns change-log entries.
    """
    entries: list[dict[str, Any]] = []
    for key in d["added_tables"]:
        entries.append({"kind": "table_added", "object": ".".join(key)})
    for key in d["added_columns"]:
        entries.append({"kind": "column_added", "object": ".".join(key)})
    for key in d["changed_columns"]:
        entries.append({"kind": "column_changed", "object": ".".join(key)})
    removed_table_ids = [table_ids[k] for k in d["removed_tables"] if k in table_ids]
    removed_column_ids = [
        column_ids[k] for k in d["removed_columns"] if k in column_ids
    ] + [
        cid
        for (sc, tb, _c), cid in column_ids.items()
        if (sc, tb) in set(d["removed_tables"])
    ]
    for key in d["removed_tables"]:
        entries.append(
            {
                "kind": "table_removed",
                "object": ".".join(key),
                "detail": {
                    "reason": report.stages.get("removed_reason", {}).get(
                        ".".join(key), "dropped"
                    )
                },
            }
        )
    for key in d["removed_columns"]:
        entries.append({"kind": "column_removed", "object": ".".join(key)})
    if not removed_table_ids and not removed_column_ids:
        return entries

    # Terms reached through removed tables
    term_links = store().query_read(
        select(s.table__term.c.term_id, s.table__term.c.table_id).where(
            s.table__term.c.table_id.in_(removed_table_ids or [""])
        )
    )
    term_ids = sorted({r["term_id"] for r in term_links})
    terms = store().query_read(select(s.term).where(s.term.c.id.in_(term_ids or [""])))
    other_links = store().query_read(
        select(s.table__term.c.term_id).where(
            and_(
                s.table__term.c.term_id.in_(term_ids or [""]),
                s.table__term.c.table_id.notin_(removed_table_ids or [""]),
            )
        )
    )
    shared_terms = {r["term_id"] for r in other_links}

    # ColumnAttributes owned by removed tables or linked to removed columns
    attrs = store().query_read(
        select(s.column_attribute).where(
            or_(
                s.column_attribute.c.table_id.in_(removed_table_ids or [""]),
                s.column_attribute.c.id.in_(
                    select(s.column__has_attribute.c.attribute_id).where(
                        s.column__has_attribute.c.column_id.in_(
                            removed_column_ids or [""]
                        )
                    )
                ),
            )
        )
    )
    attr_ids = [a["id"] for a in attrs]
    incoming_fks = store().query_read(
        select(s.column__semantic_fk).where(
            s.column__semantic_fk.c.attribute_id.in_(attr_ids or [""])
        )
    )

    # SQL-backed objects whose SQL references removed tables or columns
    def sql_owners(
        owner_table: Any, link_table: Any, owner_column: Any
    ) -> list[dict[str, Any]]:
        ids = select(owner_column).where(
            link_table.c.sql_query_id.in_(
                select(s.sql_query__table.c.sql_query_id)
                .where(s.sql_query__table.c.table_id.in_(removed_table_ids or [""]))
                .union(
                    select(s.sql_query__column.c.sql_query_id).where(
                        s.sql_query__column.c.column_id.in_(removed_column_ids or [""])
                    )
                )
            )
        )
        return store().query_read(select(owner_table).where(owner_table.c.id.in_(ids)))

    sql_attrs = sql_owners(
        s.sql_attribute, s.sql_attribute__sql, s.sql_attribute__sql.c.attribute_id
    )
    analyses = sql_owners(
        s.custom_analysis, s.custom_analysis__sql, s.custom_analysis__sql.c.analysis_id
    )

    to_delete_vectors: list[str] = []

    def orphan(kind: str, row: dict[str, Any], why: str) -> None:
        entries.append(
            {
                "kind": "semantic_orphaned",
                "object": f"{kind}:{row.get('name') or row.get('id')}",
                "detail": {"id": row.get("id"), "type": kind, "why": why},
            }
        )

    for term in terms:
        if term["id"] in shared_terms:
            entries.append(
                {
                    "kind": "semantic_unlinked",
                    "object": f"term:{term['name']}",
                    "detail": {"id": term["id"]},
                }
            )
        elif _is_auto(term):
            entries.append(
                {
                    "kind": "semantic_removed",
                    "object": f"term:{term['name']}",
                    "detail": {"id": term["id"]},
                }
            )
            to_delete_vectors.append(term["id"])
        else:
            orphan("term", term, "human written or edited")
    for attr in attrs:
        if _is_auto(attr):
            entries.append(
                {
                    "kind": "semantic_removed",
                    "object": f"column_attribute:{attr.get('name')}",
                    "detail": {"id": attr["id"]},
                }
            )
            to_delete_vectors.append(attr["id"])
        else:
            orphan("column_attribute", attr, "human written or edited")
    for fk in incoming_fks:
        entries.append(
            {
                "kind": "semantic_fk_unlinked",
                "object": f"{fk['column_id']}->{fk['attribute_id']}",
            }
        )
    for attr in sql_attrs:
        if _is_auto(attr):
            entries.append(
                {
                    "kind": "semantic_removed",
                    "object": f"sql_attribute:{attr.get('name')}",
                    "detail": {"id": attr["id"]},
                }
            )
        else:
            orphan("sql_attribute", attr, "references removed table/column")
    for analysis in analyses:
        orphan("custom_analysis", analysis, "references removed table/column")

    report.stages["impact"] = {
        "terms": len(terms),
        "column_attributes": len(attrs),
        "sql_attributes": len(sql_attrs),
        "custom_analyses": len(analyses),
        "incoming_semantic_fks": len(incoming_fks),
    }
    if not apply_cleanup:
        return entries

    # Re-check edit state right before deleting (an edit may have landed meanwhile)
    _delete_semantics(
        report,
        fk_pairs=[(fk["column_id"], fk["attribute_id"]) for fk in incoming_fks],
        unlink_terms=list(shared_terms),
        from_tables=removed_table_ids,
        attr_ids=[a["id"] for a in attrs if _is_auto(a)],
        term_ids=[
            t["id"] for t in terms if t["id"] not in shared_terms and _is_auto(t)
        ],
        sql_attr_ids=[a["id"] for a in sql_attrs if _is_auto(a)],
        vector_ids=to_delete_vectors,
    )
    return entries


def _delete_semantics(
    report: RunReport,
    *,
    fk_pairs: list[tuple[str, str]],
    unlink_terms: list[str],
    from_tables: list[str],
    attr_ids: list[str],
    term_ids: list[str],
    sql_attr_ids: list[str],
    vector_ids: list[str],
) -> None:
    """Delete generated semantic objects and their vectors.

    Rows edited since they were selected are kept (``modified`` re-checked in
    the delete itself). Shared terms are only unlinked from *from_tables*.
    """
    for column_id, attribute_id in fk_pairs:
        store().query_write(
            delete(s.column__semantic_fk).where(
                and_(
                    s.column__semantic_fk.c.column_id == column_id,
                    s.column__semantic_fk.c.attribute_id == attribute_id,
                )
            )
        )
    if unlink_terms:
        store().query_write(
            delete(s.table__term).where(
                and_(
                    s.table__term.c.term_id.in_(unlink_terms),
                    s.table__term.c.table_id.in_(from_tables or [""]),
                )
            )
        )
    for table, ids in (
        (s.column_attribute, attr_ids),
        (s.term, term_ids),
        (s.sql_attribute, sql_attr_ids),
    ):
        if ids:
            store().query_write(
                delete(table).where(
                    and_(table.c.id.in_(ids), table.c.modified.is_(None))
                )
            )
    if vector_ids:
        from auto_ontology.vdb import get_semantic_vdb

        semantic_vdb = get_semantic_vdb()
        for node_id in vector_ids:
            try:
                semantic_vdb.delete_by_id(node_id)
            except Exception as exc:  # noqa: BLE001 - recorded, retried next run
                report.fail("semantic_cleanup", node_id, str(exc))


def purge_table_semantics(report: RunReport, table_id: str) -> list[dict[str, Any]]:
    """Remove a table's generated semantic objects before it is recompiled.

    Compilation upserts Terms and attributes by name, so when the model names a
    recompiled table differently ("Test Record" -> "Test Watch Record") the old
    objects stay beside the new ones and the table carries both. Only called
    for tables ``protected_table`` let through, i.e. ones with no human-written
    or edited objects; a Term other tables share is unlinked, not deleted. If
    the recompile then fails, the table has no Term and the follow-up
    ``run_semantic_compilation`` pass (and the compile retry) rebuilds it.
    Returns change-log entries.
    """
    term_ids = [
        r["term_id"]
        for r in store().query_read(
            select(s.table__term.c.term_id).where(s.table__term.c.table_id == table_id)
        )
    ]
    terms = store().query_read(select(s.term).where(s.term.c.id.in_(term_ids or [""])))
    shared = {
        r["term_id"]
        for r in store().query_read(
            select(s.table__term.c.term_id).where(
                and_(
                    s.table__term.c.term_id.in_(term_ids or [""]),
                    s.table__term.c.table_id != table_id,
                )
            )
        )
    }
    attrs = store().query_read(
        select(s.column_attribute).where(s.column_attribute.c.table_id == table_id)
    )
    attr_ids = [a["id"] for a in attrs if _is_auto(a)]
    incoming = store().query_read(
        select(s.column__semantic_fk).where(
            s.column__semantic_fk.c.attribute_id.in_(attr_ids or [""])
        )
    )
    own_terms = [t for t in terms if t["id"] not in shared and _is_auto(t)]
    entries = [
        {
            "kind": "semantic_replaced",
            "object": f"{kind}:{row.get('name')}",
            "detail": {"id": row["id"], "table_id": table_id},
        }
        for kind, rows in (
            ("term", own_terms),
            ("column_attribute", [a for a in attrs if _is_auto(a)]),
        )
        for row in rows
    ]
    _delete_semantics(
        report,
        fk_pairs=[(fk["column_id"], fk["attribute_id"]) for fk in incoming],
        unlink_terms=sorted(shared),
        from_tables=[table_id],
        attr_ids=attr_ids,
        term_ids=[t["id"] for t in own_terms],
        sql_attr_ids=[],
        vector_ids=[t["id"] for t in own_terms] + attr_ids,
    )
    return entries


def protected_table(table: dict[str, Any]) -> bool:
    """True when a table carries human-written or edited semantic objects (plan 8.3).

    Such a table is not recompiled: Term upserts rewrite descriptions and
    synonyms, which would overwrite the edits. It is logged for review instead.
    """
    table_id = table["id"]
    terms = store().query_read(
        select(s.term.c.source, s.term.c.modified).where(
            s.term.c.id.in_(
                select(s.table__term.c.term_id).where(
                    s.table__term.c.table_id == table_id
                )
            )
        )
    )
    attrs = store().query_read(
        select(s.column_attribute.c.source, s.column_attribute.c.modified).where(
            s.column_attribute.c.table_id == table_id
        )
    )
    return any(not _is_auto(dict(r)) for r in terms + attrs)


# ---------------------------------------------------------------- stages


def _verify_catalog(database_name: str, expected: sig.Rows) -> list[str]:
    actual, _, _ = sig.catalog_rows(database_name)
    problems = []
    for kind in ("tables", "columns"):
        missing = set(expected[kind]) - set(actual[kind])
        extra = set(actual[kind]) - set(expected[kind])
        if missing:
            problems.append(f"{kind} missing after write: {sorted(missing)[:5]}")
        if extra:
            problems.append(f"{kind} left over after write: {sorted(extra)[:5]}")
    expected_pk = {(r[0], r[1]) for r in expected["pks"]}
    actual_pk_rows = {r for r in actual["pks"] if (r[0], r[1]) in expected_pk}
    if set(expected["pks"]) != actual_pk_rows:
        problems.append("primary keys differ after write")
    return problems


def _embed_hash(embed_rows: Any) -> str:
    try:
        texts = (
            sorted(str(t) for t in embed_rows["text"].tolist())
            if "text" in embed_rows
            else []
        )
    except Exception:  # noqa: BLE001
        texts = []
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()


def _write_and_embed(
    connector: Any, data: dict[str, Any], report: RunReport, lock: WriteLock
) -> tuple[Any, Any, str]:
    from auto_ontology.catalog.write import populate_tabular_data
    from auto_ontology.infra.feature_flags import is_pii_detection_enabled
    from auto_ontology.ingestion_service.ingest import _run_pii_detection
    from auto_ontology.utils import get_embed_params
    from auto_ontology.utils.embedding import batch_embed_chunks
    from auto_ontology.utils.embedding_rows import CatalogEmbeddingRowsOp
    from auto_ontology.vdb import get_data_vdb
    from auto_ontology.vdb.records import to_vdb_records

    database_name = report.database_name
    lock.stage("catalog")
    schemas = populate_tabular_data(data, num_workers=4, dialect=connector.dialect)
    tables = [x.tables_df for x in (schemas or {}).values() if x.tables_df is not None]
    columns = [
        x.columns_df for x in (schemas or {}).values() if x.columns_df is not None
    ]
    tables_df = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    columns_df = pd.concat(columns, ignore_index=True) if columns else pd.DataFrame()

    problems = _verify_catalog(database_name, sig.rows_from_extract(data))
    report.stages["catalog"] = {
        "tables": len(tables_df),
        "columns": len(columns_df),
        "problems": problems,
    }
    if problems:
        report.fail("catalog", "", "; ".join(problems))

    if is_pii_detection_enabled():
        lock.stage("pii_detection")
        try:
            _run_pii_detection(columns_df, database_name)
        except Exception as exc:  # noqa: BLE001
            report.fail("pii", "detection", str(exc))

    lock.stage("embedding")
    embed_rows = CatalogEmbeddingRowsOp(database_name=database_name)(
        (tables_df, columns_df)
    )
    expected = 0 if embed_rows is None else len(embed_rows)
    produced = written = 0
    data_vdb = None
    try:
        for chunk in batch_embed_chunks(
            embed_rows, get_embed_params(), label=database_name
        ):
            lock.check()
            records = to_vdb_records(chunk)
            if not records:
                continue
            produced += sum(len(r) for r in records)
            if data_vdb is None:
                data_vdb = get_data_vdb(database_name=database_name, reset=True)
            written += data_vdb.run(records)
    except Exception as exc:  # noqa: BLE001
        report.fail("embedding", "", str(exc))
    report.stages["embedding"] = {
        "expected": expected,
        "produced": produced,
        "written": written,
    }
    if expected and written != expected:
        report.fail("embedding", "", f"wrote {written} of {expected} embedding row(s)")
    return tables_df, columns_df, _embed_hash(embed_rows)


def _compile(report: RunReport, d: dict[str, Any], lock: WriteLock) -> None:
    from auto_ontology.ingestion_service.config import is_semantic_compilation_enabled
    from auto_ontology.semantic.compile import run_semantic_compilation

    if not is_semantic_compilation_enabled():
        report.stages["compile"] = "disabled"
        return
    database_name = report.database_name
    lock.stage("compile")
    _, table_ids, _ = sig.catalog_rows(database_name)
    failed_targets = [
        f["target"]
        for f in st.open_failures(database_name)
        if f["stage"] == "compile_table" and f["target"]
    ]
    changed = [table_ids[k] for k in d.get("changed_tables", []) if k in table_ids]
    targeted = sorted(set(changed) | set(failed_targets))
    protected: list[str] = []

    replaced: list[dict[str, Any]] = []

    def protect(table: dict[str, Any]) -> bool:
        if protected_table(table):
            protected.append(table["id"])
            return True
        # Not protected: drop its generated objects so the recompile replaces
        # them instead of adding a second, differently named set.
        entries = purge_table_semantics(report, table["id"])
        replaced.extend(entries)
        st.log_changes(report.run_id, database_name, entries)
        return False

    failures: list[dict] = []
    try:
        if targeted:
            # Targeted first (no FK/SQL-attribute stages twice: those run below)
            from auto_ontology.semantic.compile import build_semantic_embedder
            from auto_ontology.semantic.pipeline import compile_semantic_layer

            compile_semantic_layer(
                database_name,
                embedder=build_semantic_embedder(database_name, reset=False),
                failures=failures,
                table_ids=targeted,
                protect=protect,
            )
        lock.check()
        new_tables = run_semantic_compilation(database_name, failures=failures)
    except Exception as exc:  # noqa: BLE001
        report.fail("compile_post", "", str(exc))
        new_tables = 0
    failed_now = {f["target"] for f in failures}
    for f in failures:
        report.fail(f["stage"], f["target"], f.get("error", ""))
    for target in targeted:
        if target not in failed_now:
            st.clear_failure(database_name, "compile_table", target)
    report.stages["compile"] = {
        "targeted": len(targeted),
        "protected": protected,
        "replaced_objects": len(replaced),
        "new_tables": new_tables,
        "failed": sorted(failed_now),
    }
    report.stages["protected_tables"] = protected


def _post(report: RunReport, lock: WriteLock) -> None:
    from auto_ontology.ingestion_service.pii import run_pii_propagation
    from auto_ontology.ingestion_service.rules import replay_rules

    lock.stage("post")
    try:
        run_pii_propagation(report.database_name)
    except Exception as exc:  # noqa: BLE001
        report.fail("post", "pii_propagation", str(exc))
    try:
        replay_rules("schema-ingest")
    except Exception as exc:  # noqa: BLE001
        report.fail("post", "rules", str(exc))


def _empty_catalog(report: RunReport) -> None:
    """Clear the catalog and data vectors of a database whose extract stayed empty.

    ``run_ingest`` cannot express this (an empty extract skips the write), and
    ``delete_all_data`` would also delete human-written semantic objects. The
    semantic cleanup ran before this, under the normal rules.
    """
    from auto_ontology.vdb import get_data_vdb

    database_name = report.database_name
    store().query_write(
        delete(s.catalog_database).where(s.catalog_database.c.name == database_name)
    )
    get_data_vdb().delete_by_database(database_name)
    remaining, _, _ = sig.catalog_rows(database_name)
    if remaining["tables"]:
        report.fail("catalog", "", "catalog not empty after empty-catalog apply")
    report.stages["catalog"] = {"emptied": True}


# ---------------------------------------------------------------- entry point


def process_database(
    connector: Any,
    trigger: str,
    *,
    lock: WriteLock | None = None,
    wait: bool = True,
) -> RunReport:
    """Run the handler for one connection. Takes the write lock unless given one."""
    if lock is None:
        with write_lock(wait=wait) as held:
            return _process_locked(connector, trigger, held)
    return _process_locked(connector, trigger, lock)


def _process_locked(connector: Any, trigger: str, lock: WriteLock) -> RunReport:
    database_name = str(connector.database_name)
    st.reap_stale_runs()
    run_id = st.start_run(database_name, trigger)  # raises -> nothing is processed
    lock.attach_run(run_id)
    report = RunReport(run_id=run_id, database_name=database_name, trigger=trigger)
    started = time.monotonic()
    error: str | None = None
    try:
        _run(connector, report, lock)
    except Exception as exc:  # noqa: BLE001 - recorded on the run
        logger.exception("schema ingest %s failed for %s", trigger, database_name)
        error = str(exc)
        report.fail("run", "", error)
    finally:
        _finish(report, error, time.monotonic() - started)
    return report


def _run(connector: Any, report: RunReport, lock: WriteLock) -> None:
    from auto_ontology.catalog.extract import extract_tabular_db_data
    from auto_ontology.ingestion_service.ingest import _shared_connection

    database_name = report.database_name
    lock.stage("extract")
    with _shared_connection(connector):
        data = extract_tabular_db_data(connector)
        raw_tables = connector.get_tables()
    new_rows = sig.rows_from_extract(data)
    old_rows, table_ids, column_ids = sig.catalog_rows(database_name)
    snapshot = st.get_snapshot(database_name)

    lock.stage("judge")
    d = sig.diff(old_rows, new_rows)
    # Over every schema the catalog knows, so the signal's keys stay the same
    # from run to run (a set that depended on this run's deletions would always
    # differ from the previous run's).
    known_schemas = sorted(
        {r[0] for r in old_rows["tables"]} | {r[0] for r in new_rows["tables"]}
    )
    auth = sig.auth_signal(connector, known_schemas)
    present = (
        {
            (r.get("table_schema"), r.get("table_name"))
            for r in raw_tables.to_dict(orient="records")
        }
        if raw_tables is not None
        else set()
    )
    report.stages["removed_reason"] = {
        ".".join(k): ("filter_excluded" if k in present else "dropped")
        for k in d["removed_tables"]
    }
    decision, reason = judge_deletions(
        database_name, d, old_rows, new_rows, snapshot, auth
    )
    report.diff = {k: v for k, v in d.items() if k != "total_tables_before"}
    report.diff["decision"] = decision
    if decision == "hold":
        report.held = reason
        pending: dict[str, Any] = {"pending_reason": reason}
        if reason == "bulk_delete":
            new_sig = sig.signature(new_rows)
            if (snapshot or {}).get("pending_signature") != new_sig or (
                snapshot or {}
            ).get("pending_reason") != "bulk_delete":
                pending.update(pending_signature=new_sig, pending_since=_now())
            logger.warning(
                "schema-watch %s: bulk deletion held (%d table(s), %d column(s)); "
                "applies after %s if it persists",
                database_name,
                len(d["removed_tables"]),
                len(d["removed_columns"]),
                bulk_delete_delay(),
            )
        else:
            pending["pending_auth_signal"] = auth
            logger.warning(
                "schema-watch %s: deletions held, account/permission signal changed",
                database_name,
            )
        st.upsert_snapshot(database_name, **pending)
        st.log_changes(
            report.run_id,
            database_name,
            [
                {
                    "kind": "deletion_held",
                    "object": database_name,
                    "detail": {
                        "reason": reason,
                        "tables": [".".join(k) for k in d["removed_tables"]],
                        "columns": [".".join(k) for k in d["removed_columns"]],
                    },
                }
            ],
        )
        if not (d["added_tables"] or d["added_columns"] or d["changed_columns"]):
            report.result = st.RESULT_SKIPPED
            return
        _restore_held(data, old_rows, d)
        new_rows = sig.rows_from_extract(data)
        d = sig.diff(old_rows, new_rows)

    lock.stage("impact")
    entries = impact_and_cleanup(report, d, table_ids, column_ids, apply_cleanup=True)
    st.log_changes(report.run_id, database_name, entries)
    report.stages["changes"] = len(entries)

    if decision == "apply_empty":
        _empty_catalog(report)
        embed_hash = ""
    else:
        if not new_rows["tables"]:
            report.result = st.RESULT_SKIPPED
            return
        _, _, embed_hash = _write_and_embed(connector, data, report, lock)
        _compile(report, d, lock)
    _post(report, lock)

    report.stages["applied_signature"] = sig.signature(new_rows)
    report.stages["applied_rows"] = sig.rows_to_json(new_rows)
    report.stages["embed_hash"] = embed_hash
    report.stages["auth"] = auth


def _finish(report: RunReport, error: str | None, elapsed: float) -> None:
    database_name = report.database_name
    for failure in report.failures:
        st.record_failure(
            database_name,
            failure["stage"],
            failure["target"],
            report.run_id,
            failure["error"],
        )
    if report.failures:
        catalog_failed = any(f["stage"] in ("catalog", "run") for f in report.failures)
        report.result = (
            st.RESULT_FAILED if catalog_failed or error else st.RESULT_PARTIAL
        )
    elif report.result != st.RESULT_SKIPPED:
        # Everything this run covers succeeded: close those open failures.
        for stage in (
            "run",
            "catalog",
            "embedding",
            "compile_post",
            "post",
            "pii",
            "semantic_cleanup",
        ):
            st.clear_failure(database_name, stage)
        snapshot = st.get_snapshot(database_name) or {}
        applied = report.stages.get("applied_signature")
        embed_hash = report.stages.get("embed_hash")
        previous_rows = snapshot.get("applied_rows") or {}
        content_changed = (
            applied != snapshot.get("applied_signature")
            or embed_hash != previous_rows.get("_embed_hash")
            or not snapshot.get("last_success_run_id")
        )
        rows = dict(report.stages.get("applied_rows") or {})
        rows["_embed_hash"] = embed_hash
        values: dict[str, Any] = {
            "applied_signature": applied,
            "applied_rows": rows,
            "last_success_run_id": report.run_id,
            "auth_signal": report.stages.get("auth"),
            "pending_auth_signal": None,
        }
        if report.held is None:
            values.update(
                pending_signature=None, pending_since=None, pending_reason=None
            )
        if report.trigger == "force":
            values["last_force_date"] = datetime.now(_kst()).date().isoformat()
        if report.trigger in ("change", "new"):
            values["last_change_ingest_at"] = _now()
        st.upsert_snapshot(database_name, **values)
        if content_changed:
            report.generation = st.bump_generation(database_name)
    verification = {
        k: v for k, v in report.stages.items() if k not in ("applied_rows",)
    }
    verification["diff"] = {
        k: (len(v) if isinstance(v, list) else v) for k, v in report.diff.items()
    }
    verification["failures"] = report.failures
    st.finish_run(report.run_id, report.result, verification, error)
    logger.info(
        "schema ingest %s for %s: %s in %.1fs (failures=%d, generation=%s, held=%s)",
        report.trigger,
        database_name,
        report.result,
        elapsed,
        len(report.failures),
        report.generation,
        report.held,
    )


def _kst() -> timezone:
    return timezone(timedelta(hours=9))
