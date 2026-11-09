# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recurring scheduler for data ingestion, with optional source-schema watch.

Every ingest goes through :func:`processor.process_database` (one handler, one
write lock, persisted run state). Two modes:

* **watch off** (default, upstream behaviour): ingest every connection at
  startup and every 24h measured from the end of the previous pass.
* **watch on** (``SCHEMA_WATCH_ENABLED=true``): every
  ``SCHEMA_WATCH_INTERVAL_MINUTES`` (10) each connection's structure is read
  with metadata calls only and reduced to a signature
  (:mod:`schema_signature`). Ingest runs only when

  - the signature differs from the last applied one on **two passes in a row**
    (and the last change ingest is at least ``SCHEMA_WATCH_MIN_INGEST_GAP_MINUTES``
    ago);
  - an open failure is due for retry (10 min apart x3, then hourly). While a
    connection has open failures the signature path does not run, so the two
    never both start the handler;
  - the daily forced ingest is due (``SCHEMA_WATCH_FORCE_AT``, default 04:30 KST,
    tracked by date);
  - the connection is new (no catalog rows).

  An existing connection seen for the first time (feature just deployed) is not
  ingested: its snapshot is initialised from the current observation, and any
  catalog difference is applied by that night's forced ingest.

The watch pass does **not** invalidate the connector cache (that closes every
connector); new or edited connections reach this process through the web API's
ingest request and the handler.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone

from auto_ontology.connectors.registry import (
    get_connectors,
    invalidate_connectors_cache,
)
from auto_ontology.dal import schema_state as st
from auto_ontology.ingestion_service import schema_signature as sig
from auto_ontology.ingestion_service.config import is_semantic_compilation_enabled
from auto_ontology.ingestion_service.rules import replay_rules_in_thread
from auto_ontology.ingestion_service.scheduler import IntervalScheduler

logger = logging.getLogger(__name__)

# Cadence measured from the end of the previous run rather than a fixed
# wall-clock time.
INGEST_INTERVAL = timedelta(hours=24)
KST = timezone(timedelta(hours=9))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def watch_enabled() -> bool:
    return _env("SCHEMA_WATCH_ENABLED", "false").lower() in ("1", "true", "yes", "on")


def watch_interval() -> timedelta:
    return timedelta(minutes=float(_env("SCHEMA_WATCH_INTERVAL_MINUTES", "10")))


def min_ingest_gap() -> timedelta:
    return timedelta(minutes=float(_env("SCHEMA_WATCH_MIN_INGEST_GAP_MINUTES", "60")))


def force_at() -> tuple[int, int]:
    hour, _, minute = _env("SCHEMA_WATCH_FORCE_AT", "04:30").partition(":")
    return int(hour), int(minute or 0)


def force_due(snapshot: dict | None, now: datetime | None = None) -> bool:
    now = (now or datetime.now(timezone.utc)).astimezone(KST)
    hour, minute = force_at()
    if (now.hour, now.minute) < (hour, minute):
        return False
    return (snapshot or {}).get("last_force_date") != now.date().isoformat()


class DataScheduler(IntervalScheduler):
    """Drives data ingestion on startup, on a timer, and on demand."""

    name = "ingest"

    def __init__(self, interval: timedelta | None = None) -> None:
        super().__init__(
            interval or (watch_interval() if watch_enabled() else INGEST_INTERVAL)
        )

    async def _run_once(self) -> None:
        if watch_enabled():
            await self._watch_pass()
        else:
            await self._ingest_all_pass()

    # ------------------------------------------------------------ watch off

    async def _ingest_all_pass(self) -> None:
        from auto_ontology.ingestion_service.processor import process_database

        from auto_ontology.ingestion_service.write_lock import write_lock

        started = time.monotonic()

        def run_all() -> tuple[int, int, int]:
            # Reload connectors so newly added connections are picked up.
            # Invalidation closes every connector, so it happens under the write
            # lock, where no other handler run can be using one.
            with write_lock() as lock:
                invalidate_connectors_cache()
                connectors = get_connectors()
                ok = bad = 0
                for connector in connectors:
                    report = process_database(connector, "scheduled", lock=lock)
                    if report.result in (st.RESULT_SUCCEEDED, st.RESULT_SKIPPED):
                        ok += 1
                    else:
                        bad += 1
                        logger.warning(
                            "ingest: %s for connection %s",
                            report.result,
                            getattr(connector, "database_name", "?"),
                        )
                return len(connectors), ok, bad

        total, succeeded, failed = await asyncio.to_thread(run_all)
        if not total:
            logger.info(
                "ingest: no connections configured. "
                "Add a connection in Settings → Connections or set "
                "CONNECTION_STRINGS in your .env."
            )
            return
        connectors = range(total)

        # Rules are replayed at the end of the *semantic* pass when compilation
        # is on (see rules.py); with it off this is the only replay point. The
        # handler already replays after each run, so this only covers the
        # no-connection-changed case for parity with upstream.
        if not is_semantic_compilation_enabled():
            await replay_rules_in_thread(self.name)

        elapsed = time.monotonic() - started
        if failed:
            logger.warning(
                "ingest: finished with errors — %d of %d connection(s) succeeded, "
                "%d failed in %.1fs",
                succeeded,
                len(connectors),
                failed,
                elapsed,
            )
        else:
            logger.info(
                "ingest: finished successfully — %d connection(s) ingested in %.1fs",
                succeeded,
                elapsed,
            )

    # ------------------------------------------------------------ watch on

    async def _watch_pass(self) -> None:
        connectors = get_connectors()  # no invalidation on a watch pass
        if not connectors:
            return
        for connector in connectors:
            try:
                await asyncio.to_thread(self._watch_one, connector)
            except Exception:
                logger.exception(
                    "schema-watch: failed for connection %s",
                    getattr(connector, "database_name", "?"),
                )

    def _watch_one(self, connector) -> None:  # noqa: ANN001 - SQLDatabase
        from auto_ontology.ingestion_service.processor import process_database

        database_name = str(connector.database_name)
        now = datetime.now(timezone.utc)
        snapshot = st.get_snapshot(database_name)

        # 1. open failures: retry on their schedule, nothing else
        failures = st.open_failures(database_name)
        if failures:
            if any(st.retry_due(f, now) for f in failures):
                logger.info(
                    "schema-watch %s: retrying %d open failure(s)",
                    database_name,
                    len(failures),
                )
                process_database(connector, "retry")
            return

        # 2. first sight of a connection
        if snapshot is None or not snapshot.get("applied_signature"):
            catalog, _, _ = sig.catalog_rows(database_name)
            if not catalog["tables"]:
                logger.info("schema-watch %s: new connection, ingesting", database_name)
                process_database(connector, "new")
                return
            observed = sig.observe(connector)
            st.upsert_snapshot(
                database_name,
                applied_signature=sig.signature(observed),
                applied_rows={
                    **sig.rows_to_json(observed),
                    "_embed_hash": None,
                    "_initialised_from": "observation",
                },
                last_success_run_id=(snapshot or {}).get("last_success_run_id")
                or "initialised",
                auth_signal=sig.auth_signal(connector),
                last_checked_at=now,
            )
            logger.info(
                "schema-watch %s: snapshot initialised from current structure "
                "(no ingest; the forced ingest at %02d:%02d applies any catalog difference)",
                database_name,
                *force_at(),
            )
            return

        # 3. daily forced ingest
        if force_due(snapshot, now):
            logger.info("schema-watch %s: daily forced ingest", database_name)
            process_database(connector, "force")
            return

        # 4. signature comparison
        observed = sig.observe(connector)
        signature = sig.signature(observed)
        st.upsert_snapshot(database_name, last_checked_at=now)
        if signature == snapshot["applied_signature"]:
            if (
                snapshot.get("pending_signature")
                and snapshot.get("pending_reason") == "change"
            ):
                st.upsert_snapshot(
                    database_name,
                    pending_signature=None,
                    pending_since=None,
                    pending_reason=None,
                )
            return

        reason = snapshot.get("pending_reason")
        if reason == "bulk_delete" and snapshot.get("pending_signature") == signature:
            from auto_ontology.ingestion_service.processor import bulk_delete_delay

            since = snapshot.get("pending_since")
            if since is None or now - since < bulk_delete_delay():
                return  # still inside the bulk-delete delay; the handler decides then
            process_database(connector, "change")
            return

        if snapshot.get("pending_signature") != signature:
            st.upsert_snapshot(
                database_name,
                pending_signature=signature,
                pending_since=now,
                pending_reason="change" if reason != "auth_signal_changed" else reason,
            )
            logger.info(
                "schema-watch %s: structure changed, confirming on next pass",
                database_name,
            )
            return

        last = snapshot.get("last_change_ingest_at")
        if last is not None and now - last < min_ingest_gap():
            logger.info(
                "schema-watch %s: change confirmed, waiting for the ingest gap",
                database_name,
            )
            return
        logger.info("schema-watch %s: change confirmed, ingesting", database_name)
        process_database(connector, "change")
