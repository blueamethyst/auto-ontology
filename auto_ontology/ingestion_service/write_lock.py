# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One global write lock for everything that rewrites ingested state.

Ingest (scheduled, forced, manual, connection-create), delete-ingest, the
standalone semantic pass and the semantic reset all take this lock, so they
never interleave -- previously only the reset had a lock of its own.

It is a Postgres session-level advisory lock with a fixed key, so it also holds
across processes (ingestion service and web API). A session lock belongs to the
connection that took it: a pooled connection handed back mid-run would release
it or leave it on someone else's session. The lock therefore keeps a
**dedicated connection for its whole lifetime**, and the run heartbeat is
written over that same connection by a background thread. If that connection
dies the lock is gone too, so the heartbeat thread sets ``lost`` and the
handler stops at its next stage boundary.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import text

from auto_ontology.dal.session import get_engine

logger = logging.getLogger(__name__)

#: Fixed key (not ``hash()``, which changes per process). "AOSCHEMA" in hex-ish.
WRITE_LOCK_KEY = 0x414F5343_48454D41

HEARTBEAT_INTERVAL_S = 30.0


class LockLost(RuntimeError):
    """The lock connection died; another process may now hold the lock."""


class LockBusy(RuntimeError):
    """The lock is held elsewhere and the caller asked not to wait."""


class WriteLock:
    def __init__(self, connection) -> None:  # noqa: ANN001 - SQLAlchemy Connection
        self._connection = connection
        self._run_id: str | None = None
        self._stage: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._mutex = threading.Lock()
        self.lost = False

    def attach_run(self, run_id: str) -> None:
        """Start heartbeating ``run_id`` over the lock connection."""
        self._run_id = run_id
        self._stage = None
        self._beat()
        if self._thread is None:
            # One heartbeat thread per lock; a lock held across several runs
            # (a scheduled pass over every connection) just switches the run id.
            self._thread = threading.Thread(
                target=self._loop, daemon=True, name="schema-ingest-heartbeat"
            )
            self._thread.start()

    def stage(self, name: str) -> None:
        """Record the stage and fail fast if the lock was lost."""
        self._stage = name
        self.check()
        self._beat()

    def check(self) -> None:
        if self.lost:
            raise LockLost("schema ingest write lock connection was lost")

    def _beat(self) -> None:
        if self._run_id is None:
            return
        from auto_ontology.dal.schema_state import heartbeat

        with self._mutex:
            try:
                heartbeat(self._run_id, self._stage, connection=self._connection)
            except Exception:  # noqa: BLE001 - recorded as lost, handler stops
                logger.exception(
                    "schema ingest heartbeat failed; treating the lock as lost"
                )
                self.lost = True

    def _loop(self) -> None:
        while not self._stop.wait(HEARTBEAT_INTERVAL_S):
            self._beat()
            if self.lost:
                return

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


@contextmanager
def write_lock(
    wait: bool = True, timeout_s: float | None = None
) -> Iterator[WriteLock]:
    """Hold the global write lock on a dedicated connection.

    ``wait=False`` raises :class:`LockBusy` at once when held elsewhere.
    ``timeout_s`` bounds the wait.
    """
    connection = get_engine().connect()
    acquired = False
    lock: WriteLock | None = None
    try:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            got = connection.execute(
                text("SELECT pg_try_advisory_lock(:k)"), {"k": WRITE_LOCK_KEY}
            ).scalar()
            connection.commit()
            if got:
                acquired = True
                break
            if not wait or (deadline is not None and time.monotonic() >= deadline):
                raise LockBusy("schema ingest write lock is held by another run")
            time.sleep(1.0)
        lock = WriteLock(connection)
        yield lock
    finally:
        if lock is not None:
            lock.close()
        if acquired:
            try:
                connection.execute(
                    text("SELECT pg_advisory_unlock(:k)"), {"k": WRITE_LOCK_KEY}
                )
                connection.commit()
            except Exception:  # noqa: BLE001 - the session ending releases it anyway
                logger.warning(
                    "failed to release schema ingest write lock", exc_info=True
                )
        connection.close()


def lock_held_elsewhere() -> bool:
    """True when another session holds the lock (for status endpoints)."""
    with get_engine().connect() as connection:
        got = connection.execute(
            text("SELECT pg_try_advisory_lock(:k)"), {"k": WRITE_LOCK_KEY}
        ).scalar()
        if got:
            connection.execute(
                text("SELECT pg_advisory_unlock(:k)"), {"k": WRITE_LOCK_KEY}
            )
        connection.commit()
        return not got
