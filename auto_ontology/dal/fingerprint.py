# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fingerprint of everything text-to-SQL reads from the store.

A co-located caller (DABI) caches "question -> generated SQL". The SQL a
question produces depends on the catalog, the semantic layer, example SQL, the
glossary, custom prompts, agent settings, model settings and the code itself.
This module hashes all of that so the caller can put one ``version`` in its
cache key: any change gives a new key and old entries are simply never read
again.

* ``content_hash`` hashes the full rows of every store table the SQL pipeline
  reads (see :data:`CONTENT_TABLES`), minus columns that change on every ingest
  without changing meaning (``pulled``, ``last_seen``...). Rows are serialised
  with ``to_jsonb`` so key order and types are stable.
* ``config_hash`` hashes the frontend settings tables (glossary, prompts,
  configurations) and the non-secret inference environment.
* ``structure_hash`` hashes only ``target_db``'s schema shape (tables, columns,
  type names, nullability, PKs, FKs). A caller caching "SQL -> rows" uses it,
  because results only depend on the shape, not on the semantic layer.

Everything is read in one ``REPEATABLE READ`` transaction so the hashes come
from one consistent snapshot.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text

from auto_ontology.dal.session import get_engine
from auto_ontology.infra.postgres import FRONTEND_SCHEMA

#: Store tables the text-to-SQL pipeline reads, with the columns left out of the
#: hash because they move on every ingest/rule pass without changing meaning.
CONTENT_TABLES: dict[str, tuple[str, ...]] = {
    "catalog_database": ("pulled", "created"),
    "catalog_schema": (),
    "catalog_table": (),
    "catalog_column": (),
    "column__foreign_key": ("last_seen",),
    "table__join": (),
    "sql_query": ("last_seen", "last_query_timestamp", "query_count", "total_counter"),
    "sql_query__table": (),
    "sql_query__column": (),
    "term": (),
    "table__term": (),
    "column_attribute": (),
    "column_attribute__term": (),
    "column__has_attribute": (),
    "column__semantic_fk": (),
    "sql_attribute": (),
    "sql_attribute__term": (),
    "sql_attribute__sql": (),
    "custom_analysis": (),
    "custom_analysis__sql": (),
    "pql_analysis": (),
    "text_attribute": (),
    "analysis": (),
    "tag": (),
    "tag_target": ("tagged",),
    "rule": (),
    "rule__tag": (),
    "zone": (),
    "zone_target": (),
}

#: Frontend-owned settings the chat worker reads on every question.
CONFIG_TABLES: dict[str, tuple[str, ...]] = {
    "acronyms": (),
    "prompts": (),
    "configurations": (),
}

#: Environment prefixes that steer generation. Names that look like secrets are
#: always dropped, whatever the prefix.
_CONFIG_ENV_PREFIXES = (
    "REASONING_",
    "NON_REASONING_",
    "DEFAULT_MODELS_",
    "EMBED_",
    "LLM_",
    "SQL_",
    "TEXT_TO_SQL",
    "KUMO_",
)
_SECRET_MARKERS = ("KEY", "SECRET", "PASSWORD", "TOKEN", "CREDENTIAL")


def _table_digest_sql(
    table: str, excluded: tuple[str, ...], schema: str | None = None
) -> str:
    qualified = f'"{schema}"."{table}"' if schema else f'"{table}"'
    minus = "".join(f" - '{column}'" for column in excluded)
    return (
        f"SELECT count(*) AS n, "
        f"md5(coalesce(string_agg(r, E'\\n' ORDER BY r), '')) AS h "
        f"FROM (SELECT (to_jsonb(t){minus})::text AS r FROM {qualified} t) x"
    )


_STRUCTURE_SQL = """
SELECT md5(coalesce(string_agg(r, E'\\n' ORDER BY r), '')) AS h, count(*) AS n FROM (
  SELECT concat_ws('|', 'c', s.name, t.name, c.name, c.data_type, c.is_nullable) AS r
    FROM catalog_database d
    JOIN catalog_schema s ON s.database_id = d.id
    JOIN catalog_table t ON t.schema_id = s.id
    JOIN catalog_column c ON c.table_id = t.id
   WHERE d.name = :db
  UNION ALL
  SELECT concat_ws('|', 't', s.name, t.name, t.table_type, t.pk::text)
    FROM catalog_database d
    JOIN catalog_schema s ON s.database_id = d.id
    JOIN catalog_table t ON t.schema_id = s.id
   WHERE d.name = :db
) x
"""


def _existing_tables(connection: Any, schema: str) -> set[str]:
    rows = connection.execute(
        text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = :s"
        ),
        {"s": schema},
    )
    return {r[0] for r in rows}


def config_env_hash(environ: dict[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    items = sorted(
        (k, v)
        for k, v in env.items()
        if k.startswith(_CONFIG_ENV_PREFIXES)
        and not any(marker in k.upper() for marker in _SECRET_MARKERS)
    )
    return hashlib.sha256(repr(items).encode("utf-8")).hexdigest()


def build_version() -> str:
    """Git commit baked into the image (``AO_BUILD_VERSION``), or ``unknown``."""
    return os.environ.get("AO_BUILD_VERSION", "").strip() or "unknown"


def compute_fingerprint(target_db: str) -> dict[str, Any]:
    """Hash the store state text-to-SQL depends on, from one snapshot."""
    content = hashlib.sha256()
    config = hashlib.sha256()
    counts: dict[str, int] = {}
    with (
        get_engine()
        .connect()
        .execution_options(isolation_level="REPEATABLE READ") as connection
    ):
        with connection.begin():
            public_tables = _existing_tables(connection, "public")
            for table, excluded in CONTENT_TABLES.items():
                if table not in public_tables:
                    continue
                row = connection.execute(text(_table_digest_sql(table, excluded))).one()
                counts[table] = int(row.n)
                content.update(f"{table}:{row.n}:{row.h};".encode())

            frontend_tables = _existing_tables(connection, FRONTEND_SCHEMA)
            for table, excluded in CONFIG_TABLES.items():
                if table not in frontend_tables:
                    continue
                row = connection.execute(
                    text(_table_digest_sql(table, excluded, FRONTEND_SCHEMA))
                ).one()
                config.update(f"{table}:{row.n}:{row.h};".encode())

            structure = connection.execute(
                text(_STRUCTURE_SQL), {"db": target_db}
            ).one()

    config.update(config_env_hash().encode())
    content_hash = content.hexdigest()
    config_hash = config.hexdigest()
    structure_hash = hashlib.sha256(
        f"{target_db}:{structure.n}:{structure.h}".encode()
    ).hexdigest()
    return {
        "content_hash": content_hash,
        "config_hash": config_hash,
        "structure_hash": structure_hash,
        "structure_rows": int(structure.n),
        "counts": counts,
        "validated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def fingerprint_version(
    build: str, config_hash: str, content_hash: str, generation: int | None
) -> str:
    raw = f"{build}|{config_hash}|{content_hash}|{generation if generation is not None else '-'}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
