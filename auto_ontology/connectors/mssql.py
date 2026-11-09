"""MSSQL (SQL Server) connector for Auto Ontology.

Provides integration with Microsoft SQL Server databases using pymssql.

Author: WOW Press Team
Date: 2026-11-04
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd

from auto_ontology.connectors.limited_result import LimitedResult, collect_limited

try:
    import pymssql
except ImportError:
    pymssql = None  # type: ignore

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)

# Rows fetched per round trip when reading a result incrementally.
_FETCH_BATCH_ROWS = 1000


class MSSQLQueryTimeoutError(RuntimeError):
    """The statement ran past ``timeout_s`` and was cancelled.

    pymssql reports a ``query_timeout`` expiry as "TDS server connection timed
    out", which reads as a connectivity failure. Callers that classify errors
    (``is_infrastructure_error``) would then give up instead of rewriting a slow
    query, so the timeout is re-raised with a message that says what happened.
    """


def _as_query_timeout(
    exc: Exception, timeout_s: Optional[float], elapsed: float
) -> Exception:
    """Map a pymssql timeout during a capped statement to MSSQLQueryTimeoutError."""
    if not timeout_s or elapsed < float(timeout_s) - 1:
        return exc
    if "timed out" not in str(exc).lower():
        return exc
    return MSSQLQueryTimeoutError(
        f"The query was cancelled after exceeding the {int(timeout_s)}s execution "
        "limit. The SQL is valid but too slow; rewrite it to read fewer rows "
        "(narrow the period, filter earlier, aggregate in a summary table)."
    )


def _column_names(description: Any) -> list[str]:
    """Build unique DataFrame column names from a DB-API cursor description.

    SQL Server leaves unaliased expressions (e.g. ``SUM(amt)``) unnamed, so
    those get positional names; repeated names get a numeric suffix.
    """
    names: list[str] = []
    seen: dict[str, int] = {}
    for idx, col in enumerate(description, start=1):
        name = col[0] or f"column_{idx}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    return names


def _parse_connection_string(conn_str: str) -> dict[str, Any]:
    """Parse MSSQL connection string.

    Supported formats:
        mssql://user:password@host:port/database
        mssql+pymssql://user:password@host:port/database

    Args:
        conn_str: MSSQL connection URL

    Returns:
        Dictionary with connection parameters

    Raises:
        ValueError: If connection string is missing required parts
    """
    parsed = urlparse(conn_str)

    if not parsed.hostname:
        raise ValueError("MSSQL connection string must include hostname")
    if not parsed.path or parsed.path == "/":
        raise ValueError("MSSQL connection string must include database name")

    port = parsed.port or 1433
    database = parsed.path.lstrip("/")

    # Parse query parameters
    query_params = parse_qs(parsed.query)

    config = {
        "server": parsed.hostname,
        "port": port,
        "database": database,
        "user": unquote(parsed.username) if parsed.username else None,
        "password": unquote(parsed.password) if parsed.password else None,
        "charset": query_params.get("charset", ["utf8"])[0],
        "tds_version": query_params.get("tds_version", ["7.3"])[0],
    }

    return config


class MSSQLDatabase:
    """Concrete SQLDatabase backed by ``pymssql``.

    This connector supports Microsoft SQL Server 2008 and later versions.
    ``timeout_s`` caps both lock waits (``SET LOCK_TIMEOUT``) and query run
    time (the pymssql connection's ``query_timeout``).

    Attributes:
        supports_statement_timeout: Whether execute() supports timeout_s parameter
        table_filter: Optional filter for table/schema selection

    Example:
        >>> conn_str = "mssql://user:password@host:1433/database"
        >>> with MSSQLDatabase(conn_str) as db:
        ...     tables = db.get_tables()
        ...     print(f"Found {len(tables)} tables")
    """

    supports_statement_timeout = True
    table_filter = None

    def __init__(
        self,
        connection_string: str,
        *,
        table_filter: Optional[Any] = None,
    ) -> None:
        """Initialize MSSQL connection.

        Args:
            connection_string: MSSQL connection URL
            table_filter: Optional filter for table/schema selection

        Raises:
            ImportError: If pymssql is not installed
        """
        if pymssql is None:
            raise ImportError(
                "pymssql is required for MSSQL connector. "
                "Install with: pip install pymssql"
            )

        self.table_filter = table_filter
        self._config = _parse_connection_string(connection_string)
        self._conn: Optional[pymssql.Connection] = None
        self._connection_string = connection_string

    def _get_connection(self) -> "pymssql.Connection":
        """Get or create database connection.

        Returns:
            Active pymssql connection
        """
        if self._conn is None:
            logger.info(
                f"Connecting to MSSQL: {self._config['server']}:{self._config['port']}"
                f"/{self._config['database']}"
            )
            self._conn = pymssql.connect(
                server=self._config["server"],
                port=self._config["port"],
                user=self._config["user"],
                password=self._config["password"],
                database=self._config["database"],
                charset=self._config["charset"],
                tds_version=self._config["tds_version"],
                as_dict=True,
            )
        return self._conn

    def _apply_timeout(
        self, conn: Any, cursor: Any, timeout_s: Optional[float]
    ) -> None:
        """Cap lock waits and query run time for the next statement.

        ``SET LOCK_TIMEOUT`` alone only bounds waiting for locks; a long scan
        would run unbounded. ``query_timeout`` on the underlying ``_mssql``
        connection bounds the statement itself. The connection is cached, so
        the value is set on every call (``None`` clears it to 0 = no limit).
        """
        if timeout_s is not None:
            cursor.execute(f"SET LOCK_TIMEOUT {int(timeout_s * 1000)}")
        raw = getattr(conn, "_conn", None)
        if raw is not None and hasattr(raw, "query_timeout"):
            try:
                raw.query_timeout = max(1, int(timeout_s)) if timeout_s else 0
            except Exception as exc:  # noqa: BLE001 — never fail a query over the cap
                logger.warning("Could not set MSSQL query_timeout: %s", exc)

    def _discard_if_dead(self, conn: Any) -> None:
        """Drop the cached connection when the driver has marked it dead.

        A query timeout (``query_timeout``) or a network drop leaves pymssql's
        DBPROCESS dead; reusing it fails every later statement with "Not
        connected to any MS SQL server" until the process restarts. Forget it so
        the next call reconnects.
        """
        raw = getattr(conn, "_conn", None)
        if raw is None or getattr(raw, "connected", True):
            return
        logger.warning("MSSQL connection is dead; reconnecting on next use")
        try:
            conn.close()
        except Exception:  # noqa: BLE001 — already dead
            pass
        if self._conn is conn:
            self._conn = None

    @staticmethod
    def _cancel_pending(conn: Any) -> None:
        """Ask the server to stop sending rows we no longer read."""
        raw = getattr(conn, "_conn", None)
        if raw is None or not hasattr(raw, "cancel"):
            return
        try:
            raw.cancel()
        except Exception as exc:  # noqa: BLE001 — best effort
            logger.warning("MSSQL cancel after row limit failed: %s", exc)

    @property
    def dialect(self) -> str:
        """Return sqlglot dialect name for T-SQL."""
        return "tsql"

    @property
    def database_name(self) -> str:
        """Return connected database name."""
        return self._config["database"]

    def execute(
        self,
        sql: str,
        parameters: Optional[list] = None,
        *,
        timeout_s: Optional[int] = None,
    ) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame.

        Args:
            sql: SQL query to execute
            parameters: Optional query parameters (positional)
            timeout_s: Optional statement timeout in seconds

        Returns:
            Query results as pandas DataFrame

        Raises:
            pymssql.Error: On database errors
        """
        conn = self._get_connection()
        # Tuple rows, not as_dict: pymssql's dict cursor raises
        # ColumnsWithoutNamesError on unaliased expressions like SUM(x).
        cursor = conn.cursor(as_dict=False)
        started = time.monotonic()

        try:
            self._apply_timeout(conn, cursor, timeout_s)
            started = time.monotonic()

            if parameters:
                cursor.execute(sql, tuple(parameters))
            else:
                cursor.execute(sql)

            # Fetch results if query returns data
            if cursor.description:
                columns = _column_names(cursor.description)
                rows = cursor.fetchall()
                return pd.DataFrame.from_records(rows, columns=columns)

            conn.commit()
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"MSSQL execute error: {e}")
            self._discard_if_dead(conn)
            mapped = _as_query_timeout(e, timeout_s, time.monotonic() - started)
            if mapped is not e:
                raise mapped from e
            raise
        finally:
            cursor.close()

    def execute_limited(
        self,
        sql: str,
        *,
        preview_rows: int,
        max_rows: int,
        timeout_s: Optional[float] = None,
    ) -> LimitedResult:
        """Run *sql*, keep the first *preview_rows* rows and count the rest.

        Rows are read with ``fetchmany`` so at most one batch plus the preview
        is in memory. Reading stops at ``max_rows + 1`` rows; the rest of the
        result is cancelled on the server. The SQL is not rewritten.
        """
        conn = self._get_connection()
        cursor = conn.cursor(as_dict=False)
        started = time.monotonic()
        stopped_early = False
        try:
            self._apply_timeout(conn, cursor, timeout_s)
            started = time.monotonic()
            cursor.execute(sql)
            if not cursor.description:
                conn.commit()
                return LimitedResult(
                    frame=pd.DataFrame(), total_count=0, over_limit=False
                )

            columns = _column_names(cursor.description)

            def batches() -> Any:
                while True:
                    rows = cursor.fetchmany(_FETCH_BATCH_ROWS)
                    if not rows:
                        return
                    yield rows

            result = collect_limited(
                batches(), columns, preview_rows=preview_rows, max_rows=max_rows
            )
            stopped_early = result.over_limit
            return result
        except Exception as e:
            logger.error(f"MSSQL execute error: {e}")
            self._discard_if_dead(conn)
            mapped = _as_query_timeout(e, timeout_s, time.monotonic() - started)
            if mapped is not e:
                raise mapped from e
            raise
        finally:
            if stopped_early:
                self._cancel_pending(conn)
            cursor.close()

    def get_tables(self) -> pd.DataFrame:
        """Get all tables and views in the database.

        Returns:
            DataFrame with columns: table_schema, table_name, table_type
        """
        sql = """
        SELECT
            s.name AS table_schema,
            t.name AS table_name,
            CASE t.type
                WHEN 'U' THEN 'BASE TABLE'
                WHEN 'V' THEN 'VIEW'
                ELSE t.type_desc
            END AS table_type
        FROM sys.tables t
        INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
        WHERE t.is_ms_shipped = 0

        UNION ALL

        SELECT
            s.name AS table_schema,
            v.name AS table_name,
            'VIEW' AS table_type
        FROM sys.views v
        INNER JOIN sys.schemas s ON v.schema_id = s.schema_id
        WHERE v.is_ms_shipped = 0

        ORDER BY table_schema, table_name
        """
        return self.execute(sql)

    def get_columns(self) -> pd.DataFrame:
        """Get all columns for all tables and views.

        Returns:
            DataFrame with columns: table_schema, table_name, column_name,
                                   data_type, is_nullable, ordinal_position
        """
        sql = """
        SELECT
            s.name AS table_schema,
            t.name AS table_name,
            c.name AS column_name,
            TYPE_NAME(c.user_type_id) AS data_type,
            CASE c.is_nullable WHEN 1 THEN 'YES' ELSE 'NO' END AS is_nullable,
            c.column_id AS ordinal_position
        FROM sys.columns c
        INNER JOIN sys.tables t ON c.object_id = t.object_id
        INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
        WHERE t.is_ms_shipped = 0

        UNION ALL

        SELECT
            s.name AS table_schema,
            v.name AS table_name,
            c.name AS column_name,
            TYPE_NAME(c.user_type_id) AS data_type,
            CASE c.is_nullable WHEN 1 THEN 'YES' ELSE 'NO' END AS is_nullable,
            c.column_id AS ordinal_position
        FROM sys.columns c
        INNER JOIN sys.views v ON c.object_id = v.object_id
        INNER JOIN sys.schemas s ON v.schema_id = s.schema_id
        WHERE v.is_ms_shipped = 0

        ORDER BY table_schema, table_name, ordinal_position
        """
        return self.execute(sql)

    def get_pks(self) -> pd.DataFrame:
        """Get primary key information for all tables.

        Returns:
            DataFrame with columns: table_schema, table_name, column_name,
                                   ordinal_position
        """
        sql = """
        SELECT
            s.name AS table_schema,
            t.name AS table_name,
            c.name AS column_name,
            ic.key_ordinal AS ordinal_position
        FROM sys.index_columns ic
        INNER JOIN sys.indexes i ON ic.object_id = i.object_id AND ic.index_id = i.index_id
        INNER JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
        INNER JOIN sys.tables t ON i.object_id = t.object_id
        INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
        WHERE i.is_primary_key = 1
        ORDER BY table_schema, table_name, ordinal_position
        """
        return self.execute(sql)

    def get_fks(self) -> pd.DataFrame:
        """Get foreign key information for all tables.

        Returns:
            DataFrame with columns: table_schema, table_name, column_name,
                                   referenced_schema, referenced_table,
                                   referenced_column
        """
        sql = """
        SELECT
            s.name AS table_schema,
            t.name AS table_name,
            c.name AS column_name,
            rs.name AS referenced_schema,
            rt.name AS referenced_table,
            rc.name AS referenced_column
        FROM sys.foreign_key_columns fkc
        INNER JOIN sys.tables t ON fkc.parent_object_id = t.object_id
        INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
        INNER JOIN sys.columns c ON fkc.parent_object_id = c.object_id
            AND fkc.parent_column_id = c.column_id
        INNER JOIN sys.tables rt ON fkc.referenced_object_id = rt.object_id
        INNER JOIN sys.schemas rs ON rt.schema_id = rs.schema_id
        INNER JOIN sys.columns rc ON fkc.referenced_object_id = rc.object_id
            AND fkc.referenced_column_id = rc.column_id
        ORDER BY table_schema, table_name, column_name
        """
        return self.execute(sql)

    def get_views(self) -> pd.DataFrame:
        """Get view definitions.

        Returns:
            DataFrame with columns: table_schema, table_name, view_definition
        """
        sql = """
        SELECT
            s.name AS table_schema,
            v.name AS table_name,
            m.definition AS view_definition
        FROM sys.views v
        INNER JOIN sys.schemas s ON v.schema_id = s.schema_id
        INNER JOIN sys.sql_modules m ON v.object_id = m.object_id
        WHERE v.is_ms_shipped = 0
        ORDER BY table_schema, table_name
        """
        return self.execute(sql)

    def get_queries(self, hours: int = 24) -> pd.DataFrame:
        """Get recent query history.

        Attempts to use Query Store first, falls back to dm_exec_query_stats.

        Args:
            hours: Number of hours to look back

        Returns:
            DataFrame with columns: end_time, query_text
        """
        # Try Query Store first
        query_store_sql = f"""
        SELECT TOP 1000
            qsp.last_execution_time AS end_time,
            qt.query_sql_text AS query_text
        FROM sys.query_store_plan qsp
        INNER JOIN sys.query_store_query q ON qsp.query_id = q.query_id
        INNER JOIN sys.query_store_query_text qt ON q.query_text_id = qt.query_text_id
        WHERE qsp.last_execution_time >= DATEADD(HOUR, -{hours}, GETDATE())
        ORDER BY qsp.last_execution_time DESC
        """
        try:
            return self.execute(query_store_sql)
        except Exception as e:
            logger.warning(f"Query Store not available, using fallback: {e}")

        # Fallback to dm_exec_query_stats
        fallback_sql = f"""
        SELECT TOP 1000
            qs.last_execution_time AS end_time,
            SUBSTRING(st.text, 1, 4000) AS query_text
        FROM sys.dm_exec_query_stats qs
        CROSS APPLY sys.dm_exec_sql_text(qs.sql_handle) st
        WHERE qs.last_execution_time >= DATEADD(HOUR, -{hours}, GETDATE())
        ORDER BY qs.last_execution_time DESC
        """
        try:
            return self.execute(fallback_sql)
        except Exception as e:
            logger.warning(f"Query history not available: {e}")
            return pd.DataFrame(columns=["end_time", "query_text"])

    def get_table_row_counts(self) -> pd.DataFrame:
        """Get approximate row counts for all tables.

        Uses sys.dm_db_partition_stats for fast estimation.

        Returns:
            DataFrame with columns: table_schema, table_name, row_count
        """
        sql = """
        SELECT
            s.name AS table_schema,
            t.name AS table_name,
            SUM(p.rows) AS row_count
        FROM sys.tables t
        INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
        INNER JOIN sys.dm_db_partition_stats p ON t.object_id = p.object_id
        WHERE p.index_id IN (0, 1)  -- Heap or clustered index
          AND t.is_ms_shipped = 0
        GROUP BY s.name, t.name
        ORDER BY row_count DESC
        """
        return self.execute(sql)

    def sample_data(
        self, table: str, schema: str = "dbo", limit: int = 100
    ) -> pd.DataFrame:
        """Get sample rows from a table.

        Args:
            table: Table name
            schema: Schema name (default: dbo)
            limit: Maximum rows to return

        Returns:
            DataFrame with sample data
        """
        qualified = self.qualify(schema, table)
        sql = f"SELECT TOP {limit} * FROM {qualified}"
        return self.execute(sql)

    def qualify(self, schema: Optional[str], table: str) -> str:
        """Return fully qualified table reference.

        Args:
            schema: Schema name (or None for dbo)
            table: Table name

        Returns:
            Qualified table reference [schema].[table]
        """
        schema = schema or "dbo"
        return f"[{schema}].[{table}]"

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("MSSQL connection closed")

    def __enter__(self) -> "MSSQLDatabase":
        """Enter context manager."""
        return self

    def __exit__(self, *args) -> None:
        """Exit context manager, closing connection."""
        self.close()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"MSSQLDatabase(server={self._config['server']}, "
            f"database={self._config['database']})"
        )
