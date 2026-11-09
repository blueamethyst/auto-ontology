"""MSSQL (SQL Server) connector for Auto Ontology.

Provides integration with Microsoft SQL Server databases using pymssql.

Author: WOW Press Team
Date: 2026-11-04
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd

try:
    import pymssql
except ImportError:
    pymssql = None  # type: ignore

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


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
    Statement timeout is supported via SET LOCK_TIMEOUT.

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
        cursor = conn.cursor(as_dict=True)

        try:
            # Set statement timeout if specified
            if timeout_s is not None:
                cursor.execute(f"SET LOCK_TIMEOUT {timeout_s * 1000}")

            if parameters:
                cursor.execute(sql, tuple(parameters))
            else:
                cursor.execute(sql)

            # Fetch results if query returns data
            if cursor.description:
                rows = cursor.fetchall()
                return pd.DataFrame(rows)

            conn.commit()
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"MSSQL execute error: {e}")
            raise
        finally:
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

    def sample_data(self, table: str, schema: str = "dbo", limit: int = 100) -> pd.DataFrame:
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
