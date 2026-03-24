"""Database helpers for the price_data ingestion toolkit."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Sequence

from ..helpers.config import PIPELINE_CONFIG

try:  # pragma: no cover - optional dependency during refactors
    import psycopg
except ImportError:  # pragma: no cover - fallback when psycopg missing
    psycopg = None


OHLCV_TABLE_NAME = "ohlcv_hourly"
OHLCV_MINUTE_TABLE_NAME = "ohlcv_minute"


class BaseStorage:
    """Common interface for concrete storage backends."""

    def get_last_timestamp(self, symbol: str) -> datetime | None:
        raise NotImplementedError

    def replace_window(self, symbol: str, rows: Sequence[tuple]) -> None:
        raise NotImplementedError

    def ensure_schema(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    # Generic, table-aware helpers (used by minute ingestion)
    def get_last_timestamp_for(self, table_name: str, symbol: str) -> datetime | None:
        raise NotImplementedError

    def replace_window_for(self, table_name: str, symbol: str, rows: Sequence[tuple]) -> None:
        raise NotImplementedError

    def list_symbols(self, table_name: str = OHLCV_TABLE_NAME) -> List[str]:
        raise NotImplementedError

    def fetch_rows(
        self,
        symbol: str | None,
        start: datetime | None,
        end: datetime | None,
        limit: int = 500,
        table_name: str = OHLCV_TABLE_NAME,
    ) -> List[dict[str, Any]]:
        raise NotImplementedError


class SqliteStorage(BaseStorage):
    """SQLite-backed storage (legacy/local development)."""

    def __init__(self, db_path: Path):
        self._path = db_path
        self._ensure_database_dir(self._path)
        self.conn = sqlite3.connect(self._path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.ensure_schema()

    @staticmethod
    def _ensure_database_dir(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def ensure_schema(self) -> None:
        # Hourly table
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {OHLCV_TABLE_NAME} (
                symbol TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                ingested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, timestamp)
            );
            """
        )
        self.conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {OHLCV_TABLE_NAME}_symbol_idx
            ON {OHLCV_TABLE_NAME} (symbol);
            """
        )
        # Minute table
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {OHLCV_MINUTE_TABLE_NAME} (
                symbol TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                ingested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, timestamp)
            );
            """
        )
        self.conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {OHLCV_MINUTE_TABLE_NAME}_symbol_idx
            ON {OHLCV_MINUTE_TABLE_NAME} (symbol);
            """
        )
        self.conn.commit()

    def get_last_timestamp(self, symbol: str) -> datetime | None:
        cursor = self.conn.execute(
            f"SELECT MAX(timestamp) FROM {OHLCV_TABLE_NAME} WHERE symbol = ?;", (symbol,)
        )
        row = cursor.fetchone()
        return row[0] if row and row[0] is not None else None

    def replace_window(self, symbol: str, rows: Sequence[tuple]) -> None:
        if not rows:
            return
        window_start = rows[0][1]
        self.conn.execute(
            f"""
            DELETE FROM {OHLCV_TABLE_NAME}
            WHERE symbol = ? AND timestamp >= ?;
            """,
            (symbol, window_start),
        )
        self.conn.executemany(
            f"""
            INSERT INTO {OHLCV_TABLE_NAME}
                (symbol, timestamp, open, high, low, close, volume, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(symbol, timestamp) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume,
                ingested_at = CURRENT_TIMESTAMP;
            """,
            rows,
        )
        self.conn.commit()

    # Table-aware versions
    def get_last_timestamp_for(self, table_name: str, symbol: str) -> datetime | None:
        cursor = self.conn.execute(
            f"SELECT MAX(timestamp) FROM {table_name} WHERE symbol = ?;", (symbol,)
        )
        row = cursor.fetchone()
        return row[0] if row and row[0] is not None else None

    def replace_window_for(self, table_name: str, symbol: str, rows: Sequence[tuple]) -> None:
        if not rows:
            return
        window_start = rows[0][1]
        self.conn.execute(
            f"DELETE FROM {table_name} WHERE symbol = ? AND timestamp >= ?;",
            (symbol, window_start),
        )
        self.conn.executemany(
            f"""
            INSERT INTO {table_name}
                (symbol, timestamp, open, high, low, close, volume, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(symbol, timestamp) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume,
                ingested_at = CURRENT_TIMESTAMP;
            """,
            rows,
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def list_symbols(self, table_name: str = OHLCV_TABLE_NAME) -> List[str]:
        cursor = self.conn.execute(
            f"SELECT DISTINCT symbol FROM {table_name} ORDER BY symbol;"
        )
        return [row[0] for row in cursor.fetchall()]

    def fetch_rows(
        self,
        symbol: str | None,
        start: datetime | None,
        end: datetime | None,
        limit: int = 500,
        table_name: str = OHLCV_TABLE_NAME,
    ) -> List[dict[str, Any]]:
        clauses = ["1=1"]
        params: list[Any] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if start:
            clauses.append("timestamp >= ?")
            params.append(start)
        if end:
            clauses.append("timestamp <= ?")
            params.append(end)

        query = f"""
            SELECT symbol,
                   CAST(timestamp AS TEXT) AS timestamp,
                   open, high, low, close, volume
            FROM {table_name}
            WHERE {' AND '.join(clauses)}
            ORDER BY timestamp DESC
            LIMIT ?
        """
        params.append(limit)

        cursor = self.conn.execute(query, tuple(params))
        rows: List[dict[str, Any]] = []
        for row in cursor.fetchall():
            ts = row[1]
            if isinstance(ts, str):
                try:
                    ts_value = datetime.fromisoformat(ts)
                except ValueError:
                    ts_value = ts
            else:
                ts_value = ts
            rows.append(
                {
                    "symbol": row[0],
                    "timestamp": ts_value,
                    "open": row[2],
                    "high": row[3],
                    "low": row[4],
                    "close": row[5],
                    "volume": row[6],
                }
            )
        return rows


class PostgresStorage(BaseStorage):
    """PostgreSQL-backed storage for shared timeseries access."""

    def __init__(self, db_url: str):
        if psycopg is None:  # pragma: no cover - guard during optional dependency installs
            raise RuntimeError(
                "The psycopg package is required for PostgreSQL support. "
                "Install extras via `pip install price_data[postgres]`."
            )
        self.conn = psycopg.connect(db_url)
        self.conn.autocommit = False
        self.ensure_schema()

    def ensure_schema(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {OHLCV_TABLE_NAME} (
                    symbol TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (symbol, timestamp)
                );
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {OHLCV_MINUTE_TABLE_NAME} (
                    symbol TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (symbol, timestamp)
                );
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {OHLCV_TABLE_NAME}_symbol_idx
                ON {OHLCV_TABLE_NAME} (symbol, timestamp DESC);
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {OHLCV_MINUTE_TABLE_NAME}_symbol_idx
                ON {OHLCV_MINUTE_TABLE_NAME} (symbol, timestamp DESC);
                """
            )
        self.conn.commit()

    def get_last_timestamp(self, symbol: str) -> datetime | None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT MAX(timestamp) FROM {OHLCV_TABLE_NAME} WHERE symbol = %s;",
                (symbol,),
            )
            row = cur.fetchone()
        return row[0] if row and row[0] is not None else None

    def replace_window(self, symbol: str, rows: Sequence[tuple]) -> None:
        if not rows:
            return
        window_start = rows[0][1]
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                DELETE FROM {OHLCV_TABLE_NAME}
                WHERE symbol = %s AND timestamp >= %s;
                """,
                (symbol, window_start),
            )
            cur.executemany(
                f"""
                INSERT INTO {OHLCV_TABLE_NAME}
                    (symbol, timestamp, open, high, low, close, volume, ingested_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT(symbol, timestamp) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    ingested_at = NOW();
                """,
                rows,
            )
        self.conn.commit()

    # Table-aware versions
    def get_last_timestamp_for(self, table_name: str, symbol: str) -> datetime | None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT MAX(timestamp) FROM {table_name} WHERE symbol = %s;",
                (symbol,),
            )
            row = cur.fetchone()
        return row[0] if row and row[0] is not None else None

    def replace_window_for(self, table_name: str, symbol: str, rows: Sequence[tuple]) -> None:
        if not rows:
            return
        window_start = rows[0][1]
        with self.conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {table_name} WHERE symbol = %s AND timestamp >= %s;",
                (symbol, window_start),
            )
            cur.executemany(
                f"""
                INSERT INTO {table_name}
                    (symbol, timestamp, open, high, low, close, volume, ingested_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT(symbol, timestamp) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    ingested_at = NOW();
                """,
                rows,
            )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def list_symbols(self, table_name: str = OHLCV_TABLE_NAME) -> List[str]:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT DISTINCT symbol FROM {table_name} ORDER BY symbol;"
            )
            return [row[0] for row in cur.fetchall()]

    def fetch_rows(
        self,
        symbol: str | None,
        start: datetime | None,
        end: datetime | None,
        limit: int = 500,
        table_name: str = OHLCV_TABLE_NAME,
    ) -> List[dict[str, Any]]:
        clauses = ["1=1"]
        params: list[Any] = []
        if symbol:
            clauses.append("symbol = %s")
            params.append(symbol)
        if start:
            clauses.append("timestamp >= %s")
            params.append(start)
        if end:
            clauses.append("timestamp <= %s")
            params.append(end)
        params.append(limit)

        query = f"""
            SELECT symbol, timestamp, open, high, low, close, volume
            FROM {table_name}
            WHERE {' AND '.join(clauses)}
            ORDER BY timestamp DESC
            LIMIT %s;
        """

        with self.conn.cursor() as cur:
            cur.execute(query, tuple(params))
            results = cur.fetchall()

        rows: List[dict[str, Any]] = []
        for row in results:
            ts = row[1]
            rows.append(
                {
                    "symbol": row[0],
                    "timestamp": ts,
                    "open": row[2],
                    "high": row[3],
                    "low": row[4],
                    "close": row[5],
                    "volume": row[6],
                }
            )
        return rows


def prepare_ohlcv_rows(
    symbol: str, values: Iterable[tuple[datetime, float | None, float | None, float | None, float | None, float | None]]
) -> List[tuple]:
    """Prepare raw OHLCV values for insertion."""
    rows: List[tuple] = []
    for ts, open_, high, low, close, volume in values:
        rows.append((symbol, ts, open_, high, low, close, volume))
    return rows


@contextmanager
def get_storage(db_url: str | None = None, db_path: Path | None = None):
    """Yield a storage backend configured for the pipeline."""
    # Respect explicit arguments (including an empty string to force SQLite),
    # and only fall back to environment defaults when parameters are omitted.
    db_url = PIPELINE_CONFIG.db_url if db_url is None else db_url
    db_path = PIPELINE_CONFIG.db_path if db_path is None else Path(db_path)
    backend: BaseStorage
    if db_url:
        backend = PostgresStorage(db_url)
    else:
        backend = SqliteStorage(db_path)
    try:
        yield backend
    finally:
        backend.close()


__all__ = [
    "get_storage",
    "OHLCV_TABLE_NAME",
    "OHLCV_MINUTE_TABLE_NAME",
    "PostgresStorage",
    "SqliteStorage",
    "prepare_ohlcv_rows",
]
