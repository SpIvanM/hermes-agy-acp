from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Any

from hermes_agy_acp.config import get_settings


class SessionRegistry:
    """SQLite-backed persistent registry mapping Hermes sessions to ACP sessions."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is not None:
            self.db_path = Path(db_path).expanduser()
        else:
            self.db_path = get_settings().state_db_path

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    hermes_session_id TEXT PRIMARY KEY,
                    acp_session_id TEXT NOT NULL,
                    rehydration_summary TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_acp_session_id ON sessions(acp_session_id);"
            )
            conn.commit()

        if sys.platform != "win32" and self.db_path.exists():
            os.chmod(self.db_path, 0o600)

    def save_mapping(
        self,
        hermes_session_id: str,
        acp_session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a session mapping."""
        meta_json = json.dumps(metadata) if metadata else None
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (hermes_session_id, acp_session_id, metadata, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(hermes_session_id) DO UPDATE SET
                    acp_session_id = excluded.acp_session_id,
                    metadata = excluded.metadata,
                    updated_at = CURRENT_TIMESTAMP;
                """,
                (hermes_session_id, acp_session_id, meta_json),
            )
            conn.commit()

        if sys.platform != "win32" and self.db_path.exists():
            os.chmod(self.db_path, 0o600)

    def get_acp_session_id(self, hermes_session_id: str) -> str | None:
        """Get ACP session ID corresponding to Hermes session ID."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT acp_session_id FROM sessions WHERE hermes_session_id = ?",
                (hermes_session_id,),
            )
            row = cursor.fetchone()
            return str(row["acp_session_id"]) if row else None

    def get_hermes_session_id(self, acp_session_id: str) -> str | None:
        """Get Hermes session ID corresponding to ACP session ID."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT hermes_session_id FROM sessions WHERE acp_session_id = ?",
                (acp_session_id,),
            )
            row = cursor.fetchone()
            return str(row["hermes_session_id"]) if row else None

    def reset_session(self, hermes_session_id: str) -> bool:
        """Delete mapping for Hermes session ID to handle /reset."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE hermes_session_id = ?",
                (hermes_session_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def set_rehydration_summary(self, hermes_session_id: str, summary: str) -> None:
        """Store conversation summary for session rehydration."""
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET rehydration_summary = ?, updated_at = CURRENT_TIMESTAMP
                WHERE hermes_session_id = ?
                """,
                (summary, hermes_session_id),
            )
            conn.commit()

    def get_rehydration_summary(self, hermes_session_id: str) -> str | None:
        """Retrieve stored conversation summary for rehydration."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT rehydration_summary FROM sessions WHERE hermes_session_id = ?",
                (hermes_session_id,),
            )
            row = cursor.fetchone()
            return str(row["rehydration_summary"]) if row and row["rehydration_summary"] else None

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all active session mappings."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT hermes_session_id, acp_session_id, rehydration_summary, metadata, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            )
            rows = cursor.fetchall()
            results: list[dict[str, Any]] = []
            for r in rows:
                meta = json.loads(r["metadata"]) if r["metadata"] else None
                results.append(
                    {
                        "hermes_session_id": r["hermes_session_id"],
                        "acp_session_id": r["acp_session_id"],
                        "rehydration_summary": r["rehydration_summary"],
                        "metadata": meta,
                        "created_at": r["created_at"],
                        "updated_at": r["updated_at"],
                    }
                )
            return results

    def close(self) -> None:
        """Close registry resources."""
