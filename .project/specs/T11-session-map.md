# Specification T11: SQLite Session Registry (session_map.py)

## 1. Overview
The `session_map.py` module manages persistent bidirectional mapping between Hermes session identifiers (`hermes_session_id`) and ACP session identifiers (`acp_session_id`).
It persists state in SQLite (`$HERMES_HOME/state_agy_acp.db`) with strict POSIX `0600` permissions, provides session rehydration context, and handles `/reset` commands.

## 2. Public Interface

### 2.1 Classes
- `SessionRegistry`:
  - `__init__(db_path: Path | str | None = None)`:
    - If `db_path` is None, defaults to `get_settings().state_db_path`.
    - Automatically creates parent directory if missing.
    - Initializes SQLite schema.
    - Ensures db file permissions are set to `0600` (`-rw-------`).
  - Methods:
    - `save_mapping(hermes_session_id: str, acp_session_id: str, metadata: dict[str, Any] | None = None) -> None`:
      - Inserts or updates mapping for `hermes_session_id`.
    - `get_acp_session_id(hermes_session_id: str) -> str | None`:
      - Returns mapped `acp_session_id` or `None`.
    - `get_hermes_session_id(acp_session_id: str) -> str | None`:
      - Returns reverse mapped `hermes_session_id` or `None`.
    - `reset_session(hermes_session_id: str) -> bool`:
      - Handles `/reset` command by clearing mapping for `hermes_session_id`.
      - Returns `True` if record was deleted, `False` if not found.
    - `set_rehydration_summary(hermes_session_id: str, summary: str) -> None`:
      - Updates or sets rehydration summary text for session.
    - `get_rehydration_summary(hermes_session_id: str) -> str | None`:
      - Retrieves stored rehydration summary text or `None`.
    - `list_sessions() -> list[dict[str, Any]]`:
      - Returns list of active session dict records.
    - `close() -> None`:
      - Closes connection if open.

## 3. Invariants & Security
- The database file must always have file mode `0600` on POSIX systems.
- Concurrency-safe against race conditions via connection isolation or locks.
- Zero hardcoded paths.
