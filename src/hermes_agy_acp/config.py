from __future__ import annotations

import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, Field, field_validator


def resolve_binary(binary_name: str, override_path: str | None = None) -> Path | None:
    """Resolve an executable binary cross-platform with zero hardcoded system paths.

    If override_path is provided and non-empty, checks existence directly or raises FileNotFoundError.
    Otherwise searches PATH using shutil.which, handling Windows extensions (.cmd, .exe, .bat)
    and Linux/Unix naming conventions.
    """
    if override_path:
        override = Path(override_path).expanduser()
        if not override.exists() or not override.is_file():
            raise FileNotFoundError(f"Configured binary override not found: {override_path}")
        return override.resolve()

    found = shutil.which(binary_name)
    if found:
        return Path(found).resolve()

    if sys.platform == "win32":
        for ext in [".cmd", ".exe", ".bat"]:
            if not binary_name.lower().endswith(ext):
                candidate = shutil.which(f"{binary_name}{ext}")
                if candidate:
                    return Path(candidate).resolve()
    else:
        if not binary_name.endswith(".par"):
            candidate = shutil.which(f"{binary_name}.par")
            if candidate:
                return Path(candidate).resolve()

    return None


ThinkingLevel = Literal["low", "medium", "high", "off"]
LogLevel = Literal["error", "warn", "info", "debug"]
ProxyMode = Literal["agentic_proxy", "direct"]


def _default_hermes_home() -> Path:
    val = os.getenv("HERMES_HOME")
    if val:
        return Path(val).expanduser()
    return Path.home() / ".hermes"


def _default_state_db_path() -> Path:
    val = os.getenv("AGY_STATE_DB_PATH")
    if val:
        return Path(val).expanduser()
    return _default_hermes_home() / "state_agy_acp.db"


def _default_acp_args() -> list[str]:
    val = os.getenv("AGY_ACP_ARGS")
    if val is not None:
        return val.split()
    return [] if sys.platform == "win32" else ["--uid="]


def _default_thinking_level() -> ThinkingLevel:
    val = os.getenv("AGY_THINKING_LEVEL")
    if val is not None:
        return cast(ThinkingLevel, val)
    return "high"


def _default_log_level() -> LogLevel:
    val = os.getenv("AGY_LOG_LEVEL")
    if val is not None:
        return cast(LogLevel, val)
    return "warn"


def _default_mode() -> ProxyMode:
    val = os.getenv("AGY_MODE")
    if val is not None:
        return cast(ProxyMode, val)
    return "agentic_proxy"


class Settings(BaseModel):
    """12-factor typed configuration for hermes-agy-acp."""

    agy_cli_path: str | None = Field(
        default_factory=lambda: os.getenv("AGY_CLI_PATH") or None
    )
    agy_acp_server_path: str | None = Field(
        default_factory=lambda: os.getenv("AGY_ACP_SERVER_PATH") or None
    )
    agy_acp_args: list[str] = Field(default_factory=_default_acp_args)
    agy_default_model: str = Field(
        default_factory=lambda: os.getenv("AGY_DEFAULT_MODEL", "gemini-3.8-flash-high")
    )
    agy_thinking_level: ThinkingLevel = Field(default_factory=_default_thinking_level)
    agy_log_level: LogLevel = Field(default_factory=_default_log_level)
    hermes_home: Path = Field(default_factory=_default_hermes_home)
    state_db_path: Path = Field(default_factory=_default_state_db_path)
    mode: ProxyMode = Field(default_factory=_default_mode)

    @field_validator("agy_acp_args", mode="before")
    @classmethod
    def _parse_args(cls, v: object) -> list[str]:
        if isinstance(v, str):
            return v.split()
        if isinstance(v, list):
            return [str(item) for item in v]
        return []

    def get_resolved_cli_path(self) -> Path | None:
        """Resolve agy CLI binary executable."""
        return resolve_binary("agy", self.agy_cli_path)

    def get_resolved_server_path(self) -> Path | None:
        """Resolve agy ACP server binary executable."""
        return resolve_binary("agy_acp_server", self.agy_acp_server_path)

    def get_server_command(self) -> list[str]:
        """Generate server command line with binary and arguments."""
        server = self.get_resolved_server_path()
        if not server:
            raise RuntimeError("ACP server binary not found. Set AGY_ACP_SERVER_PATH or add to PATH.")
        return [str(server), *self.agy_acp_args]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return singleton configuration settings."""
    return Settings()
