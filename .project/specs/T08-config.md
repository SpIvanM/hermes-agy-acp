# Specification T08: Configuration Subsystem (config.py)

## 1. Overview
The config.py module provides 12-factor typed configuration management for hermes-agy-acp.
It defines application settings with validation, environment variable reading, and cross-platform executable discovery using shutil.which with zero hardcoded system paths.

## 2. Public Interface

### 2.1 Functions
- resolve_binary(binary_name: str, override_path: Optional[str] = None) -> Optional[Path]
  - Resolves executable binary cross-platform.
  - If override_path is provided and non-empty:
    - Resolves and verifies that the path exists and is a file. If not found, raises FileNotFoundError.
  - If override_path is None or empty:
    - Probes shutil.which(binary_name).
    - On Windows (sys.platform == 'win32'), checks extensions: .cmd, .exe, .bat if not already present.
    - On Linux/Unix, probes both raw name (e.g. gy_acp_server) and .par extension (e.g. gy_acp_server.par).
    - Returns Path(found_path).resolve() if found, else None.

### 2.2 Classes
- Settings(pydantic.BaseModel):
  - gy_cli_path: Optional[str] = None (Env: AGY_CLI_PATH)
  - gy_acp_server_path: Optional[str] = None (Env: AGY_ACP_SERVER_PATH)
  - gy_acp_args: List[str] = Field(default_factory=...) (Env: AGY_ACP_ARGS, parsed as whitespace/comma separated or list; default ['--uid='] on non-Windows if empty, [] on Windows)
  - gy_default_model: str = 'gemini-3.8-flash-high' (Env: AGY_DEFAULT_MODEL)
  - gy_thinking_level: str = 'high' (Env: AGY_THINKING_LEVEL, validated: 'low', 'medium', 'high', 'off')
  - gy_log_level: str = 'warn' (Env: AGY_LOG_LEVEL, validated: 'error', 'warn', 'info', 'debug')
  - hermes_home: Path = Field(default_factory=...) (Env: HERMES_HOME, default ~/.hermes, expanded)
  - state_db_path: Path = Field(default_factory=...) (Env: AGY_STATE_DB_PATH, default /state_agy_acp.db)
  - mode: str = 'agentic_proxy' (Env: AGY_MODE, validated: 'agentic_proxy', 'direct')

  Methods:
  - get_resolved_cli_path() -> Optional[Path]
  - get_resolved_server_path() -> Optional[Path]
  - get_server_command() -> List[str]
    - Returns command arguments list for executing ACP server: [str(server_path), *agy_acp_args].
    - If server binary cannot be resolved, raises RuntimeError.

- get_settings() -> Settings:
  - Returns cached or singleton Settings loaded from environment.

## 3. Invariants & Edge Cases
- Zero hardcoded machine paths (e.g. no /home/username/... or C:\Users\...).
- Respects active environment variables over defaults.
- Handles empty string env vars (e.g. AGY_CLI_PATH=") gracefully as None.
- On non-Windows, if AGY_ACP_ARGS is not explicitly set, defaults to ['--uid=']. On Windows, defaults to [].
- state_db_path defaults to /state_agy_acp.db.
