# Specification T15: Plugin Manifest & Registration (plugin.yaml & __init__.py)

## 1. Overview
The `__init__.py` and `plugin.yaml` files configure `hermes-agy-acp` as a NousResearch Hermes model-provider plugin.
They declare the provider profile (`antigravity-acp`), register the client factory, expose package exports, and provide CLI/chat diagnostic commands.

## 2. Public Interface

### 2.1 Manifest: `plugin.yaml`
- `kind: model-provider`
- `name: hermes-agy-acp`
- `version: 1.0.0`
- `description: ...`
- `entrypoint: hermes_agy_acp`
- `provider: antigravity-acp`

### 2.2 Provider Profile & Registration in `__init__.py`
- `ProviderProfile(BaseModel)`:
  - `name: str = "antigravity-acp"`
  - `auth_type: str = "external_process"`
  - `marker_base_url: str = "acp://antigravity"`
  - `supports_streaming: bool = True`
- `get_provider_profile() -> ProviderProfile`
- `create_client() -> AntigravityACPClient`
- `register(registry: Any = None) -> ProviderProfile`:
  - Registers the provider profile with the given registry or global table.

### 2.3 Diagnostic Commands
- `diagnostics_status() -> dict[str, Any]`:
  - Returns health dictionary containing:
    - `"status"`: `"healthy"` | `"degraded"`
    - `"cli_path"`: resolved CLI path string or `None`
    - `"server_path"`: resolved ACP server path string or `None`
    - `"default_model"`: configured default model
    - `"active_sessions"`: count of active sessions from `SessionRegistry`
- `diagnostics_reset(session_id: str) -> str`:
  - Resets session in registry. Returns user-facing status message.

### 2.4 Package Exports
- `__all__`:
  - `AntigravityACPClient`
  - `ACPProcessSupervisor`
  - `SessionRegistry`
  - `Settings`
  - `get_settings`
  - `AgySkillLearner`
  - `ProviderProfile`
  - `get_provider_profile`
  - `create_client`
  - `diagnostics_status`
  - `diagnostics_reset`
