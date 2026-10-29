# Specification T12: Antigravity ACP Client (client.py)

## 1. Overview
The `client.py` module defines `AntigravityACPClient` which connects Hermes Agent to Google Antigravity via the Agent Client Protocol (ACP).
It declares the Hermes ACP provider contract flags (`HERMES_SKIP_TRANSPORT_WRAP` and `HERMES_SKIP_ASYNC_WRAP`), translates chat completion requests into ACP session protocol messages, streams delta tokens, and transparently resolves auto-approval permissions.

## 2. Public Interface

### 2.1 Classes

#### `BaseACPClient`:
- Base class (or mockable protocol compatible with Hermes `agent.acp_client.ACPClient`).

#### `AntigravityACPClient(BaseACPClient)`:
- Class attributes:
  - `HERMES_SKIP_TRANSPORT_WRAP: bool = True`
  - `HERMES_SKIP_ASYNC_WRAP: bool = True`
- Constructor:
  - `__init__(supervisor: ACPProcessSupervisor | None = None, registry: SessionRegistry | None = None, settings: Settings | None = None)`
  - Initializes supervisor, session registry, and configuration.
- Methods:
  - `initialize() -> dict[str, Any]`:
    - Sends `initialize` request to ACP server. Returns server initialize result.
  - `ensure_session(hermes_session_id: str, cwd: str | None = None, model: str | None = None) -> str`:
    - Checks `registry.get_acp_session_id(hermes_session_id)`.
    - If found, returns existing `acp_session_id`.
    - If not found, sends `session/new` to ACP server, retrieves `sessionId`, saves mapping to registry, and returns `acp_session_id`.
  - `create_chat_completion(messages: list[dict[str, Any]], model: str | None = None, stream: bool = False, session_id: str | None = None, cwd: str | None = None) -> dict[str, Any] | Iterator[dict[str, Any]]`:
    - Translates messages to prompt text.
    - Ensures session exists via `ensure_session(session_id or "default")`.
    - Sends `session/prompt` to ACP server.
    - Handles incoming lines:
      - If `session/request_permission` received, sends auto-approval response (`decision: allow`).
      - Yields or accumulates text updates/chunks.
    - If `stream=True`: yields delta chunk dictionaries in OpenAI format:
      `{"choices": [{"delta": {"content": chunk_text}, "index": 0, "finish_reason": finish_reason}]}`
    - If `stream=False`: returns dictionary in OpenAI format:
      `{"choices": [{"message": {"role": "assistant", "content": full_text}, "index": 0, "finish_reason": "stop"}]}`
  - `reset(hermes_session_id: str) -> bool`:
    - Resets session in registry.
  - `close() -> None`:
    - Terminates supervisor and closes registry.

## 3. Invariants & Error Handling
- Transparent auto-approval: non-blocking response to any permission request.
- Detects auth errors via `is_auth_error()` and raises descriptive `PermissionError` or `RuntimeError`.
- Clean stream termination with `finish_reason: "stop"`.
