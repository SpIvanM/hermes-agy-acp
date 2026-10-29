# Specification T13: Orchestrator Middleware & Tool-Stripping Bridge (orchestrator_bridge.py)

## 1. Overview
The `orchestrator_bridge.py` module serves as the middleware bridge between Hermes Agent and Antigravity in `agentic_proxy` mode.
It strips redundant Hermes tool declarations to avoid agentic "Split-Brain", injects the `/s /user-communication` bootstrap prompt, and bridges Telegram media tags (`MEDIA:<path>`).

## 2. Public Interface

### 2.1 Functions
- `strip_hermes_tools(payload: dict[str, Any]) -> dict[str, Any]`:
  - Returns a copy of `payload` with `"tools"` and `"tool_choice"` removed if present.
- `inject_bootstrap_prompt(messages: list[dict[str, Any]], bootstrap_text: str = "/s /user-communication") -> list[dict[str, Any]]`:
  - Injects `bootstrap_text` into the prompt.
  - If a system message (`role == "system"`) exists, prepends/appends `bootstrap_text` if not already present.
  - If no system message exists, prepends a new system message `{"role": "system", "content": bootstrap_text}`.
  - Does not duplicate `bootstrap_text` if it already exists in the message content.
- `extract_media_tags(text: str) -> tuple[str, list[str]]`:
  - Parses `MEDIA:<path>` or `[MEDIA: <path>]` patterns from `text`.
  - Returns `(cleaned_text, list_of_media_paths)`.
- `format_media_tag(path: str) -> str`:
  - Formats a media path as `MEDIA:<path>`.
- `prepare_proxy_request(payload: dict[str, Any], mode: str = "agentic_proxy", bootstrap_text: str = "/s /user-communication") -> dict[str, Any]`:
  - If `mode == "agentic_proxy"`:
    - Strips tools via `strip_hermes_tools`.
    - Injects bootstrap prompt into `payload["messages"]` via `inject_bootstrap_prompt`.
  - Returns modified request payload.
- `process_proxy_response(response: dict[str, Any]) -> dict[str, Any]`:
  - Parses content in response choices for media tags and populates `response["media_files"]` if media tags found.
