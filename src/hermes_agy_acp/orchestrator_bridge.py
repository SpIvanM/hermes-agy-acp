from __future__ import annotations

import re
from typing import Any

MEDIA_REGEX = re.compile(r"\[?MEDIA:\s*([^\]\s]+)\]?", re.IGNORECASE)


def strip_hermes_tools(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip tools and tool_choice from request payload to prevent Split-Brain."""
    clean = payload.copy()
    clean.pop("tools", None)
    clean.pop("tool_choice", None)
    return clean


def inject_bootstrap_prompt(
    messages: list[dict[str, Any]],
    bootstrap_text: str = "/s /user-communication",
) -> list[dict[str, Any]]:
    """Inject bootstrap directives into the system prompt without duplicates."""
    result: list[dict[str, Any]] = [m.copy() for m in messages]

    for m in result:
        if m.get("role") == "system":
            content = m.get("content", "")
            if bootstrap_text not in content:
                m["content"] = f"{content}\n\n{bootstrap_text}".strip()
            return result

    # No existing system message; prepend one
    result.insert(0, {"role": "system", "content": bootstrap_text})
    return result


def extract_media_tags(text: str) -> tuple[str, list[str]]:
    """Extract media attachment paths and remove tags from text."""
    paths: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        paths.append(match.group(1).strip())
        return ""

    cleaned = MEDIA_REGEX.sub(_replace, text)
    # Clean up double whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, paths


def format_media_tag(path: str) -> str:
    """Format path into a MEDIA tag."""
    return f"MEDIA:{path}"


def prepare_proxy_request(
    payload: dict[str, Any],
    mode: str = "agentic_proxy",
    bootstrap_text: str = "/s /user-communication",
) -> dict[str, Any]:
    """Prepare request for ACP proxy execution according to operating mode."""
    prepared = payload.copy()

    if mode == "agentic_proxy":
        prepared = strip_hermes_tools(prepared)
        if "messages" in prepared and isinstance(prepared["messages"], list):
            prepared["messages"] = inject_bootstrap_prompt(
                prepared["messages"],
                bootstrap_text=bootstrap_text,
            )

    return prepared


def process_proxy_response(response: dict[str, Any]) -> dict[str, Any]:
    """Process model response, extracting media attachments for chat bridges."""
    processed = response.copy()
    all_media: list[str] = []

    choices = processed.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            msg = choice.get("message")
            if isinstance(msg, dict) and "content" in msg and isinstance(msg["content"], str):
                cleaned, paths = extract_media_tags(msg["content"])
                msg["content"] = cleaned
                all_media.extend(paths)

    if all_media:
        processed["media_files"] = all_media

    return processed
