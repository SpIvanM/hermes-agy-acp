from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace
from typing import Any, Iterable

MEDIA_REGEX = re.compile(r"\[?MEDIA:\s*([^\]\s]+)\]?", re.IGNORECASE)

ACP_OVERLAPPING_TOOL_NAMES: frozenset[str] = frozenset({
    "terminal",
    "process_manage",
    "read_file",
    "write_file",
    "patch",
    "search_files",
    "web_search",
    "web_extract",
    "browser_exec",
    "browser_cdp",
    "browser_dialog",
    "browser_vault",
    "execute_code",
    "delegate_task",
})

try:
    from agent.acp_openai_bridge import (  # type: ignore[import-not-found]
        StreamChunks,
        completion_to_stream_chunks,
        extract_tool_calls_from_text,
        render_tool_bridge_sections,
    )
except ImportError:
    _TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    _TOOL_CALL_JSON_RE = re.compile(
        r"\{\s*\"id\"\s*:\s*\"[^\"]+\"\s*,\s*\"type\"\s*:\s*\"function\"\s*,\s*\"function\"\s*:\s*\{.*?\}\s*\}",
        re.DOTALL,
    )
    _TOOL_CALL_CONTRACT = (
        "Available tools (OpenAI function schema). "
        "When using a tool, emit ONLY <tool_call>{...}</tool_call> with one JSON object "
        "containing id/type/function{name,arguments}. arguments must be a JSON string."
    )

    class StreamChunks(list):  # type: ignore[no-redef]
        """Chunk list compatible with agent.acp_openai_bridge.StreamChunks."""

    def completion_to_stream_chunks(completion: SimpleNamespace) -> StreamChunks:  # type: ignore[misc]
        choice = completion.choices[0]
        message = choice.message
        tool_call_deltas = None
        if message.tool_calls:
            tool_call_deltas = [
                SimpleNamespace(
                    index=index,
                    id=getattr(tc, "id", None),
                    type=getattr(tc, "type", "function"),
                    function=SimpleNamespace(
                        name=getattr(tc.function, "name", None),
                        arguments=getattr(tc.function, "arguments", None),
                    ),
                )
                for index, tc in enumerate(message.tool_calls)
            ]
        delta = SimpleNamespace(
            role="assistant",
            content=message.content or None,
            tool_calls=tool_call_deltas,
            reasoning_content=getattr(message, "reasoning_content", None),
            reasoning=getattr(message, "reasoning", None),
        )
        data_chunk = SimpleNamespace(
            choices=[SimpleNamespace(index=0, delta=delta, finish_reason=choice.finish_reason)],
            model=completion.model,
            usage=None,
        )
        usage_chunk = SimpleNamespace(choices=[], model=completion.model, usage=completion.usage)
        chunks = StreamChunks([data_chunk, usage_chunk])
        for key, value in vars(completion).items():
            if key not in ("choices", "usage", "model"):
                setattr(chunks, key, value)
        return chunks

    def _extract_named_fn(container: Any) -> tuple[dict[str, Any], str] | None:
        fn = container.get("function") if isinstance(container, dict) else None
        name = fn.get("name") if isinstance(fn, dict) else None
        return (fn, name.strip()) if isinstance(name, str) and name.strip() else None

    def render_tool_bridge_sections(  # type: ignore[misc]
        tools: list[dict[str, Any]] | None,
        tool_choice: Any = None,
        *,
        allowlist: Iterable[str] | None = None,
    ) -> list[str]:
        allowed = {str(n).strip() for n in allowlist} if allowlist is not None else None
        specs: list[dict[str, Any]] = []
        for t in tools or []:
            named = _extract_named_fn(t)
            if named is None or (allowed is not None and named[1] not in allowed):
                continue
            fn, name = named
            specs.append(
                {
                    "name": name,
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
        sections: list[str] = []
        if specs:
            sections.append(_TOOL_CALL_CONTRACT + "\n" + json.dumps(specs, ensure_ascii=False))
        if tool_choice is not None:
            sections.append(f"Tool choice hint: {json.dumps(tool_choice, ensure_ascii=False)}")
        return sections

    def extract_tool_calls_from_text(text: str) -> tuple[list[SimpleNamespace], str]:  # type: ignore[misc]
        if not isinstance(text, str) or not text.strip():
            return [], ""
        extracted: list[SimpleNamespace] = []
        consumed: list[tuple[int, int]] = []
        for pattern, group in ((_TOOL_CALL_BLOCK_RE, 1), (_TOOL_CALL_JSON_RE, 0)):
            for m in pattern.finditer(text):
                try:
                    obj = json.loads(m.group(group))
                except Exception:
                    consumed.append((m.start(), m.end()))
                    continue
                named = _extract_named_fn(obj)
                if named is not None:
                    fn, fn_name = named
                    fn_args = fn.get("arguments", "{}")
                    if not isinstance(fn_args, str):
                        fn_args = json.dumps(fn_args, ensure_ascii=False)
                    call_id = obj.get("id")
                    if not isinstance(call_id, str) or not call_id.strip():
                        call_id = f"acp_call_{len(extracted) + 1}"
                    extracted.append(
                        SimpleNamespace(
                            id=call_id,
                            call_id=call_id,
                            response_item_id=None,
                            type="function",
                            function=SimpleNamespace(name=fn_name, arguments=fn_args),
                        )
                    )
                consumed.append((m.start(), m.end()))
            if extracted:
                break
        if not consumed:
            return extracted, text.strip()
        consumed.sort()
        merged: list[tuple[int, int]] = []
        for start, end in consumed:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        parts: list[str] = []
        cursor = 0
        for start, end in merged:
            if cursor < start:
                parts.append(text[cursor:start])
            cursor = max(cursor, end)
        if cursor < len(text):
            parts.append(text[cursor:])
        cleaned = "\n".join(p.strip() for p in parts if p and p.strip()).strip()
        return extracted, cleaned


def _tool_name(tool: Any) -> str | None:
    if not isinstance(tool, dict):
        return None
    fn = tool.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str) and fn["name"].strip():
        return fn["name"].strip()
    name = tool.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def filter_hermes_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Keep only Hermes platform tools that do not overlap native Antigravity ACP tools."""
    if not tools:
        return []
    filtered: list[dict[str, Any]] = []
    for tool in tools:
        name = _tool_name(tool)
        if name is None or name in ACP_OVERLAPPING_TOOL_NAMES:
            continue
        filtered.append(tool)
    return filtered


def strip_hermes_tools(payload: dict[str, Any]) -> dict[str, Any]:
    """Legacy helper: strip all tools and tool_choice from request payload."""
    clean = payload.copy()
    clean.pop("tools", None)
    clean.pop("tool_choice", None)
    return clean


def inject_bootstrap_prompt(
    messages: list[dict[str, Any]],
    bootstrap_text: str | None = None,
) -> list[dict[str, Any]]:
    """Inject bootstrap directives only when explicitly configured via AGY_BOOTSTRAP_PROMPT."""
    effective = os.getenv("AGY_BOOTSTRAP_PROMPT", "") if bootstrap_text is None else bootstrap_text
    effective = effective.strip()
    result: list[dict[str, Any]] = [m.copy() for m in messages]
    if not effective:
        return result

    for m in result:
        if m.get("role") == "system":
            content = str(m.get("content", ""))
            if effective not in content:
                m["content"] = f"{content}\n\n{effective}".strip()
            return result

    result.insert(0, {"role": "system", "content": effective})
    return result


def extract_media_tags(text: str) -> tuple[str, list[str]]:
    """Extract media attachment paths and remove MEDIA tags while preserving text layout."""
    paths: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        paths.append(match.group(1).strip())
        return ""

    cleaned = MEDIA_REGEX.sub(_replace, text)
    if not paths:
        return text, []
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, paths


def format_media_tag(path: str) -> str:
    """Format path into a MEDIA tag."""
    return f"MEDIA:{path}"


def prepare_proxy_request(
    payload: dict[str, Any],
    mode: str = "agentic_proxy",
    bootstrap_text: str | None = None,
) -> dict[str, Any]:
    """Prepare request for ACP execution: filter overlapping tools and apply optional bootstrap."""
    prepared = payload.copy()

    if mode == "agentic_proxy":
        if "tools" in prepared:
            filtered = filter_hermes_tools(prepared.get("tools"))
            if filtered:
                prepared["tools"] = filtered
            else:
                prepared.pop("tools", None)
                prepared.pop("tool_choice", None)
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


__all__ = [
    "ACP_OVERLAPPING_TOOL_NAMES",
    "MEDIA_REGEX",
    "StreamChunks",
    "completion_to_stream_chunks",
    "extract_media_tags",
    "extract_tool_calls_from_text",
    "filter_hermes_tools",
    "format_media_tag",
    "inject_bootstrap_prompt",
    "prepare_proxy_request",
    "process_proxy_response",
    "render_tool_bridge_sections",
    "strip_hermes_tools",
]
