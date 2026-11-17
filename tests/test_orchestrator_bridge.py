from __future__ import annotations

from hermes_agy_acp.orchestrator_bridge import (
    ACP_OVERLAPPING_TOOL_NAMES,
    filter_hermes_tools,
    inject_bootstrap_prompt,
    prepare_proxy_request,
)


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Tool {name}",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_acp_overlapping_tool_names_complete():
    expected = {
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
    }
    assert expected.issubset(ACP_OVERLAPPING_TOOL_NAMES)


def test_filter_hermes_tools_keeps_only_non_overlapping():
    tools = [
        _tool("terminal"),
        _tool("process_manage"),
        _tool("read_file"),
        _tool("write_file"),
        _tool("patch"),
        _tool("search_files"),
        _tool("web_search"),
        _tool("web_extract"),
        _tool("browser_exec"),
        _tool("browser_cdp"),
        _tool("browser_dialog"),
        _tool("browser_vault"),
        _tool("execute_code"),
        _tool("delegate_task"),
        _tool("memory"),
        _tool("manage_connections"),
        _tool("cronjob_manage"),
        _tool("clarify"),
        _tool("session_search"),
        _tool("skills_list"),
    ]
    filtered = filter_hermes_tools(tools)
    names = [t["function"]["name"] for t in filtered]
    assert names == [
        "memory",
        "manage_connections",
        "cronjob_manage",
        "clarify",
        "session_search",
        "skills_list",
    ]


def test_no_unconditional_bootstrap_injection(monkeypatch):
    monkeypatch.delenv("AGY_BOOTSTRAP_PROMPT", raising=False)
    msgs = [{"role": "user", "content": "hello"}]
    out_msgs = inject_bootstrap_prompt(msgs)
    assert out_msgs == msgs

    payload = {
        "messages": msgs,
        "tools": [_tool("read_file"), _tool("memory")],
    }
    prepared = prepare_proxy_request(payload)
    assert prepared["messages"] == msgs
    assert [t["function"]["name"] for t in prepared["tools"]] == ["memory"]
