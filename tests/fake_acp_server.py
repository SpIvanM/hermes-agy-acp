from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _log(log_path: str | None, entry: dict) -> None:
    if not log_path:
        return
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def main() -> int:
    scenario = os.getenv("FAKE_ACP_SCENARIO", "basic")
    log_path = os.getenv("FAKE_ACP_LOG", "")
    session_id = os.getenv("FAKE_ACP_SESSION_ID", f"acp-sess-{os.getpid()}")
    next_srv_id = 9000

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        msg = json.loads(line)
        _log(log_path, msg)
        method = msg.get("method")
        req_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "initialize":
            _emit(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": 1,
                        "agentInfo": {"name": "fake-acp", "version": "1.0.0"},
                        "agentCapabilities": {"loadSession": True},
                    },
                }
            )
        elif method == "session/new":
            result: dict = {"sessionId": session_id}
            if scenario != "legacy_models_only":
                result["configOptions"] = [
                    {
                        "id": "model",
                        "category": "model",
                        "type": "select",
                        "currentValue": "gemini-3.8-flash",
                        "options": [
                            {"value": "gemini-4-argon"},
                            {"value": "gemini-4-hydrogen"},
                            {"value": "gemini-3.8-flash"},
                        ],
                    },
                    {
                        "id": "thinking_level",
                        "category": "thought_level",
                        "type": "select",
                        "currentValue": "high",
                        "options": [
                            {"value": "low"},
                            {"value": "medium"},
                            {"value": "high"},
                        ],
                    },
                ]
            else:
                result["models"] = {
                    "currentModelId": "gemini-4-argon",
                    "availableModels": [
                        {"modelId": "gemini-4-argon"},
                        {"modelId": "gemini-4-hydrogen"},
                    ],
                }
            _emit({"jsonrpc": "2.0", "id": req_id, "result": result})
            _emit(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "available_commands_update",
                            "availableCommands": [],
                        },
                    },
                }
            )
        elif method == "session/load":
            if scenario == "load_fails":
                _emit({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32002, "message": "not found"}})
            else:
                _emit({"jsonrpc": "2.0", "id": req_id, "result": {"sessionId": params.get("sessionId", session_id)}})
        elif method == "session/set_mode":
            if scenario == "early_eof_on_set_mode":
                return 0
            _emit({"jsonrpc": "2.0", "id": req_id, "result": {}})
        elif method in ("session/set_config_option", "session/set_model"):
            _emit({"jsonrpc": "2.0", "id": req_id, "result": {}})
        elif method == "session/cancel":
            _log(log_path, {"event": "cancelled", "sessionId": params.get("sessionId")})
            return 0
        elif method == "session/prompt":
            prompt_blocks = params.get("prompt") or []
            prompt_text = " ".join(b.get("text", "") for b in prompt_blocks if isinstance(b, dict))

            if scenario == "permission_and_fs":
                next_srv_id += 1
                _emit(
                    {
                        "jsonrpc": "2.0",
                        "id": next_srv_id,
                        "method": "session/request_permission",
                        "params": {
                            "sessionId": session_id,
                            "options": [
                                {"optionId": "reject_once", "kind": "reject_once", "name": "Reject"},
                                {"optionId": "allow_always", "kind": "allow_always", "name": "Allow Always"},
                                {"optionId": "allow_once", "kind": "allow_once", "name": "Allow Once"},
                            ],
                        },
                    }
                )
                perm_reply = json.loads(sys.stdin.readline())
                _log(log_path, {"perm_reply": perm_reply})

                cwd = os.getenv("FAKE_ACP_CWD", os.getcwd())
                target_file = str(Path(cwd) / "created_by_acp.txt")
                next_srv_id += 1
                _emit(
                    {
                        "jsonrpc": "2.0",
                        "id": next_srv_id,
                        "method": "fs/write_text_file",
                        "params": {"sessionId": session_id, "path": target_file, "content": "hello-fs"},
                    }
                )
                w_reply = json.loads(sys.stdin.readline())
                _log(log_path, {"write_reply": w_reply})

                next_srv_id += 1
                _emit(
                    {
                        "jsonrpc": "2.0",
                        "id": next_srv_id,
                        "method": "fs/read_text_file",
                        "params": {"sessionId": session_id, "path": target_file},
                    }
                )
                r_reply = json.loads(sys.stdin.readline())
                _log(log_path, {"read_reply": r_reply})

                next_srv_id += 1
                _emit(
                    {
                        "jsonrpc": "2.0",
                        "id": next_srv_id,
                        "method": "fs/read_text_file",
                        "params": {"sessionId": session_id, "path": "/etc/passwd"},
                    }
                )
                bad_reply = json.loads(sys.stdin.readline())
                _log(log_path, {"bad_read_reply": bad_reply})

            if scenario == "noise_and_thought":
                _emit(
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": "other-session-id",
                            "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "WRONG_SESSION"}},
                        },
                    }
                )
                _emit(
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": session_id,
                            "update": {"sessionUpdate": "tool_call", "title": "SHOULD_NOT_APPEAR"},
                        },
                    }
                )
                _emit(
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": session_id,
                            "update": {"sessionUpdate": "tool_call_update", "status": "completed"},
                        },
                    }
                )
                _emit(
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": session_id,
                            "update": {"sessionUpdate": "usage_update", "used": 10},
                        },
                    }
                )
                _emit(
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": session_id,
                            "update": {"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "thinking-step"}},
                        },
                    }
                )
                _emit(
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": session_id,
                            "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "clean-answer"}},
                        },
                    }
                )
            elif scenario == "tool_call":
                _emit(
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": session_id,
                            "update": {
                                "sessionUpdate": "agent_message_chunk",
                                "content": {
                                    "type": "text",
                                    "text": (
                                        "Checking memory now.\n"
                                        "<tool_call>{\"id\": \"call_mem_1\", \"type\": \"function\", "
                                        "\"function\": {\"name\": \"memory\", \"arguments\": \"{\\\"action\\\": \\\"list\\\"}\"}}</tool_call>"
                                    ),
                                },
                            },
                        },
                    }
                )
            elif scenario == "slow_hang":
                _emit(
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": session_id,
                            "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "started..."}},
                        },
                    }
                )
                while True:
                    cancel_raw = sys.stdin.readline()
                    if not cancel_raw:
                        break
                    if not cancel_raw.strip():
                        continue
                    cancel_msg = json.loads(cancel_raw.strip())
                    _log(log_path, cancel_msg)
                    if cancel_msg.get("method") == "session/cancel":
                        return 0
                time.sleep(5)
            elif scenario == "echo_prompt_marker":
                marker = prompt_text.splitlines()[-1] if prompt_text else ""
                time.sleep(0.05)
                _emit(
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": session_id,
                            "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": f"ECHO:{marker}"}},
                        },
                    }
                )
            elif scenario == "env_dump":
                proxy_info = {
                    k: os.getenv(k)
                    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy")
                }
                _emit(
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": session_id,
                            "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": json.dumps(proxy_info)}},
                        },
                    }
                )
            else:
                _emit(
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": session_id,
                            "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "ok"}},
                        },
                    }
                )

            _emit({"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "end_turn"}})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
