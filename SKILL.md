---
name: hermes-agy-acp
description: "Adapter and bridge connecting NousResearch Hermes Agent with Google Antigravity ACP"
---

# Hermes Antigravity ACP Integration Skill

## Overview

Use this skill when interacting with the `hermes-agy-acp` adapter, configuring Antigravity as a Hermes model provider, managing ACP server lifecycles, or debugging session rehydration and skill learning pipelines.

## Triggers & Intent
- Trigger when the user mentions `hermes-agy-acp`, `antigravity-acp`, or configuring ACP for Hermes Agent.
- Trigger when troubleshooting Split-Brain tool calling between Hermes and Antigravity.
- Trigger when checking status of the ACP process pool or resetting session mappings.

## Operational Commands

### 1. Diagnostics & Health Check
Run the status diagnostic routine to verify that both `agy` and `agy_acp_server` binaries are resolved and that the SQLite registry is functional:
```python
from hermes_agy_acp import diagnostics_status
status = diagnostics_status()
print(status)
```

### 2. Session Reset (/reset)
To unlink a stale or hung ACP session and initiate a fresh context:
```python
from hermes_agy_acp import diagnostics_reset
result = diagnostics_reset("<hermes_session_id>")
print(result)
```

### 3. Skill Synthesis Verification
To synthesize and deploy a skill learned from a session:
```python
from hermes_agy_acp import AgySkillLearner
learner = AgySkillLearner()
learner.background_review_hook(session_id="session-123", transcript="...")
```

## Safety Invariants
- Never commit `.env` or credentials to git.
- Keep database file permissions strictly set to POSIX `0600`.
- Preserve tool stripping in `agentic_proxy` mode to prevent agent hallucination.
