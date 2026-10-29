# Specification T16: Documentation, Skill Manifest & CI Workflow (T16)

## 1. Overview
The final deliverable sets provide bilingual developer documentation (`README.md`, `README.ru.md`), an autonomous agent skill manifest (`SKILL.md`), a 12-factor environment template (`.env.example`), and continuous integration automation (`.github/workflows/ci.yml`).

## 2. Public Interface & Artifacts

### 2.1 Bilingual Documentation
- `README.md` (English):
  - Project charter and architecture overview.
  - Integration modes: `agentic_proxy` (default) vs `direct`.
  - Installation and plugin discovery paths (`$HERMES_HOME/plugins/hermes-agy-acp`).
  - Configuration reference table for environment variables.
  - Diagnostic CLI commands and troubleshooting guide.
- `README.ru.md` (Russian):
  - Русскоязычная версия: цели проекта, архитектура, режимы работы, установка, конфигурация, диагностика.

### 2.2 Agent Skill Manifest
- `SKILL.md`:
  - agentskills.io format with YAML frontmatter:
    ```yaml
    ---
    name: hermes-agy-acp
    description: "Adapter and bridge connecting NousResearch Hermes Agent with Google Antigravity ACP"
    ---
    ```
  - Operational runbook, slash commands (`/agy status`, `/agy reset`), and safety invariants.

### 2.3 Environment Template
- `.env.example`:
  - Covers `AGY_CLI_PATH`, `AGY_ACP_SERVER_PATH`, `AGY_ACP_ARGS`, `AGY_DEFAULT_MODEL`, `AGY_THINKING_LEVEL`, `AGY_LOG_LEVEL`, `HERMES_HOME`, `AGY_STATE_DB_PATH`, `AGY_MODE`.

### 2.4 CI Workflow
- `.github/workflows/ci.yml`:
  - Triggered on push and pull_request to `main`.
  - Matrix across `ubuntu-latest` and `windows-latest`.
  - Python versions: `3.11`, `3.12`.
  - Steps: checkout, setup-python, install dependencies (`.[dev]`), ruff check, mypy --strict, pytest.
