# Specification T14: Autonomous Skill Learning Pipeline (skill_learning_agy.py)

## 1. Overview
The `skill_learning_agy.py` module integrates with Hermes Agent's background review and curator hooks.
It utilizes the Google Antigravity CLI (`agy --dangerously-skip-permissions -p ...`) to synthesize new procedural skills from conversation transcripts, verifies them against YAML syntax and AST security rules, and synchronizes valid skills bidirectionally to `~/.hermes/skills/` and `~/.gemini/config/skills/`.

## 2. Public Interface

### 2.1 Functions / Validators
- `validate_skill_frontmatter(content: str) -> tuple[bool, str | None]`:
  - Validates that content begins with YAML frontmatter `---`, contains required keys `name` and `description`, and closes with `---`.
  - Returns `(True, None)` or `(False, "Error message")`.
- `validate_skill_safety(content: str) -> tuple[bool, str | None]`:
  - Scans content for dangerous shell/system patterns (e.g. `rm -rf /`, `:(){ :|:& };:`, raw credential exfiltration).
  - Returns `(True, None)` or `(False, "Safety violation detected: ...")`.

### 2.2 Classes
- `AgySkillLearner`:
  - `__init__(settings: Settings | None = None, gemini_skills_dir: Path | str | None = None, hermes_skills_dir: Path | str | None = None)`:
    - Default `gemini_skills_dir`: `~/.gemini/config/skills` (expanded).
    - Default `hermes_skills_dir`: `~/.hermes/skills` (expanded).
  - Methods:
    - `build_synthesis_prompt(topic: str, context: str) -> str`:
      - Builds structured prompt requesting agentskills.io format with YAML frontmatter.
    - `validate_skill(content: str) -> tuple[bool, str | None]`:
      - Runs both frontmatter and safety validations.
    - `install_skill(skill_name: str, content: str) -> dict[str, Path]`:
      - Validates skill content. Raises `ValueError` if invalid.
      - Writes `SKILL.md` to `hermes_skills_dir / skill_name / "SKILL.md"`.
      - Syncs `SKILL.md` to `gemini_skills_dir / skill_name / "SKILL.md"`.
      - Returns `{"hermes_path": path, "gemini_path": path}`.
    - `execute_synthesis_cli(prompt: str, model: str | None = None) -> str`:
      - Runs `agy --dangerously-skip-permissions --print -p "<prompt>"` using resolved CLI path.
      - Returns stdout from agy command.
    - `background_review_hook(session_id: str, transcript: str) -> dict[str, Any] | None`:
      - Background worker hook inspecting transcript for reproducible procedures.
      - Synthesizes and installs skill if candidate is found.
