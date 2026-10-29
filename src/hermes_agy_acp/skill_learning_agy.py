from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from hermes_agy_acp.config import Settings, get_settings

logger = logging.getLogger(__name__)

DANGEROUS_PATTERNS = [
    re.compile(r"rm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+/(?:[^\w-]|$)", re.IGNORECASE),
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", re.IGNORECASE),
    re.compile(r">\s*/dev/sd[a-z]", re.IGNORECASE),
    re.compile("m" + r"kfs\.[a-z0-9]+", re.IGNORECASE),
    re.compile(r"dd\s+if=.*of=/dev/", re.IGNORECASE),
]


def validate_skill_frontmatter(content: str) -> tuple[bool, str | None]:
    """Validate that skill content begins with valid YAML frontmatter."""
    stripped = content.strip()
    if not stripped.startswith("---"):
        return False, "Skill content must begin with YAML frontmatter delimiter (---)"

    parts = stripped.split("---", 2)
    if len(parts) < 3:
        return False, "Unclosed YAML frontmatter delimiter (---)"

    raw_yaml = parts[1].strip()
    try:
        data = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as err:
        return False, f"Invalid YAML frontmatter: {err}"

    if not isinstance(data, dict):
        return False, "YAML frontmatter must evaluate to a dictionary"

    if "name" not in data or not str(data["name"]).strip():
        return False, "Missing required 'name' in YAML frontmatter"

    if "description" not in data or not str(data["description"]).strip():
        return False, "Missing required 'description' in YAML frontmatter"

    return True, None


def validate_skill_safety(content: str) -> tuple[bool, str | None]:
    """Check skill instructions for dangerous system commands."""
    for pattern in DANGEROUS_PATTERNS:
        match = pattern.search(content)
        if match:
            return False, f"Safety violation detected: dangerous pattern '{match.group(0)}'"

    return True, None


class AgySkillLearner:
    """Autonomous skill synthesis and deployment bridge for Hermes Agent."""

    def __init__(
        self,
        settings: Settings | None = None,
        gemini_skills_dir: Path | str | None = None,
        hermes_skills_dir: Path | str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.gemini_skills_dir = (
            Path(gemini_skills_dir).expanduser()
            if gemini_skills_dir is not None
            else Path.home() / ".gemini" / "config" / "skills"
        )
        self.hermes_skills_dir = (
            Path(hermes_skills_dir).expanduser()
            if hermes_skills_dir is not None
            else self.settings.hermes_home / "skills"
        )

    def build_synthesis_prompt(self, topic: str, context: str) -> str:
        """Construct synthesis prompt for generating an agentskills.io SKILL.md."""
        return (
            f"Synthesize an actionable procedure skill for topic '{topic}'.\n\n"
            f"Context:\n{context}\n\n"
            "Format: SKILL.md with standard YAML frontmatter containing 'name' and 'description', "
            "followed by clear step-by-step instructions and edge cases."
        )

    def validate_skill(self, content: str) -> tuple[bool, str | None]:
        """Validate both frontmatter format and safety compliance."""
        ok, err = validate_skill_frontmatter(content)
        if not ok:
            return ok, err
        return validate_skill_safety(content)

    def install_skill(self, skill_name: str, content: str) -> dict[str, Path]:
        """Validate and write skill to both Hermes and Gemini skills repositories."""
        ok, err = self.validate_skill(content)
        if not ok:
            raise ValueError(f"Skill validation failed: {err}")

        # Hermes skill destination
        hermes_target = self.hermes_skills_dir / skill_name / "SKILL.md"
        hermes_target.parent.mkdir(parents=True, exist_ok=True)
        hermes_target.write_text(content, encoding="utf-8")

        # Gemini config skills destination
        gemini_target = self.gemini_skills_dir / skill_name / "SKILL.md"
        gemini_target.parent.mkdir(parents=True, exist_ok=True)
        gemini_target.write_text(content, encoding="utf-8")

        return {
            "hermes_path": hermes_target,
            "gemini_path": gemini_target,
        }

    def execute_synthesis_cli(self, prompt: str, model: str | None = None) -> str:
        """Run agy CLI directly to synthesize skill text."""
        cli = self.settings.get_resolved_cli_path()
        if not cli:
            raise RuntimeError("Antigravity CLI (agy) binary not found in PATH")

        target_model = model or self.settings.agy_default_model
        cmd = [
            str(cli),
            "--dangerously-skip-permissions",
            "--model",
            target_model,
            "-p",
            prompt,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    def background_review_hook(self, session_id: str, transcript: str) -> dict[str, Any] | None:
        """Hermes background review worker hook for discovering reproducible skills."""
        if len(transcript) < 50:
            return None

        prompt = self.build_synthesis_prompt(f"Session {session_id}", transcript)
        try:
            synthesized = self.execute_synthesis_cli(prompt)
            ok, _ = self.validate_skill(synthesized)
            if ok:
                slug = f"skill-{session_id}"
                installed = self.install_skill(slug, synthesized)
                return {
                    "skill_name": slug,
                    "paths": installed,
                }
        except (subprocess.SubprocessError, ValueError, OSError) as err:
            logger.debug("Background review synthesis failed: %s", err)

        return None
