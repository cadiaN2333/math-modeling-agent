from pathlib import Path


def test_env_template_is_safe_and_root_env_is_ignored() -> None:
    project_root = Path(__file__).resolve().parents[1]
    ignore_rules = (project_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    template_lines = (project_root / ".env.example").read_text(encoding="utf-8").splitlines()

    assert "/.env" in ignore_rules
    key_line = next(line for line in template_lines if line.startswith("DEEPSEEK_API_KEY="))
    assert key_line == "DEEPSEEK_API_KEY="
    assert "DEEPSEEK_MODEL=deepseek-flash" in template_lines
    assert "DEEPSEEK_BASE_URL=https://api.deepseek.com" in template_lines
