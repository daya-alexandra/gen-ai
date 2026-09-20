"""Подготовить локальный .env и при возможности перенести доступ из соседней работы."""
from pathlib import Path

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parent
EMPTY = {"", "PASTE_YOUR_KEY_HERE", "your_token_here"}


def prepare(root=ROOT):
    root = Path(root)
    env = root / ".env"
    if env.exists():
        return (dotenv_values(env).get("LLM_AUTH_TOKEN") or "").strip() not in EMPTY

    template = (root / ".env.example").read_text(encoding="utf-8")
    candidates = [
        root.parent / name / ".env"
        for name in ("lab5_pwc", "lab4_agent", "lab3_rag", "lab2_text", "lab1_verified")
    ]
    candidates += [root.parent / ".env", root.parent.parent / "genai_labs" / ".env"]

    for source in candidates:
        if source == env or not source.exists():
            continue
        values = dotenv_values(source)
        key = (values.get("LLM_AUTH_TOKEN") or "").strip()
        if key in EMPTY or any(char in key for char in "\r\n\x00"):
            continue
        replacements = {"LLM_AUTH_TOKEN": key}
        for name in ("LLM_MODEL", "LLM_BASE_URL"):
            if values.get(name):
                replacements[name] = values[name]
        lines = []
        for line in template.splitlines():
            name = line.split("=", 1)[0]
            if name in replacements:
                value = replacements[name].replace("\\", "\\\\").replace("'", "\\'")
                line = f"{name}='{value}'"
            lines.append(line)
        env.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True

    env.write_text(template, encoding="utf-8")
    return False


if __name__ == "__main__":
    import sys

    ready = prepare()
    print("API key is configured locally." if ready else "Fill LLM_AUTH_TOKEN in .env; keep this file private.")
    sys.exit(0 if ready else 2)
