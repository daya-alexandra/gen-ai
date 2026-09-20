"""Минимальный API-runtime: журнал, кэш, бюджет и безопасное возобновление."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARIFF = {
    "checked_on": "2026-09-19",
    "source": "https://api-docs.deepseek.com/quick_start/pricing/",
    "input_miss_per_million": 0.30,
    "input_hit_per_million": 0.006,
    "output_per_million": 1.20,
    "off_peak_multiplier": 0.5,
}

# Точная сигнатура первой опубликованной версии. Совместимость разрешена только
# для неё: входной CSV, модель, тариф и адрес API должны остаться прежними.
LEGACY_RELEASE_FILES = {
    "bootstrap.py": "6b9ce4d6d1d383e1fd237a6f83928b6b4ec6085b05aafdeac8fbce93d03329c8",
    "collect_results.py": "2f94be861724ee32d00b52091ad1aaefda4fe147e4a72978215f5fe4e1959bf3",
    "experiments.py": "a5fa19122651d97e890eee103839151cf6e995570b76ee3b1b35542dd143c5e5",
    "run_lab.py": "e19d1edf947db869116f5e7fcce2e81c48d46891a3eec95c68624cb3097cbbab",
    "runtime.py": "fae7d3625a16a651ffbfb57c7fc23e1ca328ff15d6949b97161fb990b586f8b8",
    "test_lab.py": "e7baa69fcd982715ce10677b33c103d445a3814acdcd849503c38daf30b9aad5",
    "verify_results.py": "f2d5d53785fda58c56e97e1c16e9298b5950dee4727829f37aa2d9af43e3d596",
    "input/headlines.csv": "dc774d1b34db12fde7a2bb3063aacacec705fca6d6ee7bc48248eec4a81658f2",
}

# Версия первого исправления: zero-shot/role уже увеличены до 800 токенов,
# но ответы персон ещё имели прежний предел 350 токенов.
TOKEN_LIMIT_FIX_FILES = {
    "bootstrap.py": "6b9ce4d6d1d383e1fd237a6f83928b6b4ec6085b05aafdeac8fbce93d03329c8",
    "collect_results.py": "2f94be861724ee32d00b52091ad1aaefda4fe147e4a72978215f5fe4e1959bf3",
    "experiments.py": "ff39b19b237dccc009a3ff9e229a97fc4ebebeda7ec4b1005fae957357bcbee7",
    "run_lab.py": "e19d1edf947db869116f5e7fcce2e81c48d46891a3eec95c68624cb3097cbbab",
    "runtime.py": "744f39404fe121cbdadcb847b8e38ac28099a97689bbf04f5d62c7f7fdd47511",
    "test_lab.py": "4d2ab4190b82efb6f6d8ce16aa38c6d5ac599ca7ff0da76b8f7d7ee1e56cdfba",
    "verify_results.py": "9828f764716c0bd0454125f11b3ebd1691d611647f37bc3acfffa5c0f68683cb",
    "input/headlines.csv": "dc774d1b34db12fde7a2bb3063aacacec705fca6d6ee7bc48248eec4a81658f2",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def dump(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for attempt in range(7):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(0.05 * (attempt + 1))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def events(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_hashes() -> dict[str, str]:
    paths = sorted(ROOT.glob("*.py")) + sorted((ROOT / "input").glob("*"))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def compatible_legacy_config(previous: dict, current: dict) -> bool:
    """Разрешить продолжение только с точно известной исходной версии."""
    known_files = (LEGACY_RELEASE_FILES, TOKEN_LIMIT_FIX_FILES)
    if not any(previous.get("files_sha256") == item for item in known_files):
        return False
    previous_without_files = {key: value for key, value in previous.items() if key != "files_sha256"}
    current_without_files = {key: value for key, value in current.items() if key != "files_sha256"}
    return previous_without_files == current_without_files


def usage_cost(usage: dict | None) -> float | None:
    if not usage or usage.get("prompt_tokens") is None or usage.get("completion_tokens") is None:
        return None
    prompt = usage["prompt_tokens"]
    cached = min(prompt, usage.get("prompt_cache_hit_tokens", 0) or 0)
    return ((prompt - cached) * 0.30 + cached * 0.006 + usage["completion_tokens"] * 1.20) / 1_000_000


class ApiRun:
    def __init__(self, output="output", transport=None):
        if transport is None:
            from dotenv import load_dotenv

            load_dotenv(ROOT.parent / ".env")
            load_dotenv(ROOT / ".env", override=True)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.transport = transport
        self.client = None
        self.model = os.getenv("LLM_MODEL", "deepseek-flash")
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
        if self.base_url not in {"https://api.deepseek.com", "https://api.deepseek.com/v1"}:
            raise ValueError("Для этого комплекта разрешён официальный адрес DeepSeek API")
        if self.model not in {"deepseek-flash", "deepseek-v4-flash"}:
            raise ValueError("Используй модель DeepSeek Flash, для которой рассчитан бюджет")
        self.config = {
            "suite": "lab1_api_basics",
            "model": self.model,
            "base_url": self.base_url,
            "run_kind": "offline_test" if transport else "llm_api",
            "files_sha256": file_hashes(),
            "tariff": TARIFF,
        }
        self.meta_path = self.output / "run.json"
        self.trace_path = self.output / "api_trace.jsonl"
        self.ledger_path = self.output / "budget.json"
        if self.meta_path.exists():
            self.meta = read(self.meta_path)
            if self.meta["config"] != self.config:
                previous = self.meta["config"]
                if not compatible_legacy_config(previous, self.config):
                    raise ValueError("Код или входные данные изменились. Выбери новую папку --output.")
                compatible = self.meta.setdefault("compatible_cache_configs", [])
                if previous not in compatible:
                    compatible.append(previous)
                self.meta.setdefault("resume_migrations", []).append(
                    {
                        "at": now(),
                        "reason": "increase_generation_token_limits",
                        "from_files_sha256": previous["files_sha256"],
                        "to_files_sha256": self.config["files_sha256"],
                    }
                )
                self.meta["config"] = self.config
                dump(self.meta_path, self.meta)
        else:
            self.meta = {
                "run_id": str(uuid.uuid4()),
                "created_at": now(),
                "status": "prepared",
                "active_seconds": 0.0,
                "config": self.config,
            }
            dump(self.meta_path, self.meta)
        self.compatible_cache_configs = self.meta.get("compatible_cache_configs", [])
        self.started = time.perf_counter()

    def _log(self, event: dict) -> None:
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"run_id": self.meta["run_id"], "ts": now(), **event}, ensure_ascii=False) + "\n")

    def _reserve(self, request: dict) -> str:
        ledger = read(self.ledger_path) if self.ledger_path.exists() else {}
        if len(ledger) >= int(os.getenv("LLM_MAX_CALLS", "280")):
            raise RuntimeError("Достигнут лимит API-вызовов; прогресс сохранён")
        rough_input = len(json.dumps(request, ensure_ascii=False).encode("utf-8")) / 3
        upper = (rough_input * 0.30 + request["max_tokens"] * 1.20) / 1_000_000
        if sum(item["reserved_usd"] for item in ledger.values()) + upper > float(os.getenv("LLM_BUDGET_USD", "1")):
            raise RuntimeError("Достигнут лимит оценочного бюджета; прогресс сохранён")
        ticket = str(uuid.uuid4())
        ledger[ticket] = {"reserved_usd": upper, "status": "reserved"}
        dump(self.ledger_path, ledger)
        return ticket

    def _settle(self, ticket: str, usage: dict | None) -> None:
        ledger = read(self.ledger_path)
        observed = usage_cost(usage)
        if observed is None:
            ledger[ticket]["status"] = "usage_missing"
        else:
            ledger[ticket] = {"reserved_usd": observed, "status": "observed_usage"}
        dump(self.ledger_path, ledger)

    def _send(self, request: dict, stage: str) -> dict:
        if self.transport is not None:
            return self.transport(request, stage)
        if self.client is None:
            key = os.getenv("LLM_AUTH_TOKEN", "").strip()
            if key in {"", "PASTE_YOUR_KEY_HERE", "your_token_here"}:
                raise RuntimeError("Заполни LLM_AUTH_TOKEN в локальном .env; ключ в чат не отправляй")
            from openai import OpenAI

            self.client = OpenAI(
                api_key=key,
                base_url=self.base_url,
                timeout=float(os.getenv("LLM_TIMEOUT", "120")),
                max_retries=0,
            )
        response = self.client.chat.completions.create(
            **request, extra_body={"thinking": {"type": "disabled"}}
        )
        return response.model_dump(mode="json")

    def call(self, stage: str, messages: list[dict], temperature: float, max_tokens: int = 180) -> dict:
        request = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        key = digest([stage, request, self.config])
        cache = self.output / "cache" / f"{key}.json"
        for candidate_config in [self.config, *self.compatible_cache_configs]:
            candidate_key = digest([stage, request, candidate_config])
            candidate_cache = self.output / "cache" / f"{candidate_key}.json"
            if candidate_cache.exists():
                return {
                    **read(candidate_cache)["value"],
                    "cache_replay": True,
                    "cache_key": candidate_key,
                }
        ticket = self._reserve(request)
        started = time.perf_counter()
        base_event = {
            "stage": stage,
            "cache_key": key,
            "ticket": ticket,
            "request_sha256": digest(request),
            "request": request,
        }
        try:
            response = self._send(request, stage)
        except Exception as exc:
            self._log({**base_event, "status": "api_error", "error_type": type(exc).__name__})
            raise RuntimeError(f"API: {type(exc).__name__}. Проверь ключ, баланс и сеть; прогресс сохранён") from None
        usage = response.get("usage")
        self._settle(ticket, usage)
        choice = response["choices"][0]
        text = (choice["message"].get("content") or "").strip()
        if choice.get("finish_reason") != "stop" or not text:
            self._log(
                {
                    **base_event,
                    "status": "invalid",
                    "response_id": response.get("id"),
                    "server_model": response.get("model"),
                    "finish_reason": choice.get("finish_reason"),
                    "content_chars": len(text),
                    "usage": usage,
                    "seconds": time.perf_counter() - started,
                    "cost_upper_usd": usage_cost(usage),
                }
            )
            raise RuntimeError(f"Незавершённый ответ на этапе {stage}; повторный запуск продолжит работу")
        value = {
            "stage": stage,
            "text": text,
            "response_id": response.get("id"),
            "server_model": response.get("model"),
            "usage": usage,
            "seconds": time.perf_counter() - started,
        }
        self._log({**base_event, **value, "status": "valid", "cost_upper_usd": usage_cost(usage)})
        dump(cache, {"stage": stage, "response_id": response.get("id"), "value": value})
        return {**value, "cache_replay": False, "cache_key": key}

    def finish(self, status: str) -> None:
        self.meta.update(
            status=status,
            updated_at=now(),
            active_seconds=self.meta["active_seconds"] + time.perf_counter() - self.started,
        )
        dump(self.meta_path, self.meta)

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
