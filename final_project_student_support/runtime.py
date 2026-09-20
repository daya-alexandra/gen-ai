"""API-runtime проекта: строгий JSON, повторы, кэш, журнал и бюджет."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError


ROOT = Path(__file__).resolve().parent
TARIFF = {
    "checked_on": "2026-09-19",
    "source": "https://api-docs.deepseek.com/quick_start/pricing/",
    "input_miss_per_million": 0.30,
    "input_hit_per_million": 0.006,
    "output_per_million": 1.20,
    "off_peak_multiplier": 0.5,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def dump(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def events(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_hashes() -> dict[str, str]:
    paths = sorted(ROOT.glob("*.py")) + sorted(path for path in (ROOT / "input").rglob("*") if path.is_file())
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def usage_cost(usage: dict | None) -> float | None:
    if not usage or usage.get("prompt_tokens") is None or usage.get("completion_tokens") is None:
        return None
    prompt = usage["prompt_tokens"]
    cached = min(prompt, usage.get("prompt_cache_hit_tokens", 0) or 0)
    return ((prompt - cached) * 0.30 + cached * 0.006 + usage["completion_tokens"] * 1.20) / 1_000_000


def parse_json_object(text: str):
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    return json.loads(text)


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
            raise ValueError("Для проекта разрешён официальный адрес DeepSeek API")
        if self.model not in {"deepseek-flash", "deepseek-v4-flash"}:
            raise ValueError("Используй модель DeepSeek Flash, для которой рассчитан бюджет")
        self.config = {
            "suite": "final_project_student_support",
            "model": self.model,
            "base_url": self.base_url,
            "run_kind": "offline_test" if transport else "llm_api",
            "files_sha256": file_hashes(),
            "tariff": TARIFF,
        }
        self.meta_path = self.output / "run.json"
        self.trace_path = self.output / "api_trace.jsonl"
        self.validation_path = self.output / "validation_trace.jsonl"
        self.ledger_path = self.output / "budget.json"
        if self.meta_path.exists():
            self.meta = read(self.meta_path)
            if self.meta["config"] != self.config:
                raise ValueError("Код или входные данные изменились. Выбери новую папку --output.")
        else:
            self.meta = {
                "run_id": str(uuid.uuid4()),
                "created_at": now(),
                "status": "prepared",
                "active_seconds": 0.0,
                "config": self.config,
            }
            dump(self.meta_path, self.meta)
        self.started = time.perf_counter()

    def _append(self, path: Path, event: dict) -> None:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"run_id": self.meta["run_id"], "ts": now(), **event}, ensure_ascii=False, default=str) + "\n")

    def _reserve(self, request: dict) -> str:
        ledger = read(self.ledger_path) if self.ledger_path.exists() else {}
        if len(ledger) >= int(os.getenv("LLM_MAX_CALLS", "160")):
            raise RuntimeError("Достигнут лимит API-вызовов; прогресс сохранён")
        rough_input = len(json.dumps(request, ensure_ascii=False).encode("utf-8")) / 3
        upper = (rough_input * 0.30 + request["max_tokens"] * 1.20) / 1_000_000
        if sum(item["reserved_usd"] for item in ledger.values()) + upper > float(os.getenv("LLM_BUDGET_USD", "3")):
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
            **request,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return response.model_dump(mode="json")

    def _call_text(self, stage: str, messages: list[dict], temperature: float, max_tokens: int, schema_sha256: str) -> dict:
        request = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        key = digest([stage, request, schema_sha256, self.config])
        cache = self.output / "cache" / f"{key}.json"
        if cache.exists():
            return {**read(cache)["value"], "cache_replay": True, "cache_key": key}
        ticket = self._reserve(request)
        started = time.perf_counter()
        base_event = {
            "stage": stage,
            "cache_key": key,
            "ticket": ticket,
            "request_sha256": digest(request),
            "request": request,
            "schema_sha256": schema_sha256,
        }
        try:
            response = self._send(request, stage)
        except Exception as error:
            self._append(self.trace_path, {**base_event, "status": "api_error", "error_type": type(error).__name__})
            raise RuntimeError(f"API: {type(error).__name__}. Проверь ключ, баланс и сеть; прогресс сохранён") from None
        usage = response.get("usage")
        self._settle(ticket, usage)
        choice = response["choices"][0]
        text = (choice["message"].get("content") or "").strip()
        if choice.get("finish_reason") != "stop" or not text:
            self._append(self.trace_path, {**base_event, "status": "invalid_response", "response_id": response.get("id")})
            raise RuntimeError(f"Незавершённый ответ на этапе {stage}; повторный запуск продолжит работу")
        value = {
            "stage": stage,
            "text": text,
            "response_id": response.get("id"),
            "server_model": response.get("model"),
            "usage": usage,
            "seconds": time.perf_counter() - started,
        }
        self._append(self.trace_path, {**base_event, **value, "status": "api_response", "cost_upper_usd": usage_cost(usage)})
        dump(cache, {"stage": stage, "response_id": response.get("id"), "value": value})
        return {**value, "cache_replay": False, "cache_key": key}

    def call_json(
        self,
        stage: str,
        messages: list[dict],
        response_model,
        post_validate=None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        attempts: int = 3,
    ) -> dict:
        schema = response_model.model_json_schema()
        schema_sha256 = digest(schema)
        conversation = list(messages)
        last_error = None
        for attempt in range(1, attempts + 1):
            raw = self._call_text(f"{stage}/try{attempt}", conversation, temperature, max_tokens, schema_sha256)
            try:
                parsed = parse_json_object(raw["text"])
                value = response_model.model_validate(parsed)
                if post_validate is not None:
                    post_validate(value)
                self._append(
                    self.validation_path,
                    {"stage": stage, "attempt": attempt, "status": "schema_valid", "cache_key": raw["cache_key"]},
                )
                return {"data": value.model_dump(mode="json"), "api": raw, "attempt": attempt}
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                last_error = f"{type(error).__name__}: {error}"
                self._append(
                    self.validation_path,
                    {"stage": stage, "attempt": attempt, "status": "schema_invalid", "cache_key": raw["cache_key"], "error": last_error[:1200]},
                )
                if attempt < attempts:
                    conversation += [
                        {"role": "assistant", "content": raw["text"]},
                        {
                            "role": "user",
                            "content": "Исправь JSON и верни только полный объект. Ошибка проверки: " + last_error[:700],
                        },
                    ]
        raise RuntimeError(f"Не удалось получить валидный JSON на этапе {stage}: {last_error}")

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
