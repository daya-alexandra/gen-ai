"""Адаптация JSON-клиента преподавателя из семинар_2/starter/llm_client.py.

Сохранён интерфейс make_client / response_model / max_retries. Добавлены журнал
каждой попытки, передача max_tokens и остановка при ошибках доступа/сети. Проверка
TLS включена. Скрытые повторы SDK отключены: max_retries=3 здесь означает один
первоначальный ответ и максимум три исправления ошибки JSON/Pydantic.
Источник и версия: SOURCES.md.
"""

from __future__ import annotations

import json
import os
import re
import time
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
load_dotenv(Path(__file__).with_name(".env"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.runtime import Budget, api_options, get_model as suite_model


def get_model() -> str:
    return suite_model()


def explain_error(exc: Exception) -> str:
    """Краткая диагностика без содержимого HTTP-ответа и секретов."""
    status = getattr(exc, "status_code", None)
    messages = {
        401: "API отклонил ключ (HTTP 401). Нужен действующий ключ в LLM_AUTH_TOKEN.",
        402: "На API-аккаунте недостаточно средств (HTTP 402). Обратись к владельцу учебного доступа.",
        403: "API запретил запрос (HTTP 403). Проверь права доступа у владельца ключа.",
        404: "API не нашёл ресурс (HTTP 404). Проверь LLM_BASE_URL и LLM_MODEL.",
        429: "Превышен лимит запросов (HTTP 429). Повтори запуск позже с --resume.",
    }
    if status in messages:
        return messages[status]
    if type(exc).__name__ in {"APIConnectionError", "APITimeoutError", "ConnectionError"}:
        return "Нет соединения с API или истёк таймаут. Проверь сеть и адрес API; затем используй --resume."
    if not os.getenv("LLM_AUTH_TOKEN") or os.getenv("LLM_AUTH_TOKEN") == "PASTE_YOUR_KEY_HERE":
        return "Ключ не настроен: скопируй .env.example в .env и заполни LLM_AUTH_TOKEN."
    return f"Ошибка {type(exc).__name__}. Проверь настройки и run.json; ключи в журнал не записываются."


def _extract_first_json(text: str):
    text = re.sub(r"<\|[^|>]*\|>", "", text).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char in "{[":
            try:
                obj, _ = decoder.raw_decode(text, index)
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError("В ответе модели нет JSON-объекта")


class _Completions:
    def __init__(self, raw_client, on_attempt: Callable | None = None):
        self.raw_client = raw_client
        self.on_attempt = on_attempt

    def create(self, *, model: str, messages: list[dict], response_model: type[BaseModel],
               max_retries: int = 3, temperature: float = 0.8, **kwargs):
        if max_retries < 0:
            raise ValueError("max_retries должен быть неотрицательным")
        msgs = [dict(message) for message in messages]
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
        instruction = "\nВерни только один JSON-объект по схеме, без Markdown:\n" + schema
        if msgs and msgs[0]["role"] == "system":
            msgs[0]["content"] += instruction
        else:
            msgs.insert(0, {"role": "system", "content": instruction})
        for attempt in range(max_retries + 1):
            started = time.perf_counter()
            event = {"attempt": attempt + 1, "model": model}
            try:
                request = dict(model=model, messages=msgs, response_format={"type": "json_object"}, temperature=temperature, **kwargs)
                # Real clients share the suite budget; offline test doubles never reserve funds.
                ticket = Budget().reserve(request) if isinstance(self.raw_client, OpenAI) else None
                response = self.raw_client.chat.completions.create(**request, **api_options())
                if ticket:
                    usage_data = response.usage.model_dump()
                    Budget().settle(ticket, usage_data)
            except Exception as exc:
                # Никогда не записываем заголовки запроса, ключи и текст HTTP-ошибки.
                event.update(status="api_error", error_type=type(exc).__name__,
                             status_code=getattr(exc, "status_code", None),
                             seconds=round(time.perf_counter() - started, 3))
                if self.on_attempt:
                    self.on_attempt(event)
                raise
            raw = response.choices[0].message.content or ""
            usage = response.usage
            event.update(
                response_id=response.id, raw_response=raw,
                seconds=round(time.perf_counter() - started, 3),
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
            )
            try:
                result = response_model.model_validate(_extract_first_json(raw))
            except (ValidationError, ValueError) as exc:
                errors = (exc.errors(include_url=False, include_context=False)
                          if isinstance(exc, ValidationError)
                          else [{"type": "invalid_json", "msg": str(exc)}])
                event.update(status="validation_error", errors=errors)
                if self.on_attempt:
                    self.on_attempt(event)
                if attempt == max_retries:
                    raise
                msgs.extend([
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "Исправь ошибки и верни JSON: "
                     + json.dumps(errors, ensure_ascii=False, default=str)},
                ])
            else:
                event["status"] = "valid"
                if self.on_attempt:
                    self.on_attempt(event)
                return result
        raise RuntimeError("Недостижимая ветка")


class JsonClient:
    def __init__(self, raw_client, on_attempt=None):
        self.raw_client = raw_client
        self.chat = SimpleNamespace(completions=_Completions(raw_client, on_attempt))

    def close(self):
        self.raw_client.close()


def make_client(on_attempt=None) -> JsonClient:
    base = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    key = os.getenv("LLM_AUTH_TOKEN")
    if not key or key in {"your_token_here", "PASTE_YOUR_KEY_HERE"}:
        raise RuntimeError("Укажи действующий LLM_AUTH_TOKEN в .env; см. .env.example")
    api_options()
    raw = OpenAI(api_key=key, base_url=base, timeout=float(os.getenv("LLM_TIMEOUT", "60")),
                 max_retries=0)
    return JsonClient(raw, on_attempt)
