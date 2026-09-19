"""JSON-клиент с журналом, возобновлением и явными повторами валидации."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent
LOCK = threading.RLock()
TARIFF = {'checked_on': '2026-09-19', 'source': 'https://api-docs.deepseek.com/quick_start/pricing/',
          'peak_input_miss_per_million': .30, 'peak_input_hit_per_million': .006,
          'peak_output_per_million': 1.20, 'off_peak_multiplier': .5}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def events(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.exists() else []


def code_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))}


def usage_cost(usage):
    if not usage or usage.get('prompt_tokens') is None or usage.get('completion_tokens') is None:
        return None
    inp, out = usage['prompt_tokens'], usage['completion_tokens']
    hit = min(inp, usage.get('prompt_cache_hit_tokens', 0) or 0)
    return ((inp - hit) * .30 + hit * .006 + out * 1.20) / 1_000_000


class GroundingError(ValueError):
    def __init__(self, message, checks):
        super().__init__(message)
        self.checks = checks


class Run:
    def __init__(self, output, config, transport=None):
        # Тестовый транспорт использует только явно заданное окружение.
        # Локальный .env не должен подменять лимиты теста или загружать его ключ.
        if transport is None:
            try:
                from dotenv import load_dotenv
                load_dotenv(ROOT.parent / '.env')
                load_dotenv(ROOT / '.env', override=True)
            except ImportError:
                raise RuntimeError('Установи зависимости из requirements.txt') from None
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.output / 'run.json'
        self.trace = self.output / 'api_trace.jsonl'
        self.ledger = self.output / 'budget.json'
        self.transport = transport
        self.raw = None
        self.model = os.getenv('LLM_MODEL', 'deepseek-flash')
        self.base_url = os.getenv('LLM_BASE_URL', 'https://api.deepseek.com').rstrip('/')
        if self.base_url not in {'https://api.deepseek.com', 'https://api.deepseek.com/v1'}:
            raise ValueError('Разрешён официальный адрес DeepSeek API; проверь LLM_BASE_URL')
        if self.model not in {'deepseek-flash', 'deepseek-v4-flash'}:
            raise ValueError('Тариф рассчитан для DeepSeek Flash: используй LLM_MODEL=deepseek-flash')
        self.config = {**config, 'model': self.model, 'base_url': self.base_url, 'thinking': 'disabled',
                       'run_kind': 'offline_test' if transport else 'llm_api',
                       'code_sha256': code_hashes(), 'tariff': TARIFF}
        if self.meta_path.exists():
            self.meta = read(self.meta_path)
            if self.meta['config'] != self.config:
                raise ValueError('Код, входные данные или настройки изменились. Выбери новую папку --output.')
        else:
            self.meta = {'run_id': str(uuid.uuid4()), 'created_at': now(), 'status': 'prepared',
                         'config': self.config, 'active_seconds': 0.0}
            dump(self.meta_path, self.meta)
        self.session_start = time.perf_counter()

    def log(self, event):
        with LOCK, self.trace.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'run_id': self.meta['run_id'], 'ts': now(), **event}, ensure_ascii=False) + '\n')

    def reserve(self, request):
        with LOCK:
            ledger = read(self.ledger) if self.ledger.exists() else {}
            upper = (len(json.dumps(request, ensure_ascii=False).encode()) + 4096) * .30 / 1e6
            upper += request['max_tokens'] * 1.20 / 1e6
            if len(ledger) >= int(os.getenv('LLM_MAX_CALLS', '120')):
                raise RuntimeError('Достигнут лимит API-вызовов. Прогресс сохранён.')
            if sum(x['reserved_usd'] for x in ledger.values()) + upper > float(os.getenv('LLM_BUDGET_USD', '1')):
                raise RuntimeError('Достигнут лимит оценочного бюджета. Прогресс сохранён.')
            ticket = str(uuid.uuid4())
            ledger[ticket] = {'reserved_usd': upper, 'status': 'reserved'}
            dump(self.ledger, ledger)
            return ticket

    def settle(self, ticket, usage):
        with LOCK:
            ledger = read(self.ledger)
            cost = usage_cost(usage)
            if cost is not None:
                ledger[ticket] = {'reserved_usd': cost, 'status': 'observed_usage'}
            else:
                ledger[ticket]['status'] = 'usage_missing'
            dump(self.ledger, ledger)

    def send(self, request):
        if self.transport is not None:
            return self.transport(request)
        with LOCK:
            if self.raw is None:
                key = os.getenv('LLM_AUTH_TOKEN', '')
                if not key or key in {'PASTE_YOUR_KEY_HERE', 'your_token_here'}:
                    raise RuntimeError('Заполни LLM_AUTH_TOKEN в локальном .env. Ключ в чат отправлять не нужно.')
                from openai import OpenAI
                self.raw = OpenAI(api_key=key, base_url=self.base_url,
                                  timeout=float(os.getenv('LLM_TIMEOUT', '120')), max_retries=0)
        response = self.raw.chat.completions.create(**request, extra_body={'thinking': {'type': 'disabled'}})
        return response.model_dump(mode='json')

    def create(self, *, stage, messages, response_model, max_retries=3, validate=None,
               temperature=.2, max_tokens=4500, model=None):
        if not 0 <= max_retries <= 3:
            raise ValueError('Разрешено от 0 до 3 повторов валидации')
        if model is not None and model != self.model:
            raise ValueError('Модель запроса должна совпадать с моделью эксперимента')
        msgs = [dict(x) for x in messages]
        msgs.insert(0, {'role': 'system', 'content': 'Верни один JSON-объект без Markdown по схеме: '
                       + json.dumps(response_model.model_json_schema(), ensure_ascii=False)})
        key = digest([stage, msgs, temperature, max_tokens, self.config])
        cache_path = self.output / 'cache' / (key + '.json')
        if cache_path.exists():
            result = response_model.model_validate(read(cache_path)['value'])
            if validate:
                validate(result)
            return result
        for attempt in range(1, max_retries + 2):
            request = {'model': self.model, 'messages': msgs, 'temperature': temperature,
                       'max_tokens': max_tokens, 'response_format': {'type': 'json_object'}}
            ticket = self.reserve(request)
            started = time.perf_counter()
            event = {'stage': stage, 'attempt': attempt, 'cache_key': key,
                     'request_sha256': digest(request)}
            try:
                response = self.send(request)
            except Exception as exc:
                self.log({**event, 'status': 'api_error', 'error_type': type(exc).__name__,
                          'http_status': getattr(exc, 'status_code', None),
                          'seconds': time.perf_counter() - started})
                # При таймауте сервер мог выполнить запрос: резерв не обнуляется.
                raise RuntimeError(f'API: {type(exc).__name__}, HTTP {getattr(exc, "status_code", "—")}. '
                                   'Проверь ключ, баланс и сеть. Повторный запуск продолжит работу.') from None
            usage = response.get('usage')
            self.settle(ticket, usage)
            choice = response['choices'][0]
            raw = choice['message'].get('content') or ''
            event.update(response_id=response.get('id'), server_model=response.get('model'),
                         usage=usage, cost_upper_usd=usage_cost(usage), raw_response=raw,
                         finish_reason=choice.get('finish_reason'), seconds=time.perf_counter() - started,
                         quote_checks=[])
            try:
                if choice.get('finish_reason') != 'stop':
                    raise ValueError('Ответ не завершён: finish_reason=' + str(choice.get('finish_reason')))
                value = response_model.model_validate_json(raw)
                if validate:
                    event['quote_checks'] = validate(value)
            except (ValidationError, ValueError) as exc:
                if isinstance(exc, GroundingError):
                    event['quote_checks'] = exc.checks
                errors = (exc.errors(include_url=False, include_context=False, include_input=False)
                          if isinstance(exc, ValidationError) else [{'msg': str(exc)}])
                self.log({**event, 'status': 'validation_error', 'error_type': type(exc).__name__, 'errors': errors})
                if attempt == max_retries + 1:
                    raise RuntimeError(f'{stage}: не удалось получить валидный ответ за {attempt} попытки') from None
                msgs += [{'role': 'assistant', 'content': raw},
                         {'role': 'user', 'content': 'Исправь ошибки и верни полный JSON: ' + json.dumps(errors, ensure_ascii=False)}]
            else:
                self.log({**event, 'status': 'valid'})
                dump(cache_path, {'stage': stage, 'response_id': response.get('id'),
                                  'value': value.model_dump(mode='json')})
                return value

    def finish(self, status, **extra):
        self.meta.update(status=status, updated_at=now(),
                         active_seconds=self.meta['active_seconds'] + time.perf_counter() - self.session_start, **extra)
        dump(self.meta_path, self.meta)
        self.session_start = time.perf_counter()

    def close(self):
        if self.raw:
            self.raw.close()


def make_client(run):
    """Совместимый с семинаром интерфейс response_model и max_retries=3."""
    return SimpleNamespace(chat=SimpleNamespace(completions=run))
