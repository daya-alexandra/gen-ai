"""Хеши входов, сохранение заданий и проверка журналов без сети."""
import hashlib
import json
import math
import re
import os
from contextlib import contextmanager
from pathlib import Path
from llm_client import Run, dump, read, digest, events, usage_cost

ROOT = Path(__file__).resolve().parent

@contextmanager
def experiment_lock(output):
    """ОС снимает блокировку и при закрытии окна; второй запуск не тратит API."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    path = output.parent / ('.' + output.name + '.run.lock')
    f = path.open('a+b')
    f.seek(0, 2)
    if f.tell() == 0:
        f.write(b'0'); f.flush()
    f.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise RuntimeError('Этот эксперимент уже запущен в другом окне. Дождись его завершения.') from None
    try:
        # Только собственные незавершённые атомарные записи от прерванного процесса.
        if output.exists():
            for tmp in output.rglob('*.tmp'):
                if re.search(r'\.[0-9a-f]{32}\.tmp$', tmp.name):
                    tmp.unlink()
        yield
    finally:
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()

def input_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / 'data').glob('*.csv'))}

def make_run(output, suite, settings, transport=None):
    r = Run(Path(output) / 'generation', {'suite': suite, 'settings': settings, 'inputs': input_hashes()}, transport=transport)
    if suite != 'interactive' and r.config['data_mode'] != 'snapshot':
        raise ValueError('Сравнительный eval использует MACRO_DATA_MODE=snapshot. Live — только для отдельных запросов.')
    return r

def cached_task(output, stage, operation):
    path = Path(output) / 'tasks' / (digest(stage) + '.json')
    if path.exists():
        saved = read(path)
        if saved['stage'] != stage:
            raise ValueError('Несовпадение этапа')
        if saved['sha256'] != digest(saved['result']):
            raise ValueError('Результат этапа изменён')
        return saved['result']
    result = operation()
    dump(path, {'stage': stage, 'sha256': digest(result), 'result': result})
    return result

def numbers(text):
    return [float(x.replace(',', '.')) for x in re.findall(r'(?<![\w])[-+]?\d+(?:[.,]\d+)?', text or '')]

def has_number(text, target, atol=.025):
    return any(math.isclose(n, target, abs_tol=atol, rel_tol=.003) for n in numbers(text))

def api_totals(output):
    log = [e for e in events(Path(output) / 'generation/api_trace.jsonl') if e.get('response_id')]
    upper = sum(usage_cost(e.get('usage')) or 0 for e in log)
    return {'api_responses': len(log), 'missing_usage': sum(usage_cost(e.get('usage')) is None for e in log),
            'prompt_tokens': sum((e.get('usage') or {}).get('prompt_tokens', 0) for e in log),
            'completion_tokens': sum((e.get('usage') or {}).get('completion_tokens', 0) for e in log),
            'estimated_usd_interval': [upper * .5, upper],
            'format_errors': sum(e['status'] in ('validation_error', 'native_invalid') for e in log)}

def verify_api(output, require_api=True, models=None):
    output = Path(output)
    meta = read(output / 'generation/run.json')
    from llm_client import code_hashes
    assert meta['status'] == 'completed', 'Прогон не завершён'
    assert meta['config']['code_sha256'] == code_hashes(), 'Код изменён'
    assert meta['config']['inputs'] == input_hashes(), 'Данные изменены'
    if require_api:
        assert meta['config']['run_kind'] == 'llm_api', 'Это локальный тест, не API-прогон'
    log = events(output / 'generation/api_trace.jsonl')
    valid = {e['cache_key']: e for e in log if e['status'] in ('valid', 'native_valid')}
    seen = set()
    for e in log:
        assert e['run_id'] == meta['run_id']
        assert e.get('request_sha256') == digest(e['request'])
        if e.get('response_id'):
            assert e['response_id'] not in seen, 'Повтор response_id'
            seen.add(e['response_id'])
    caches = {}
    for p in (output / 'generation/cache').glob('*.json'):
        c = read(p)
        assert p.stem in valid, 'Кэш без ответа в журнале'
        e = valid[p.stem]
        assert c['stage'] == e['stage'] and c['response_id'] == e['response_id']
        if e['status'] == 'native_valid':
            expected = e['raw_response']
        else:
            model = (models or {}).get(e.get('response_schema'))
            assert model is not None, 'Неизвестная схема структурированного ответа'
            expected = model.model_validate_json(e['raw_response']).model_dump(mode='json')
        assert c['value'] == expected
        caches[p.stem] = c
    for p in (output / 'tasks').glob('*.json'):
        item = read(p)
        assert item['sha256'] == digest(item['result']), 'Результат этапа изменён'
    manifest = read(output / 'artifact_manifest.json')
    for name, expected in manifest.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected, 'Изменён файл ' + name
    assert seen, 'Нет ответов API'
    return meta, caches

def seal_output(output):
    output = Path(output)
    manifest = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'artifact_manifest.json'}
    dump(output / 'artifact_manifest.json', manifest)

def verify_agent(result, caches):
    from schemas import dispatch
    expected_prefix = result['stage'] + '/step'
    for key in result['api_cache_keys']:
        assert key in caches and caches[key]['stage'].startswith(expected_prefix), 'Подменён этап API'
    for event in result['trace']:
        if 'call' not in event:
            continue
        cache = caches[event['api_cache_key']]
        call = next((c for c in cache['value'].get('tool_calls', []) if c['id'] == event['tool_call_id']), None)
        assert call and call['function']['name'] == event['call'], 'Вызова нет в ответе API'
        try:
            args = json.loads(call['function']['arguments'])
        except (ValueError, TypeError):
            assert event['obs'].get('error_type') == 'invalid_arguments_json'
            continue
        assert args == event['args']
        observed = dispatch(event['call'], args, result.get('allowed_tools'))
        assert observed == event['obs'], 'Наблюдение инструмента изменено'
    if result.get('answer'):
        last = caches[result['api_cache_keys'][-1]]['value']
        assert not last.get('tool_calls') and last.get('content') == result['answer'], 'Ответ изменён'
