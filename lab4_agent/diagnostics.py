"""Намеренные сбои: воспроизводимая диагностика, НЕ ответы DeepSeek."""
import uuid
from pathlib import Path
from agent import run_agent
from llm_client import dump
from suite_common import make_run

FAULTS = [
    {'id': 'broken_json', 'name': 'calculate', 'args': '{"expression":', 'type': 'invalid_arguments_json',
     'cause': 'Оборванная JSON-строка аргументов; до исполнения функции дело не дошло.',
     'fix': 'Проверять JSON и схему аргументов, возвращать модели observation с ошибкой.'},
    {'id': 'phantom_tool', 'name': 'get_cumulative_inflation', 'args': '{}', 'type': 'unknown_tool',
     'cause': 'Синтетический ответ просит инструмент, которого нет в реестре.',
     'fix': 'Белый список функций и обратная связь; для планов — валидатор до Исполнителя.'},
    {'id': 'division_zero', 'name': 'calculate', 'args': '{"expression":"1/0"}', 'type': 'tool_error',
     'cause': 'Схема корректна, но вычисление математически не определено.',
     'fix': 'Проверять знаменатель, ловить исключение в инструменте и не выдавать число.'},
]

def run_diagnostics(output):
    output = Path(output)
    records = []
    for fault in FAULTS:
        def transport(request, fault=fault):
            if any(m['role'] == 'tool' for m in request['messages']):
                message = {'role': 'assistant', 'content': 'Диагностический вызов завершился ошибкой. Это контролируемый локальный тест.'}
                finish = 'stop'
            else:
                message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'diagnostic_call', 'type': 'function',
                            'function': {'name': fault['name'], 'arguments': fault['args']}}]}
                finish = 'tool_calls'
            return {'id': 'offline-' + uuid.uuid4().hex, 'model': 'scripted_fault', 'choices': [{'message': message, 'finish_reason': finish}],
                    'usage': {'prompt_tokens': 0, 'completion_tokens': 0}}
        run = make_run(output / fault['id'], 'controlled_fault', {'fault': fault['id']}, transport=transport)
        result = run_agent('Контролируемая инъекция ' + fault['id'], run=run, stage=fault['id'],
                           trace_path=output / 'trace.jsonl', run_kind='controlled_test')
        assert any(e.get('obs', {}).get('error_type') == fault['type'] for e in result['trace'])
        records.append({**fault, 'result': result})
        run.finish('completed')
        run.close()
    dump(output / 'diagnostics.json', {'run_kind': 'controlled_test', 'api_calls': 0, 'cases': records})
    return records
