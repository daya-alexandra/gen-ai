"""Завершить лабораторную, сохраняя исходный код и оплаченный кэш IE/MAP.

Дополнительный профиль генерации записывается в output/continuation.json.
Он меняет только ещё не завершённые REDUCE, judge и discovery. Схемы,
промпты, ключи кэша и ответы этапов IE, аспектов и MAP остаются прежними.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Annotated

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pydantic import ConfigDict, Field, StringConstraints
import pipeline
import prompts
from llm_client import Run, code_hashes, digest, dump, events, make_client, now, read
from schema import (Action, ActionVerdict, Claim, Discovery, DiscoveredAspect,
                    Distortion, Evidence, JudgeReport, MapSummary, Reviews,
                    Aspects, Summary)
from verify_results import verify

ORIGINAL_ASK = pipeline.ask
SHORT_NOTE = Annotated[str, StringConstraints(min_length=8, max_length=400)]


class CompactClaim(Claim):
    text: str = Field(min_length=8, max_length=500)
    evidence: list[Evidence] = Field(min_length=1, max_length=2)


class CompactAction(Action):
    text: str = Field(min_length=8, max_length=500)
    evidence: list[Evidence] = Field(min_length=1, max_length=2)


class CompactSummary(Summary):
    model_config = ConfigDict(extra='forbid', title='Summary')
    findings: list[CompactClaim] = Field(min_length=3, max_length=5)
    action_items: list[CompactAction] = Field(min_length=2, max_length=3)
    limitations: list[SHORT_NOTE] = Field(min_length=1, max_length=5)


class CompactVerdict(ActionVerdict):
    explanation: str = Field(min_length=10, max_length=600)
    evidence: list[Evidence] = Field(max_length=2)


class CompactDistortion(Distortion):
    correction: str = Field(min_length=8, max_length=600)


class CompactJudge(JudgeReport):
    model_config = ConfigDict(extra='forbid', title='JudgeReport')
    verdicts: list[CompactVerdict] = Field(max_length=5)
    distortions: list[CompactDistortion] = Field(max_length=5)
    feedback: list[SHORT_NOTE] = Field(max_length=5)


class CompactAspect(DiscoveredAspect):
    description: str = Field(min_length=10, max_length=400)
    evidence: list[Evidence] = Field(min_length=1, max_length=2)


class CompactDiscovery(Discovery):
    model_config = ConfigDict(extra='forbid', title='Discovery')
    aspects: list[CompactAspect] = Field(min_length=3, max_length=6)


REDUCE_NOTE = '''Сформируй компактный итог, а не пересказ всех MAP-фактов подряд.
В findings строго от 3 до 5 сгруппированных наблюдений, в action_items от 2 до 3
действий. У каждого пункта не больше двух evidence. Цитаты копируй целиком из
MAP: сокращать или склеивать их нельзя. Выбери подходящие короткие цитаты.
Объединяй только совместимые наблюдения, сохраняя различия платформ и оговорки.
Не объявляй единичный случай общим правилом и не скрывай противоречия.
Если места недостаточно, сократи число пунктов до минимума схемы, но обязательно
заверши весь JSON, включая action_items и limitations. Не перечисляй по одному
пункту на каждый отзыв. Размер ответа ограничен; не добавляй пояснений вне JSON.'''
JUDGE_NOTE = '''Дай компактный, но критический вердикт каждому action_id.
До двух цитат на вердикт, до пяти искажений и до пяти замечаний. Не повышай
оценку ради прохождения порога: используй исходную рубрику и фактическую опору.
Не придумывай искажений. Полностью заверши JSON и не пересказывай все отзывы.'''
DISCOVER_NOTE = '''Верни от 3 до 6 различимых тем, до двух цитат на тему.
Кратко объясни отличие темы от фиксированных аспектов. Не дроби одну тему
на синонимы и не пересказывай весь корпус. Полностью заверши JSON.'''
PROFILES = {
    Summary: (CompactSummary, 12000, REDUCE_NOTE),
    JudgeReport: (CompactJudge, 10000, JUDGE_NOTE),
    Discovery: (CompactDiscovery, 8000, DISCOVER_NOTE),
}


def policy():
    return {
        'name': 'compact_continuation_v1',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'profiles': {model.__name__: {
            'max_tokens': limit, 'schema_sha256': digest(compact.model_json_schema()),
            'additional_instruction': note,
        } for model, (compact, limit, note) in PROFILES.items()},
        'unchanged_stages': ['ie_*', 'aspects_*', 'map_*'],
        'max_retries': 3,
    }


def prefix_requests(sources):
    for i in range(0, len(sources), 5):
        part = sources[i:i+5]
        payload = [r.model_dump() for r in part]
        yield f'ie_{i//5:02d}', Reviews, prompts.IE, payload, part
        yield f'aspects_{i//5:02d}', Aspects, prompts.ASPECT, payload, part
    for i, part in enumerate(pipeline.split_chunks(sources)):
        yield f'map_{i:02d}', MapSummary, prompts.MAP, [r.model_dump() for r in part], part


def audit_prefix(output, sources):
    """Сопоставить прежние ключи кэша, принятые ответы и исходные цитаты."""
    out = Path(output)
    if not (out/'run.json').exists():
        return []
    meta = read(out/'run.json')
    if meta['config']['code_sha256'] != code_hashes():
        raise ValueError('Базовые .py-файлы изменились. Для продолжения нужна исходная версия кода.')
    if meta['config']['input_sha256'] != digest([s.model_dump() for s in sources]):
        raise ValueError('Исходные отзывы изменились; нельзя использовать прежний кэш.')
    accepted = {e['response_id']: e for e in events(out/'api_trace.jsonl') if e['status']=='valid'}
    verified = []
    for stage, model, system, payload, part in prefix_requests(sources):
        msgs = [{'role':'system', 'content':'Верни один JSON-объект без Markdown по схеме: '
                 + json.dumps(model.model_json_schema(), ensure_ascii=False)},
                {'role':'system', 'content':system},
                {'role':'user', 'content':json.dumps(payload, ensure_ascii=False)}]
        key = digest([stage, msgs, .2, 5000, meta['config']])
        path = out/'cache'/(key+'.json')
        if not path.exists():
            continue
        cached = read(path)
        event = accepted.get(cached['response_id'])
        if event is None or event['cache_key'] != key or event['stage'] != stage or cached['stage'] != stage:
            raise ValueError('Несоответствие кэша и журнала: '+stage)
        value = model.model_validate_json(event['raw_response'])
        if cached['value'] != value.model_dump(mode='json'):
            raise ValueError('Сохранённое значение отличается от ответа модели: '+stage)
        pipeline.validate_result(value, part)
        verified.append(stage)
    return verified


def record_continuation(run, sources):
    path = run.output/'continuation.json'
    expected = policy()
    if path.exists():
        manifest = read(path)
        if manifest['policy'] != expected or manifest['run_id'] != run.meta['run_id']:
            raise ValueError('Профиль продолжения изменился. Сохрани прежнюю версию или выбери новый --output.')
        verify_preserved_prefix(run.output, manifest)
        return manifest
    cached = audit_prefix(run.output, sources)
    log_bytes = run.trace.read_bytes() if run.trace.exists() else b''
    preserved = {p.relative_to(run.output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((run.output/'cache').glob('*.json'))}
    manifest = {
        'run_id':run.meta['run_id'], 'run_kind':run.config['run_kind'],
        'started_at':now(), 'policy':expected, 'base_config':run.config,
        'before_run':read(run.meta_path),
        'before_budget':read(run.ledger) if run.ledger.exists() else {},
        'preserved_cache_sha256':preserved, 'verified_cached_stages':cached,
        'trace_prefix_bytes':len(log_bytes),
        'trace_prefix_sha256':hashlib.sha256(log_bytes).hexdigest(),
        'status':'started',
    }
    dump(path, manifest)
    print(f'Продолжение: проверено сохранённых этапов IE/аспектов/MAP — {len(cached)}.', flush=True)
    return manifest


def verify_preserved_prefix(output, manifest):
    out=Path(output)
    for rel, expected in manifest['preserved_cache_sha256'].items():
        path=Path(rel)
        if path.is_absolute() or '..' in path.parts or len(path.parts)!=2 or path.parts[0]!='cache':
            raise ValueError('Некорректный путь сохранённого кэша')
        if hashlib.sha256((out/path).read_bytes()).hexdigest()!=expected:
            raise ValueError('Изменён ранее принятый ответ: '+rel)
    trace=(out/'api_trace.jsonl').read_bytes() if (out/'api_trace.jsonl').exists() else b''
    prefix=trace[:manifest['trace_prefix_bytes']]
    if len(prefix)!=manifest['trace_prefix_bytes'] or hashlib.sha256(prefix).hexdigest()!=manifest['trace_prefix_sha256']:
        raise ValueError('Изменён журнал исходного прогона')


def compact_ask(run, stage, model, system, payload, sources, **constraints):
    if model not in PROFILES:
        return ORIGINAL_ASK(run, stage, model, system, payload, sources, **constraints)
    compact, max_tokens, note = PROFILES[model]
    return make_client(run).chat.completions.create(
        model=run.model, stage=stage, response_model=compact, max_retries=3,
        temperature=0 if model is JudgeReport else .2, max_tokens=max_tokens,
        messages=[{'role':'system','content':system},
                  {'role':'user','content':json.dumps(payload,ensure_ascii=False)},
                  {'role':'user','content':note}],
        validate=lambda value:pipeline.validate_result(value,sources,**constraints))


def continue_run(input_path, output=None, *, transport=None, diagnostic=True):
    sources=pipeline.load_input(input_path)
    out=Path(output or ROOT/'output')
    # Проверка до загрузки/дозаписи результатов. Обычный Run дополнительно
    # проверяет всю исходную конфигурацию, модель и режим API/заглушки.
    audit_prefix(out,sources)
    if (out/'continuation.json').exists():
        manifest=read(out/'continuation.json')
        if manifest['policy']!=policy():
            raise ValueError('Профиль продолжения изменился; прежний результат сохранён без изменений.')
        verify_preserved_prefix(out,manifest)
    recorded=False
    original=pipeline.ask

    def routed(run,stage,model,system,payload,part,**constraints):
        nonlocal recorded
        if not recorded:
            record_continuation(run,sources)
            recorded=True
        return compact_ask(run,stage,model,system,payload,part,**constraints)

    pipeline.ask=routed
    try:
        result=pipeline.analyze(input_path,out,transport=transport,diagnostic=diagnostic)
    finally:
        pipeline.ask=original
        if recorded:
            manifest=read(out/'continuation.json')
            verify_preserved_prefix(out,manifest)
            manifest.update(status=read(out/'run.json')['status'],updated_at=now())
            dump(out/'continuation.json',manifest)
    write_notes(out)
    return result


def write_notes(output):
    out=Path(output);m=read(out/'continuation.json')
    text = ('# Продолжение после остановки REDUCE\n\n'
            f'Режим: **{m["run_kind"]}**. Сохранённых этапов перед продолжением: '
            f'{len(m["verified_cached_stages"])}.\n\n'
            'В исходной попытке REDUCE ответ мог достигнуть лимита 5000 токенов. '
            'Добавлен явно зафиксированный профиль: 3–5 наблюдений, 2–3 действия, '
            'не более двух цитат на пункт, до 12000 выходных токенов для сводки. '
            'Для судьи и autodiscovery ограничена многословность; критерии оценки, '
            'порог 0,7 и проверка цитат сохранены. При низкой оценке выполняется '
            'обычная повторная сводка с обратной связью.\n\n'
            'Оплаченные ответы IE, аспектов и MAP не редактировались и не переносились '
            'под новыми идентификаторами. Кэш и префикс журнала проверяются по SHA-256. '
            'В run.json сохранены хеши базового конвейера; дополнительный исполняемый '
            'код, схемы и инструкции записаны в continuation.json. Ключи новых ответов '
            'учитывают изменённую схему, инструкции и лимит токенов.\n\n'
            'Метрики и стоимость после завершения включают оба этапа запуска: '
            'предыдущие неудачные ответы также остаются в api_trace.jsonl и budget.json. '
            'Тестовые проверки продолжения не являются ответами DeepSeek.\n')
    (out/'continuation_notes.md').write_text(text,encoding='utf-8')


def verify_continuation(output, input_path, *, allow_test=False):
    out=Path(output)
    result=verify(out,input_path,allow_test=allow_test)
    manifest=read(out/'continuation.json')
    meta=read(out/'run.json')
    if manifest['policy']!=policy() or manifest['run_id']!=meta['run_id'] or manifest['base_config']!=meta['config']:
        raise ValueError('Профиль продолжения не соответствует коду или исходному прогону')
    if manifest['run_kind']!=meta['config']['run_kind']:
        raise ValueError('Не совпадает режим происхождения ответов')
    verify_preserved_prefix(out,manifest)
    sources=pipeline.load_input(input_path)
    cached=audit_prefix(out,sources)
    if len(cached)!=len(list(prefix_requests(sources))):
        raise ValueError('Не все этапы IE/аспектов/MAP присутствуют в кэше')
    # Ключи новых запросов отдельно сверяются с компактными схемами и инструкциями.
    logs=events(out/'api_trace.jsonl')
    accepted={e['response_id']:e for e in logs if e['status']=='valid'}
    for p in (out/'cache').glob('*.json'):
        item=read(p);stage=item['stage']
        original_model=Summary if stage.startswith('reduce_') else JudgeReport if stage.startswith(('judge_','diagnostic_judge')) else Discovery if stage=='discovery' else None
        if original_model:
            compact,limit,note=PROFILES[original_model]
            compact.model_validate_json(accepted[item['response_id']]['raw_response'])
            if stage=='reduce_v1':
                system=prompts.REDUCE;payload=read(out/'map_summaries.json')
            elif stage=='reduce_v2':
                system=prompts.REDUCE_REVISED
                payload={'map_summaries':read(out/'map_summaries.json'),
                         'previous_summary':read(out/'summary_v1.json'),
                         'judge_feedback':read(out/'judge_v1.json')}
            elif stage.startswith('judge_') or stage=='diagnostic_judge':
                system=prompts.JUDGE
                summary_file='diagnostic_summary.json' if stage=='diagnostic_judge' else 'summary_'+stage.split('_')[1]+'.json'
                payload={'sources':[s.model_dump() for s in sources],'summary':read(out/summary_file)}
            else:
                system=prompts.DISCOVER;payload=[s.model_dump() for s in sources]
            msgs=[{'role':'system','content':'Верни один JSON-объект без Markdown по схеме: '
                   +json.dumps(compact.model_json_schema(),ensure_ascii=False)},
                  {'role':'system','content':system},
                  {'role':'user','content':json.dumps(payload,ensure_ascii=False)},
                  {'role':'user','content':note}]
            key=digest([stage,msgs,0 if original_model is JudgeReport else .2,limit,meta['config']])
            if p.stem!=key:
                raise ValueError('Ключ ответа не соответствует профилю продолжения: '+stage)
    result.update(continuation_verified=True, previously_cached_stages=len(manifest['verified_cached_stages']))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default=str(ROOT/'input'))
    parser.add_argument('--output',default=str(ROOT/'output'))
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    try:
        if not args.verify_only:
            continue_run(args.input,args.output)
        print(json.dumps(verify_continuation(args.output,args.input),ensure_ascii=False,indent=2))
        return 0
    except (ValueError,RuntimeError,AssertionError,KeyError,FileNotFoundError) as exc:
        print('Продолжение остановлено: '+str(exc),file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
