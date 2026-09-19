"""Независимое повторное вычисление ретрива и сверка ответов с журналом API."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from eval import evaluate_one, experiment_config
from llm_client import digest, events, usage_cost
from rag_core import ROOT, read, sha
from schema import RAGAnswer


def verify_retrieval(output):
    out=Path(output)
    docs,gold,config=experiment_config(ROOT/'data',ROOT/'gold.json')
    meta=read(out/'retrieval_run.json')
    if meta['run_kind']!='local_retrieval' or meta['status']!='completed':
        raise ValueError('Нет завершённого измерения поиска')
    if meta['config']!=config:
        raise ValueError('Изменились код, корпус, разметка или настройки. Новый эксперимент — в новой папке --output.')
    saved=read(out/'evaluation.json')
    for strategy in ['A','B']:
        chunks,result=evaluate_one(docs,gold,strategy)
        assert read(out/f'chunks_{strategy}.json')==[asdict(c) for c in chunks], 'Изменён индекс '+strategy
        assert saved[strategy]==result, 'Результаты ретрива не совпали с пересчётом '+strategy
    return {'retrieval_verified':True,'documents':len(docs),'gold_questions':len(gold),
            'A_hit_rate_at_5':saved['A']['hit_rate_at_5'],'B_hit_rate_at_5':saved['B']['hit_rate_at_5']}


def verify(output, *, require_api=False, allow_test=False):
    out=Path(output)
    status=verify_retrieval(out)
    gen=out/'generation'
    if not (gen/'run.json').exists():
        if require_api:
            raise ValueError('Ответы DeepSeek ещё не получены; запусти pipeline.py generate')
        return {**status,'api_generation_verified':False,'note':'Ретрив проверен без API. Генерация ещё не выполнена.'}
    meta=read(gen/'run.json')
    if meta['status']!='completed':
        raise ValueError('API-прогон не завершён')
    if not allow_test and meta['config']['run_kind']!='llm_api':
        raise ValueError('Тестовую заглушку нельзя выдать за API-результат')
    config=read(out/'retrieval_run.json')['config']
    assert meta['config']['retrieval_config_sha256']==digest(config)
    assert meta['config']['code_sha256']==config['code_sha256']
    assert meta['config']['evaluation_sha256']==sha((out/'evaluation.json').read_bytes())
    log=events(gen/'api_trace.jsonl')
    received=[e for e in log if 'raw_response' in e]
    ids=[e['response_id'] for e in received]
    assert all(ids) and len(ids)==len(set(ids)), 'response_id не уникальны'
    valid={e['response_id']:e for e in log if e['status']=='valid'}
    cache={}
    for p in (gen/'cache').glob('*.json'):
        item=read(p);event=valid[item['response_id']]
        assert item['stage']==event['stage'] and p.stem==event['cache_key']
        value=RAGAnswer.model_validate_json(event['raw_response'])
        assert value.model_dump(mode='json')==item['value'],'Кэш отличается от ответа'
        cache[item['stage']]=item['value']
    from pipeline import check_citations
    evaluation=read(out/'evaluation.json')
    count,abstentions=0,0
    expected_stages=set()
    for strategy in ['A','B']:
        answers=read(out/f'answers_{strategy}.json')
        queries=evaluation[strategy]['questions']
        assert len(answers)==len(queries),'Ответы получены не на все вопросы'
        for item,query in zip(answers,queries):
            stage=f'{strategy}:{query["id"]}'
            expected_stages.add(stage)
            assert item['id']==query['id'] and item['question']==query['question']
            assert item['context_sha256']==digest(query['top5'])
            assert item['context_ids']==[c['chunk_id'] for c in query['top5']]
            assert item['answer']==cache[stage],'Итоговый ответ отличается от принятого'
            answer=RAGAnswer.model_validate(item['answer'])
            check_citations(answer,query['top5'])
            abstentions+=answer.status=='insufficient_context'
            count+=1
    assert set(cache)==expected_stages,'Лишние или пропущенные этапы генерации'
    metrics=read(out/'generation_metrics.json')
    assert metrics['answer_count']==count and metrics['abstentions']==abstentions
    assert metrics['api_responses']==len(received)
    for field in ['prompt_tokens','completion_tokens']:
        assert metrics[field]==sum((e.get('usage') or {}).get(field,0) or 0 for e in received)
    costs=[usage_cost(e.get('usage')) for e in received]
    if all(c is not None for c in costs):
        assert abs(metrics['observed_cost_interval_usd'][1]-sum(costs))<1e-10
    return {**status,'api_generation_verified':True,'run_kind':meta['config']['run_kind'],
            'answers':count,'abstentions':abstentions,
            'limits':'Проверена согласованность файлов; не биллинг провайдера и не смысловая точность всех ответов.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT/'output'))
    parser.add_argument('--require-api',action='store_true')
    args=parser.parse_args()
    try:
        print(json.dumps(verify(args.output,require_api=args.require_api),ensure_ascii=False,indent=2))
    except (AssertionError,ValueError,KeyError,FileNotFoundError) as exc:
        raise SystemExit('Проверка не пройдена: '+str(exc))
