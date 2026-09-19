"""Лексический RAG: поиск по чанкам → ответ DeepSeek с проверяемыми цитатами."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval import experiment_config, run_eval, write_report
from llm_client import Run, GroundingError, make_client, events, digest
from prompts import SYSTEM
from rag_core import ROOT, BM25, build_chunks, dump, read, sha
from schema import RAGAnswer


def normalize(text):
    return ' '.join(text.split())


def check_citations(answer, context):
    chunks={c['chunk_id']:c['text'] for c in context}
    checks=[{'chunk_id':c.chunk_id,'quote':c.quote,
             'valid':c.chunk_id in chunks and normalize(c.quote) in normalize(chunks[c.chunk_id])}
            for c in answer.citations]
    if any(not c['valid'] for c in checks):
        raise GroundingError('Цитаты должны дословно присутствовать в указанном чанке текущего контекста: '+
                             json.dumps([c for c in checks if not c['valid']],ensure_ascii=False),checks)
    return checks


def answer_question(run, stage, question, context):
    packet={'question':question,'context':[{'chunk_id':c['chunk_id'],'text':c['text']} for c in context]}
    return make_client(run).chat.completions.create(
        stage=stage,model=run.model,response_model=RAGAnswer,max_retries=3,
        temperature=0,max_tokens=2500,
        messages=[{'role':'system','content':SYSTEM},
                  {'role':'user','content':json.dumps(packet,ensure_ascii=False)}],
        validate=lambda value:check_citations(value,context))


def generate(output=None, *, transport=None):
    out=Path(output or ROOT/'output')
    from verify_results import verify_retrieval
    verify_retrieval(out)
    evaluation=read(out/'evaluation.json')
    config=read(out/'retrieval_run.json')['config']
    run=Run(out/'generation',{'retrieval_config_sha256':digest(config),
            'evaluation_sha256':sha((out/'evaluation.json').read_bytes()),
            'mode':'all_gold_A_B'},transport=transport)
    total=0
    try:
        for strategy in ['A','B']:
            answers=[]
            for row in evaluation[strategy]['questions']:
                print(f'{strategy}: вопрос {row["id"]} / {len(evaluation[strategy]["questions"])}',flush=True)
                context=row['top5']
                answer=answer_question(run,f'{strategy}:{row["id"]}',row['question'],context)
                answers.append({'id':row['id'],'question':row['question'],
                                'context_ids':[c['chunk_id'] for c in context],
                                'context_sha256':digest(context),'answer':answer.model_dump(mode='json')})
                dump(out/f'answers_{strategy}.json',answers)
                total+=1
        log=events(run.trace)
        responses=[e for e in log if 'raw_response' in e]
        costs=[e['cost_upper_usd'] for e in responses]
        quoted=[q for e in responses for q in e.get('quote_checks',[])]
        all_answers=[r for s in ['A','B'] for r in read(out/f'answers_{s}.json')]
        metrics={'run_kind':run.config['run_kind'],'answer_count':len(all_answers),
                 'api_responses':len(responses),
                 'validation_error_responses':sum(e['status']=='validation_error' for e in log),
                 'quote_checks':len(quoted),'ghost_quote_checks':sum(not q['valid'] for q in quoted),
                 'abstentions':sum(r['answer']['status']=='insufficient_context' for r in all_answers),
                 'prompt_tokens':sum((e.get('usage') or {}).get('prompt_tokens',0) or 0 for e in responses),
                 'completion_tokens':sum((e.get('usage') or {}).get('completion_tokens',0) or 0 for e in responses),
                 'observed_cost_interval_usd':[sum(costs)*.5,sum(costs)] if all(c is not None for c in costs) else None,
                 'cost_note':'Оценка off-peak…peak по usage; не банковское списание, без налогов и ответов с неизвестной стоимостью.'}
        dump(out/'generation_metrics.json',metrics)
        run.finish('completed',answer_count=total)
        write_report(out)
        print('Генерация завершена; теперь проверка и сбор архива.',flush=True)
        return metrics
    except BaseException:
        run.finish('interrupted',answer_count=total)
        raise
    finally:
        run.close()


def ask(query, strategy='B', output=None):
    docs,_,config=experiment_config(ROOT/'data',ROOT/'gold.json')
    context=BM25(build_chunks(docs,strategy)).retrieve(query)
    folder=Path(output or ROOT/'output')/'interactive'/digest([query,strategy,config])[:16]
    run=Run(folder,{'mode':'interactive','query':query,'strategy':strategy,'retrieval':config})
    try:
        value=answer_question(run,'interactive',query,context)
        dump(folder/'context.json',context)
        dump(folder/'answer.json',value.model_dump())
        run.finish('completed')
        print(value.model_dump_json(indent=2))
        return value
    except BaseException:
        run.finish('interrupted')
        raise
    finally:
        run.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=['eval','generate','all','ask'])
    parser.add_argument('question',nargs='?')
    parser.add_argument('--strategy',choices=['A','B'],default='B')
    parser.add_argument('--output',default=str(ROOT/'output'))
    args=parser.parse_args()
    try:
        if args.task in {'eval','all'}:
            run_eval(output=args.output)
        if args.task in {'generate','all'}:
            generate(args.output)
        if args.task=='ask':
            if not args.question:
                parser.error('После ask укажи вопрос в кавычках')
            ask(args.question,args.strategy,args.output)
    except (ValueError,RuntimeError,AssertionError,FileNotFoundError) as exc:
        print(str(exc),file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
