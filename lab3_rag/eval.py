"""Реальное сравнение поиска A/B на gold; API и ключ не требуются."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from rag_core import (ROOT, CONFIG, BM25, build_chunks, corpus_stats, dump, fingerprint,
                      hit_rate, load_corpus, load_gold, read, sha)


def experiment_config(data_dir, gold_path):
    docs = load_corpus(data_dir)
    gold = load_gold(gold_path,docs)
    config = {'retrieval':CONFIG, 'corpus':corpus_stats(docs),
              'gold_sha256':sha(Path(gold_path).read_bytes()),
              'answer_evidence_sha256':sha((ROOT/'answer_evidence.json').read_bytes()),
              'code_sha256':fingerprint()}
    return docs,gold,config


def evaluate_one(docs, gold, strategy):
    chunks = build_chunks(docs,strategy)
    index = BM25(chunks)
    evidence = {r['id']:r['evidence'] for r in read(ROOT/'answer_evidence.json')}
    if set(evidence) != {r['id'] for r in gold}:
        raise ValueError('Для каждого gold-вопроса нужны диагностические опорные фразы')
    rows = []
    for question in gold:
        ranked = index.rank(question['question'])
        top = ranked[:5]
        score = hit_rate(top,question['gold_sources'])
        retrieved_sources = {r['source_id'] for r in top}
        missing = [s for s in question['gold_sources'] if s not in retrieved_sources]
        best_gold = []
        for source in question['gold_sources']:
            rank,item = next((i,r) for i,r in enumerate(ranked,1) if r['source_id'] == source)
            best_gold.append({'source_id':source,'rank':rank,**item})
        coverage=[]
        for item in evidence[question['id']]:
            if item['source_id'] not in question['gold_sources'] or item['quote'] not in docs[item['source_id']]:
                raise ValueError('Опорная фраза отсутствует в обязательном источнике')
            found=[r['chunk_id'] for r in top if r['source_id']==item['source_id'] and item['quote'] in r['text']]
            coverage.append({**item,'found_in':found})
        rows.append({**question,'score':score,'any_hit':score>0,'all_sources_hit':score==1,
                     'evidence_coverage':sum(bool(x['found_in']) for x in coverage)/len(coverage),
                     'evidence_checks':coverage,
                     'top5':top,'missing_sources':missing,'best_gold_chunks':best_gold})
    return chunks,{'strategy':strategy,'chunk_count':len(chunks),'questions':rows,
                   'hit_rate_at_5':sum(r['score'] for r in rows)/len(rows),
                   'any_hit_at_5':sum(r['any_hit'] for r in rows)/len(rows),
                   'all_sources_at_5':sum(r['all_sources_hit'] for r in rows)/len(rows),
                   'mean_evidence_coverage':sum(r['evidence_coverage'] for r in rows)/len(rows),
                   'mean_context_characters':sum(sum(len(t['text']) for t in r['top5']) for r in rows)/len(rows)}


def snippet(text, max_chars=230):
    return ' '.join(text.split())[:max_chars].replace('|','\\|')


def write_report(output):
    out = Path(output)
    result = read(out/'evaluation.json')
    meta = read(out/'retrieval_run.json')
    a,b = result['A'],result['B']
    stats = meta['config']['corpus']
    lines = ['# Лабораторная № 3: RAG и сравнение чанкинга','',
             '**Ретрив измерен локально на настоящих текстах; запросы к LLM для этих чисел не нужны.**','',
             '## 1. Корпус','',
             f'Авторская техническая документация нашего учебного проекта: {stats["documents"]} документов, '
             f'{stats["characters"]:,} символов; в каждом от 500 до 5000 слов. Это подготовленный снимок '
             'документации, а не реальные клиентские сообщения. Происхождение описано в CORPUS.md.','',
             '## 2. Gold','',
             'Первые 20 вопросов составлены до пилота. После одинакового результата 0,975 добавлены '
             '10 практических вопросов; исходные вопросы и источники не менялись. На 30 вопросах '
             'зафиксировано основное сравнение. Опорные фразы для дополнительной диагностики '
             'размечены после проверки документной метрики; это исследовательский анализ, не независимый тест. '
             'Ход работы отражён в PROTOCOL.md. Для прямых вопросов указан основной документ '
             'соответствующей темы; multi-hop требует разные факты из нескольких документов. '
             'Перекрытие сведений между разделами ограничивает точность разметки на уровне документов.','',
             '| ID | Тип | Сложный | Вопрос | Обязательные источники |',
             '|---|---|---|---|---|']
    for row in a['questions']:
        lines.append(f'| {row["id"]} | {row["type"]} | {"да" if row.get("hard") else "нет"} | '
                     f'{row["question"]} | {", ".join(row["gold_sources"])} |')
    lines += ['', '## 3. Результаты','',
              'В обеих конфигурациях один BM25, одинаковые токенизация, параметры k1=1,5 и b=0,75 '
              'и TOP-5 **чанков**. Дедупликация по документу не выполняется. Векторный поиск не использован: '
              'это явно обозначенный лексический RAG, которому не нужен дополнительный API эмбеддингов.','',
              '**Метрика преподавателя:** для вопроса — доля его gold_sources, найденных в TOP-5; '
              'итог — среднее этих долей. Для multi-hop попадание одного из двух обязательных документов '
              'даёт 0,5. В терминологии retrieval это macro source-recall@5; здесь сохранено название hit-rate@5 из задания.','',
              '| Стратегия | Чанков | hit-rate@5 | Все gold-источники@5 | Хотя бы один@5 | Средний контекст, символов |',
              '|---|---:|---:|---:|---:|---:|']
    for r in (a,b):
        label = 'A: 2000, без перекрытия' if r['strategy']=='A' else 'B: границы абзацев/фраз, 800 / 120'
        lines.append(f'| {label} | {r["chunk_count"]} | {r["hit_rate_at_5"]:.4f} | '
                     f'{r["all_sources_at_5"]:.4f} | {r["any_hit_at_5"]:.4f} | {r["mean_context_characters"]:.0f} |')
    lines += ['', 'B выбирает ближайшую подходящую границу абзаца, строки, предложения или слова '
              'во второй половине окна и сохраняет перекрытие 120 символов. Это уменьшает число разрывов '
              'внутри формулировок. Начало очередного окна после перекрытия может оказаться внутри слова; '
              'исходные смещения и весь текст сохраняются. A в точности реализует text[i:i+2000].','',
              'При одинаковом TOP-5 объём переданного контекста различается. Поэтому измеряется '
              'эффект всей конфигурации чанкинга, включая нормализацию длины BM25 и число чанков; '
              'это не изолированное доказательство преимущества сохранения границ предложений.','',
              f'Дополнительное покрытие опорных фраз: A={a["mean_evidence_coverage"]:.4f}, '
              f'B={b["mean_evidence_coverage"]:.4f}. Фраза засчитывается, если целиком присутствует '
              'в одном из TOP-5 чанков правильного источника. Это строгая диагностическая проверка '
              'записанного фрагмента, а не измерение истинности ответа LLM.','',
              '## 4. Анализ ошибок','']
    pairs = [(x,y) for x,y in zip(a['questions'],b['questions']) if x['score'] != y['score']]
    pairs.sort(key=lambda pair:(-abs(pair[0]['score']-pair[1]['score']),pair[0]['id']))
    document_difference_count=len(pairs)
    if len(pairs)<2:
        lines += [f'Различий по hit-rate на уровне документов: {len(pairs)}. '
                  'Ниже отдельно показаны различия наличия опорных фактов внутри найденных документов. '
                  'Они не выдаются за различия основной метрики: документ может попасть в TOP-5 '
                  'через фрагмент, в котором нет нужной части ответа.','']
        pairs=[(x,y) for x,y in zip(a['questions'],b['questions']) if x['evidence_coverage']!=y['evidence_coverage']]
        pairs.sort(key=lambda p:(-int(max(p[0]['evidence_coverage'],p[1]['evidence_coverage'])==1),p[0]['id']))
    for x,y in pairs[:3]:
        lines += [f'### Вопрос {x["id"]}: {x["question"]}','',
                  f'A: {x["score"]:.2f}; B: {y["score"]:.2f}. Gold: {", ".join(x["gold_sources"])}.','']
        for label,row in [('A',x),('B',y)]:
            lines += [f'**{label}:** TOP-5 = ' + ', '.join(f'`{t["chunk_id"]}`' for t in row['top5']) + '.',
                      'Пропущены: '+(', '.join(row['missing_sources']) or 'нет')+'.']
            for hit in row['best_gold_chunks']:
                lines.append(f'- Лучший чанк обязательного источника `{hit["source_id"]}`: '
                             f'`{hit["chunk_id"]}`, место {hit["rank"]}, BM25={hit["score"]:.4f}, '
                             f'смещения [{hit["start"]}, {hit["end"]}). Фрагмент: «{snippet(hit["text"])}…»')
            for item in row['evidence_checks']:
                found=', '.join(item['found_in']) or 'в TOP-5 нет целого фрагмента'
                lines.append(f'- Опорная фраза `{item["source_id"]}`: «{item["quote"]}» → {found}.')
            leaders = [t for t in row['top5'] if t['source_id'] not in row['gold_sources']]
            if leaders:
                t=leaders[0]
                lines.append(f'- Конкурирующий чанк: `{t["chunk_id"]}`, BM25={t["score"]:.4f}; '
                             f'«{snippet(t["text"])}…»')
            lines.append('')
        lines += ['Один и тот же вопрос ранжирует разный набор фрагментов. На коротких блоках '
                  'совпадающие слова могут получить больший вес, но соседний факт или второй источник '
                  'может оказаться за пределами пятёрки. Приведённые позиции и опорные фразы показывают '
                  'ограничения выбранного контекста; сами по себе они не оценивают правильность будущего ответа LLM.','']
    if not pairs:
        lines += ['Для дальнейшего анализа следует посмотреть вопросы с неполным покрытием у обеих '
                  'стратегий, заранее дополнить эталон новыми сценариями и провести отдельный эксперимент.','']
    delta=b['hit_rate_at_5']-a['hit_rate_at_5']
    winner='B' if delta>0 else 'A' if delta<0 else 'ничья'
    lines += ['## 5. Вывод','',
              f'На зафиксированном gold результат: **{winner}**; разность B−A = {delta:+.4f}. '
              'Это наблюдение для данного корпуса и BM25, а не универсальное ранжирование стратегий. '
              'Синонимы без общих токенов остаются слабым местом лексического поиска; следующий эксперимент '
              'может добавить мультиязычные эмбеддинги, сохранив обе схемы разбиения.','',
              '## Статус генерации ответов','']
    gen=out/'generation'/'run.json'
    if gen.exists() and read(gen)['status']=='completed':
        gm=read(out/'generation_metrics.json')
        lines += [f'Генерация: {gm["run_kind"]}, {gm["answer_count"]} ответов, '
                  f'{gm["api_responses"]} полученных ответов API с учётом повторов. '
                  'Числа retrieval не пересчитывались по результатам генерации.']
    else:
        lines += ['**Реальный DeepSeek-прогон ещё не выполнен.** RUN_LAB3.cmd последовательно '
                  'сгенерирует ответы на все gold-вопросы для A и B, проверит их цитаты и соберёт архив. '
                  'Отказ модели при недостаточном контексте сохраняется как результат.']
    lines += ['', 'Оценка retrieval выполнена по документам; попадание документа не гарантирует наличие '
              'нужного факта в выбранном чанке. Проверка цитат не подтверждает семантическую верность '
              'всего ответа. Ручная разметка, независимая проверка ответов и несколько корпусов '
              'нужны для выводов о рабочем качестве системы.', '']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    dump(out/'submission_status.json',{'retrieval_run_completed':True,
         'different_document_hit_questions':document_difference_count,
         'different_evidence_questions':sum(x['evidence_coverage']!=y['evidence_coverage'] for x,y in zip(a['questions'],b['questions'])),
         'at_least_two_document_hit_difference_examples':document_difference_count>=2,
         'at_least_two_context_difference_examples':len(pairs)>=2,
         'api_generation_completed':gen.exists() and read(gen).get('status')=='completed'})


def run_eval(data_dir=None, gold_path=None, output=None):
    out=Path(output or ROOT/'output')
    docs,gold,config=experiment_config(data_dir or ROOT/'data',gold_path or ROOT/'gold.json')
    meta_path=out/'retrieval_run.json'
    if meta_path.exists() and read(meta_path)['config']!=config:
        raise ValueError('Изменились код, корпус или gold. Выбери новую папку --output для нового эксперимента.')
    started=time.perf_counter()
    results={}
    for strategy in ['A','B']:
        chunks,result=evaluate_one(docs,gold,strategy)
        dump(out/f'chunks_{strategy}.json',[asdict(c) for c in chunks])
        results[strategy]=result
        print(f'{strategy}: chunks={len(chunks)}, hit-rate@5={result["hit_rate_at_5"]:.4f}',flush=True)
    dump(out/'evaluation.json',results)
    dump(meta_path,{'run_kind':'local_retrieval','status':'completed','config':config,
                    'finished_at':datetime.now(timezone.utc).isoformat(),
                    'seconds':time.perf_counter()-started,'llm_api_used':False})
    write_report(out)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT/'output'))
    parser.add_argument('--data',default=str(ROOT/'data'))
    parser.add_argument('--gold',default=str(ROOT/'gold.json'))
    args=parser.parse_args()
    run_eval(args.data,args.gold,args.output)
