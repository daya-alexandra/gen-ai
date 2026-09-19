"""Проверка сохранённого API-прогона без новых запросов и без изменения результатов."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from llm_client import ROOT, read, events, code_hashes, usage_cost
from pipeline import load_input, validate_result, check_quotes
from schema import Reviews, Aspects, MapSummary, Summary, JudgeReport, Discovery


def verify(output, input_path, *, allow_test=False):
    out = Path(output)
    meta = read(out / 'run.json')
    if not allow_test and meta['config']['run_kind'] != 'llm_api':
        raise ValueError('Это тестовый прогон; его нельзя сдавать как результат API')
    if meta['status'] not in {'completed', 'needs_review'}:
        raise ValueError('Прогон не завершён')
    sources = load_input(input_path)
    source_data = [r.model_dump() for r in sources]
    expected_hash = hashlib.sha256(json.dumps(source_data, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert meta['config']['input_sha256'] == expected_hash, 'Изменились исходные отзывы'
    assert read(out / 'input_snapshot.json') == source_data, 'Снимок входа отличается'
    assert meta['config']['code_sha256'] == code_hashes(), 'Изменился код после запуска'
    ie = Reviews.model_validate(read(out / 'reviews.json'))
    ap = Aspects.model_validate(read(out / 'aspects.json'))
    summary = Summary.model_validate(read(out / 'summary.json'))
    judge = JudgeReport.model_validate(read(out / 'judge_report.json'))
    discovery = Discovery.model_validate(read(out / 'discovered_aspects.json'))
    for value in (ie, ap, summary, discovery):
        validate_result(value, sources)
    validate_result(judge, sources, expected_actions=[a.action_id for a in summary.action_items], summary=summary)
    chunks = read(out / 'chunks.json')
    ids = [s.review_id for s in sources]
    assert [rid for chunk in chunks for rid in chunk] == ids, 'Разбиение потеряло или продублировало отзывы'
    maps = read(out / 'map_summaries.json')
    assert len(maps) == len(chunks), 'Не все MAP-фрагменты обработаны'
    for chunk, item in zip(chunks, maps):
        validate_result(MapSummary.model_validate(item), [r for r in sources if r.review_id in chunk])
    validate_result(summary, sources, allowed_evidence=maps)
    log = events(out / 'api_trace.jsonl')
    good = {e['response_id']: e for e in log if e['status'] == 'valid'}
    received = [e for e in log if 'raw_response' in e]
    response_ids = [e['response_id'] for e in received]
    assert all(response_ids) and len(response_ids) == len(set(response_ids)), 'Повторяющиеся/пустые response_id'
    cached = {}
    for file in (out / 'cache').glob('*.json'):
        item = read(file)
        event = good[item['response_id']]
        assert file.stem == event['cache_key'] and item['stage'] == event['stage'], 'Несоответствие cache и trace'
        stage = item['stage']
        response_model = next(model for prefix, model in [
            ('ie_', Reviews), ('aspects_', Aspects), ('map_', MapSummary), ('reduce_', Summary),
            ('judge_', JudgeReport), ('diagnostic_judge', JudgeReport), ('discovery', Discovery)] if stage.startswith(prefix))
        accepted = response_model.model_validate_json(event['raw_response']).model_dump(mode='json')
        assert item['value'] == accepted, 'Кэш отличается от принятого ответа модели'
        cached[item['stage']] = item['value']
    joined_ie = sorted([r for s, v in cached.items() if s.startswith('ie_') for r in v['reviews']], key=lambda r: r['review_id'])
    joined_ap = sorted([r for s, v in cached.items() if s.startswith('aspects_') for r in v['reviews']], key=lambda r: r['review_id'])
    assert ie.model_dump(mode='json')['reviews'] == joined_ie, 'IE отличается от ответов'
    assert ap.model_dump(mode='json')['reviews'] == joined_ap, 'Аспекты отличаются от ответов'
    assert maps == [cached[f'map_{i:02d}'] for i in range(len(chunks))], 'MAP отличается от ответов'
    version = 2 if (out / 'judge_v2.json').exists() else 1
    assert read(out / 'summary_v1.json') == cached['reduce_v1']
    assert read(out / 'judge_v1.json') == cached['judge_v1']
    first_judge = JudgeReport.model_validate(cached['judge_v1'])
    assert (version == 2) == (first_judge.overall_score < .7), 'Нарушено правило повторного REDUCE'
    assert summary.model_dump() == cached[f'reduce_v{version}']
    assert judge.model_dump() == cached[f'judge_v{version}']
    assert discovery.model_dump() == cached['discovery']
    if meta['config']['diagnostic']:
        assert read(out / 'diagnostic_judge.json') == cached['diagnostic_judge']
        assert read(out / 'diagnostic_expected.json')['kind'] == 'controlled_fault_injection'
    metrics = read(out / 'metrics.json')
    checks = [q for e in log if not e['stage'].startswith('diagnostic') for q in e.get('quote_checks', [])]
    ghost = sum(not q['valid'] for q in checks)
    assert metrics['ghost_quote_checks'] == ghost
    assert metrics['quotes_checked_across_attempts'] == len(checks)
    assert metrics['judge_overall_score'] == judge.overall_score
    assert metrics['valid_reviews'] == len(ie.reviews) == len(sources)
    assert metrics['api_responses_including_diagnostics'] == len(received)
    for field in ('prompt_tokens', 'completion_tokens'):
        assert metrics[field] == sum((e.get('usage') or {}).get(field, 0) or 0 for e in received)
    costs = [usage_cost(e.get('usage')) for e in received]
    if all(c is not None for c in costs):
        assert abs(metrics['observed_usage_cost_interval_usd'][1] - sum(costs)) < 1e-10
    assert all(q['valid'] for q in check_quotes(summary, sources))
    assert (out / 'heatmap.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert all(x in (out / 'выводы.md').read_text(encoding='utf-8') for x in ['## 1.', '## 2.', '## 3.'])
    return {'verified': True, 'run_kind': meta['config']['run_kind'], 'source_count': len(sources),
            'response_count': len(received), 'judge_score': judge.overall_score,
            'numeric_criteria_met': metrics['numeric_criteria_met'],
            'limits': 'Проверяется согласованность локальных файлов, а не биллинг и серверные журналы провайдера.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT / 'output'))
    parser.add_argument('--input', default=str(ROOT / 'input'))
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.output, args.input), ensure_ascii=False, indent=2))
    except (AssertionError, ValueError, KeyError, FileNotFoundError) as exc:
        raise SystemExit('Проверка не пройдена: ' + str(exc))
