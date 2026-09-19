"""IE → аспекты → параллельный MAP → REDUCE → judge → autodiscovery."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import prompts
from llm_client import ROOT, Run, GroundingError, dump, read, events, make_client
from schema import (ASPECTS, SourceReview, Reviews, Aspects, MapSummary, Summary,
                    JudgeReport, Discovery)


def load_input(path):
    path = Path(path)
    files = sorted(path.glob('*.jsonl')) if path.is_dir() else [path]
    rows = [SourceReview.model_validate_json(line) for f in files
            for line in f.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not 30 <= len(rows) <= 100:
        raise ValueError('Требуется от 30 до 100 отзывов')
    ids = [x.review_id for x in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('ID исходных отзывов должны быть уникальны')
    return rows


def norm(text):
    # Только эквивалентные пробелы: отрицания, числа, регистр не меняются.
    return ' '.join(text.split())


def walk_evidence(obj):
    if hasattr(obj, 'model_dump'):
        obj = obj.model_dump(mode='json')
    if isinstance(obj, dict):
        if 'quote' in obj and 'review_id' in obj:
            yield obj
        for value in obj.values():
            yield from walk_evidence(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_evidence(value)


def check_quotes(obj, sources):
    lookup = {r.review_id: r.text for r in sources}
    return [{'review_id': e['review_id'], 'quote': e['quote'],
             'valid': e['review_id'] in lookup and norm(e['quote']) in norm(lookup[e['review_id']])}
            for e in walk_evidence(obj)]


def validate_result(value, sources, *, expected_actions=None, summary=None, allowed_evidence=None):
    checks = check_quotes(value, sources)
    if any(not x['valid'] for x in checks):
        bad = [x for x in checks if not x['valid']]
        raise GroundingError('Цитата отсутствует в указанном отзыве: ' + json.dumps(bad, ensure_ascii=False), checks)
    ids = {x.review_id for x in sources}
    if isinstance(value, (Reviews, Aspects)):
        got = [x.review_id for x in value.reviews]
        if len(got) != len(ids) or set(got) != ids:
            raise GroundingError('Нужен каждый входной review_id ровно один раз', checks)
        for row in value.reviews:
            if any(e['review_id'] != row.review_id for e in walk_evidence(row)):
                raise GroundingError('Нельзя переносить цитату одного автора в строку другого', checks)
    if isinstance(value, MapSummary):
        if len(value.source_ids) != len(ids) or set(value.source_ids) != ids:
            raise GroundingError('MAP должен перечислить все ID фрагмента ровно по одному разу', checks)
    if allowed_evidence is not None:
        allowed = {(e['review_id'], norm(e['quote'])) for e in walk_evidence(allowed_evidence)}
        if any((e['review_id'], norm(e['quote'])) not in allowed for e in walk_evidence(value)):
            raise GroundingError('REDUCE может использовать только цитаты из MAP', checks)
    if isinstance(value, JudgeReport):
        got = [v.action_id for v in value.verdicts]
        if expected_actions is not None and (len(got) != len(expected_actions) or set(got) != set(expected_actions)):
            raise GroundingError('Судья должен оценить каждую рекомендацию ровно один раз', checks)
        texts = [x.text for x in summary.findings + summary.action_items] if summary else []
        for d in value.distortions:
            if not any(norm(d.mistaken_text) in norm(t) for t in texts):
                raise GroundingError('mistaken_text должен дословно присутствовать в проверяемой сводке', checks)
    return checks


def split_chunks(rows, max_chars=3800):
    """Границы по отзывам: ни один отзыв не теряется и не дублируется."""
    chunks, current, size = [], [], 0
    for row in rows:
        if current and size + len(row.text) > max_chars:
            chunks.append(current)
            current, size = [], 0
        current.append(row)
        size += len(row.text)
    if current:
        chunks.append(current)
    return chunks


def ask(run, stage, model, system, payload, sources, **constraints):
    return make_client(run).chat.completions.create(
        model=run.model, stage=stage, response_model=model, max_retries=3,
        temperature=0 if model is JudgeReport else .2, max_tokens=5000,
        messages=[{'role': 'system', 'content': system},
                  {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
        validate=lambda value: validate_result(value, sources, **constraints))


def build_heatmap(aspects, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    data = np.array([[next(a.score if a.score is not None else np.nan for a in row.aspects if a.aspect == name)
                      for name in ASPECTS] for row in aspects.reviews], dtype=float)
    cmap = plt.get_cmap('RdYlGn').copy()
    cmap.set_bad('#e5e7eb')
    fig, ax = plt.subplots(figsize=(10, max(8, len(data) * .28)), layout='constrained')
    im = ax.imshow(data, cmap=cmap, vmin=-2, vmax=2, aspect='auto')
    ax.set_xticks(range(6), ASPECTS, rotation=25, ha='right')
    ax.set_yticks(range(len(data)), [r.review_id for r in aspects.reviews])
    ax.set_title('Оценки аспектов: −2 … +2\nСерый цвет / «—»: аспект не упоминается', pad=14)
    for i in range(len(data)):
        for j in range(6):
            ax.text(j, i, '—' if np.isnan(data[i, j]) else str(int(data[i, j])), ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, ticks=[-2, -1, 0, 1, 2], shrink=.5)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def diagnostics(run, sources, summary):
    """Контролируемые ошибки; не выдаются за ошибки основного прогона."""
    altered = copy.deepcopy(summary)
    altered.findings[0].text = 'Все пользователи без исключения сообщили о потере всех задач каждый день.'
    altered.findings[1].text = 'Пользователям гарантирован возврат денег за подписку в течение одной минуты.'
    altered.findings[2].text = 'Покупка подписки доказанно устраняет любые сбои на всех моделях телефонов.'
    altered.action_items[-1].action_id = 'A99'
    altered.action_items[-1].text = 'Добавить продажу авиабилетов внутри планировщика задач.'
    dump(run.output / 'diagnostic_summary.json', altered.model_dump(mode='json'))
    result = ask(run, 'diagnostic_judge', JudgeReport, prompts.JUDGE,
                 {'sources': [x.model_dump() for x in sources], 'summary': altered.model_dump()},
                 sources, expected_actions=[x.action_id for x in altered.action_items], summary=altered)
    dump(run.output / 'diagnostic_judge.json', result.model_dump(mode='json'))
    dump(run.output / 'diagnostic_expected.json', {
        'kind': 'controlled_fault_injection', 'not_main_run_errors': True,
        'inserted_distortions': [x.text for x in altered.findings[:3]],
        'unsupported_action_id': 'A99',
        'judge_detected_action': next(x.support for x in result.verdicts if x.action_id == 'A99')})


def build_metrics(run, sources, summary, judge):
    log = events(run.trace)
    primary = [e for e in log if not e['stage'].startswith('diagnostic')]
    quoted = [q for e in primary for q in e.get('quote_checks', [])]
    ghost = sum(not q['valid'] for q in quoted)
    final_quotes = check_quotes(summary, sources)
    responses = [e for e in log if 'raw_response' in e]
    priced = [e for e in responses if e['cost_upper_usd'] is not None]
    cost = sum(e['cost_upper_usd'] for e in priced)
    ratio = ghost / len(quoted) if quoted else 1
    return {'run_kind': run.config['run_kind'], 'source_count': len(sources),
            'valid_reviews': len(read(run.output / 'reviews.json')['reviews']),
            'validation_error_responses': sum(e['status'] == 'validation_error' for e in primary),
            'pydantic_error_responses': sum(e.get('error_type') == 'ValidationError' for e in primary),
            'quotes_checked_across_attempts': len(quoted), 'ghost_quote_checks': ghost,
            'ghost_quote_ratio': ratio, 'final_ghost_quotes': sum(not q['valid'] for q in final_quotes),
            'judge_overall_score': judge.overall_score, 'api_responses_including_diagnostics': len(responses),
            'prompt_tokens': sum((e.get('usage') or {}).get('prompt_tokens', 0) or 0 for e in responses),
            'completion_tokens': sum((e.get('usage') or {}).get('completion_tokens', 0) or 0 for e in responses),
            'cache_hit_tokens': sum((e.get('usage') or {}).get('prompt_cache_hit_tokens', 0) or 0 for e in responses),
            'responses_without_usage': len(responses) - len(priced),
            'observed_usage_cost_interval_usd': [cost * .5, cost] if len(priced) == len(responses) else None,
            'cost_note': 'Оценка по usage: off-peak…peak, без налогов. Это не выписка провайдера. Диагностика включена.',
            'active_seconds': run.meta['active_seconds'],
            'numeric_criteria_met': ratio <= .10 and judge.overall_score >= .7 and all(q['valid'] for q in final_quotes)}


def write_report(run, sources, summary, judge, discovery, metrics):
    out = run.output
    kind = 'РЕАЛЬНЫЙ API-ПРОГОН' if run.config['run_kind'] == 'llm_api' else 'ТЕСТОВЫЙ ПРОГОН — НЕ РЕЗУЛЬТАТ API'
    interval = metrics['observed_usage_cost_interval_usd']
    price = f'${interval[0]:.5f}–${interval[1]:.5f}' if interval is not None else 'не определена: отсутствует часть usage'
    lines = ['# Выводы по лабораторной № 2', '', kind, '', '## 1. Что получилось', '',
             f'Обработано {len(sources)} синтетических отзывов о вымышленном приложении «ПланДень». '
             'Корпус подготовлен с помощью ИИ; он предназначен для проверки метода и не репрезентативен. '
             f'В результате IE сохранено {metrics["valid_reviews"]} валидных объектов. '
             f'На отдельных попытках ответов Pydantic отклонил {metrics["pydantic_error_responses"]}; '
             f'всего ошибок JSON, схемы или привязки к источнику: {metrics["validation_error_responses"]}. '
             'Ошибочная попытка ответа и окончательно потерянный отзыв — разные единицы; при завершении все входные ID сохранены.', '',
             f'Проверок цитат во всех попытках основного конвейера: {metrics["quotes_checked_across_attempts"]}; '
             f'ghost-проверок: {metrics["ghost_quote_checks"]} ({metrics["ghost_quote_ratio"]:.1%}). '
             f'В финальной сводке отсутствующих цитат: {metrics["final_ghost_quotes"]}. '
             'Проверка требует дословного фрагмента именно указанного отзыва; нормализуются только пробелы. '
             'Она не доказывает правильность интерпретации — это отдельно оценивает judge.', '',
             f'Итоговый overall_score судьи: {judge.overall_score:.3f}. '
             f'Активное время с учётом возобновлений: {metrics["active_seconds"]:.2f} с. '
             f'Получено {metrics["api_responses_including_diagnostics"]} ответов, '
             f'{metrics["prompt_tokens"]} входных и {metrics["completion_tokens"]} выходных токенов. '
             f'Расчётная стоимость по usage, включая диагностический запрос: {price}. '
             'Тарифный диапазон off-peak…peak взят из официальной документации DeepSeek на 19.09.2026; '
             'налоги и запросы без полученного ответа в эту сумму не входят.', '',
             'MAP выполняется параллельно по целым отзывам; REDUCE объединяет только полученные карты. '
             'В heatmap серый цвет обозначает отсутствие упоминания, а не нейтральное отношение.', '',
             '## 2. Где модель ошибалась', '']
    found = []
    for p in sorted(out.glob('judge_v*.json')):
        report = JudgeReport.model_validate(read(p))
        for d in report.distortions:
            item = f'[{p.name}, {d.original.review_id}] «{d.original.quote}» → «{d.mistaken_text}». Исправление: {d.correction}'
            if item not in found:
                found.append(item)
    for e in events(run.trace):
        if e['stage'].startswith('diagnostic'):
            continue
        for q in e.get('quote_checks', []):
            if not q['valid']:
                found.append(f'[{e["stage"]}, {q["review_id"]}] Модель выдала «{q["quote"]}»; такого фрагмента в указанном источнике нет. Ответ отклонён.')
    lines.extend(['- ' + x for x in found[:3]] or ['В основном прогоне проверками не зарегистрированы конкретные искажения. Ошибки не добавлялись искусственно в основной результат.'])
    weak = [(p.name, v) for p in sorted(out.glob('judge_v*.json'))
            for v in JudgeReport.model_validate(read(p)).verdicts if v.support != 'supported']
    lines += ['', 'Пример работы судьи в основном прогоне:']
    lines += [f'- [{name}] {v.action_id}: {v.support}. {v.explanation}' for name, v in weak[:1]] or ['Все рекомендации оценены как supported; пример отрицательного вердикта ниже относится к контролируемой проверке.']
    if (out / 'diagnostic_judge.json').exists():
        control = JudgeReport.model_validate(read(out / 'diagnostic_judge.json'))
        v = next(x for x in control.verdicts if x.action_id == 'A99')
        lines += ['', '**Отдельная контролируемая проверка.** В копию сводки специально внесены три '
                  'ложных утверждения и рекомендация продавать авиабилеты. Это ошибки тестовой заготовки, '
                  'а не естественные ошибки генератора. Сводка и вердикт сохранены в diagnostic_*.json.',
                  f'Судья оценил A99 как **{v.support}**: {v.explanation}']
        lines += [f'- [{d.original.review_id}] «{d.original.quote}» → «{d.mistaken_text}». {d.correction}' for d in control.distortions[:3]]
    new = [a for a in discovery.aspects if a.related_fixed_aspect is None]
    lines += ['', 'Autodiscovery сопоставлен с шестью фиксированными аспектами:']
    lines += [f'- {a.name}: {a.description} — ' + (f'соответствует «{a.related_fixed_aspect}».' if a.related_fixed_aspect else 'кандидат в дополнительную тему; требуется ручная проверка разграничения.') for a in discovery.aspects]
    lines += ['', f'Кандидатов вне Literal: {len(new)}. Синонимы фиксированных аспектов не считаются новыми темами.', '',
              '## 3. Что бы изменили в production', '',
              'Сохранила бы строгую схему, привязку цитат к ID, явную отметку отсутствующих аспектов, '
              'журнал запросов и возобновление по сохранённым ответам. Переработала бы рубрику аспектов '
              'на реальных размеченных отзывах: например, синхронизация может относиться и к надёжности, '
              'и к отдельному сценарию использования. Разбиение по целым отзывам подходит короткому корпусу, '
              'но длинные отзывы потребуют дополнительной сегментации с сохранением смещений.', '',
              'Добавила бы независимую ручную разметку, precision/recall извлечения и согласованность '
              'оценок, A/B двух моделей и нескольких запусков, тесты на инструкции внутри отзывов, '
              'защиту персональных данных и выборочную ручную проверку рекомендаций. Генератор и судья '
              'здесь используют одну модель в отдельных запросах, поэтому их согласие не является '
              'независимым доказательством качества. Синтетический корпус не позволяет делать выводы '
              'о настоящем приложении или клиентской базе.', '']
    (out / 'выводы.md').write_text('\n'.join(lines), encoding='utf-8')
    dump(out / 'submission_status.json', {
        'real_api_run': run.config['run_kind'] == 'llm_api', 'numeric_criteria_met': metrics['numeric_criteria_met'],
        'natural_error_examples_found': len(found), 'natural_weak_verdict_found': bool(weak),
        'review_needed': 'Прочитать выводы и сверить трактовку аспектов. Контролируемые ошибки не подменяют примеры естественных ошибок.',
        'two_natural_error_examples_available': len(found) >= 2})


def analyze(input_path, output=None, *, transport=None, diagnostic=True):
    sources = load_input(input_path)
    payload = [r.model_dump() for r in sources]
    run = Run(output or ROOT / 'output', {
        'input_sha256': hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        'source_ids': [r.review_id for r in sources], 'batch_size': 5, 'map_max_chars': 3800,
        'map_workers': 3, 'diagnostic': diagnostic}, transport=transport)
    try:
        dump(run.output / 'input_snapshot.json', payload)
        reviews, aspects = [], []
        for i in range(0, len(sources), 5):
            part = sources[i:i + 5]
            data = [r.model_dump() for r in part]
            print(f'IE и аспекты: {i + 1}–{i + len(part)} / {len(sources)}', flush=True)
            ie = ask(run, f'ie_{i // 5:02d}', Reviews, prompts.IE, data, part)
            ap = ask(run, f'aspects_{i // 5:02d}', Aspects, prompts.ASPECT, data, part)
            reviews.extend(ie.reviews)
            aspects.extend(ap.reviews)
        reviews = Reviews(reviews=sorted(reviews, key=lambda r: r.review_id))
        aspects = Aspects(reviews=sorted(aspects, key=lambda r: r.review_id))
        dump(run.output / 'reviews.json', reviews.model_dump(mode='json'))
        dump(run.output / 'aspects.json', aspects.model_dump(mode='json'))
        build_heatmap(aspects, run.output / 'heatmap.png')
        chunks = split_chunks(sources)
        dump(run.output / 'chunks.json', [[r.review_id for r in p] for p in chunks])
        print(f'MAP: {len(chunks)} фрагментов, до 3 запросов параллельно', flush=True)
        def map_one(item):
            i, part = item
            return ask(run, f'map_{i:02d}', MapSummary, prompts.MAP, [r.model_dump() for r in part], part)
        with ThreadPoolExecutor(max_workers=3) as pool:
            maps = list(pool.map(map_one, enumerate(chunks)))
        maps_data = [m.model_dump() for m in maps]
        dump(run.output / 'map_summaries.json', maps_data)
        summary = ask(run, 'reduce_v1', Summary, prompts.REDUCE, maps_data, sources, allowed_evidence=maps_data)
        for version in (1, 2):
            print(f'Проверка сводки судьёй: версия {version}', flush=True)
            dump(run.output / f'summary_v{version}.json', summary.model_dump())
            judge = ask(run, f'judge_v{version}', JudgeReport, prompts.JUDGE,
                        {'sources': payload, 'summary': summary.model_dump()}, sources,
                        expected_actions=[a.action_id for a in summary.action_items], summary=summary)
            dump(run.output / f'judge_v{version}.json', judge.model_dump())
            if judge.overall_score >= .7 or version == 2:
                break
            summary = ask(run, 'reduce_v2', Summary, prompts.REDUCE_REVISED,
                          {'map_summaries': maps_data, 'previous_summary': summary.model_dump(),
                           'judge_feedback': judge.model_dump()}, sources, allowed_evidence=maps_data)
        dump(run.output / 'summary.json', summary.model_dump())
        dump(run.output / 'judge_report.json', judge.model_dump())
        print('Autodiscovery: поиск дополнительных тем', flush=True)
        discovery = ask(run, 'discovery', Discovery, prompts.DISCOVER, payload, sources)
        dump(run.output / 'discovered_aspects.json', discovery.model_dump())
        if diagnostic:
            print('Отдельная контролируемая проверка судьи', flush=True)
            diagnostics(run, sources, summary)
        run.finish('completed')
        metrics = build_metrics(run, sources, summary, judge)
        dump(run.output / 'metrics.json', metrics)
        write_report(run, sources, summary, judge, discovery, metrics)
        if not metrics['numeric_criteria_met']:
            run.finish('needs_review')
            raise RuntimeError('Числовые критерии задания не достигнуты. Все результаты сохранены; нужна доработка по judge_report.json.')
        print(f'Готово: {run.output.resolve()} | judge={judge.overall_score:.3f}', flush=True)
        return metrics
    except BaseException:
        if run.meta['status'] != 'needs_review':
            run.finish('interrupted')
        raise
    finally:
        run.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_path', nargs='?', default=str(ROOT / 'input'))
    parser.add_argument('--output', default=str(ROOT / 'output'))
    parser.add_argument('--no-diagnostics', action='store_true')
    args = parser.parse_args()
    try:
        analyze(args.input_path, args.output, diagnostic=not args.no_diagnostics)
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
