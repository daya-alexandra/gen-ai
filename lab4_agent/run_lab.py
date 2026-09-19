"""Полный eval, отдельная диагностика, отчёт и возобновление."""
import argparse
from pathlib import Path
from agent import run_agent
from diagnostics import run_diagnostics
from eval import CASES, check_case
from llm_client import dump, read
from suite_common import make_run, cached_task, api_totals, seal_output, experiment_lock

def write_report(output, results, diagnostics, totals, kind):
    output = Path(output)
    passed = sum(r['evaluation']['ok'] for r in results)
    lines = ['# Лабораторная № 4. Макроагент', '',
             '**Режим: ' + ('реальный API-прогон' if kind == 'llm_api' else 'ЛОКАЛЬНЫЙ ТЕСТ, НЕ API-ПРОГОН') + '**', '',
             '## 1. Новый инструмент', '',
             '`compare_periods(metric: str, period_a: str, period_b: str) -> dict` вызывает существующие функции получения данных и возвращает delta=b−a, ratio=b/a, фактические даты и источники. Поддерживаются key_rate, fx_USD, fx_EUR, fx_CNY, cpi, unemployment.', '',
             'Примеры: курс USD за январь 2022 / апрель 2026 (Q5), безработица за январь 2022 / март 2026 (Q6), ставка и инфляция в разных периодах (Q8). Месяц означает последнее доступное значение, не среднее. Разность процентных показателей выражена в п.п.', '',
             '## 2. Eval', '', f'Пройдено **{passed} из {len(results)}**. Критерии включают реальные успешные вызовы, ожидаемые числа и оговорку об источнике. Это автоматическая проверка заданных признаков, не полная экспертная оценка текста.', '',
             '| id | query | ok? | steps | tools_used |', '|---|---|---|---|---|']
    for r in results:
        lines.append(f"| Q{r['id']} | {r['query']} | {'да' if r['evaluation']['ok'] else 'нет'} | {r['result']['steps']} | {', '.join(r['evaluation']['tools_used'])} |")
    lines += ['', '### Что не прошло', '']
    failed = [r for r in results if not r['evaluation']['ok']]
    for r in failed:
        reasons = [k for k, ok in r['evaluation']['checks'].items() if not ok]
        lines += [f"- Q{r['id']}: {', '.join(reasons)}. Ответ: {r['result'].get('answer') or r['result'].get('error')}"]
    if not failed:
        lines += ['В обычном eval все заданные проверки прошли. Это не основание выдумывать ошибки; ниже отдельные намеренные сбои.']
    natural = [(r['id'], e) for r in results for e in r['result']['trace'] if e.get('obs', {}).get('error') or e.get('error_type')]
    lines += ['', '## 3. Диагностика', '', f'В обычных прогонах обнаружено событий ошибок: {len(natural)}. Полные трассы находятся в trace.jsonl; восстановленные ошибки тоже остаются в журнале.', '',
              'Три воспроизводимых типа ниже получены **намеренной инъекцией**, как в блоке «ломаем агента» семинара. Ответы транспорта заданы вручную; инструменты и обработчики исполняются реально. Это не три случайные ошибки DeepSeek и не API-результаты. Источник: diagnostics/trace.jsonl.', '']
    import json
    for d in diagnostics:
        lines += [f"### {d['type']}", '', '```jsonl']
        lines += [json.dumps(e, ensure_ascii=False) for e in d['result']['trace'][:3]]
        lines += ['```', '', '**Причина:** ' + d['cause'], '', '**Исправление:** ' + d['fix'], '']
    for q, event in natural[:6]:
        lines += [f'Дополнительное наблюдение из API Q{q}: `{event.get("obs", {}).get("error_type", event.get("error_type"))}`, шаг {event["step"]}.']
    lines += ['', '## 4. Вывод', '',
              'Наиболее хрупко соответствие даты, единицы измерения и смысла метрики: формально правильный вызов и JSON ещё не гарантируют правильный экономический ответ. Валидатор аргументов, ограничение повторов и наблюдаемые ошибки защищают цикл, а проверка чисел и происхождения данных дополняет их.', '',
              'Все числа eval относятся к фиксированному учебному снимку 2026-04-22. В CSV присутствуют синтетические значения; это не текущая официальная статистика. Годовые темпы инфляции не превращаются в помесячные.', '',
              f"Получено ответов API: {totals['api_responses']}; входных токенов {totals['prompt_tokens']}, выходных {totals['completion_tokens']}. Интервал расчётной стоимости по сохранённым тарифам: ${totals['estimated_usd_interval'][0]:.5f}–${totals['estimated_usd_interval'][1]:.5f}; без проверки биллинга.", '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')

def execute(output='output', transport=None):
    output = Path(output)
    run = make_run(output, 'lab4', {'cases': CASES, 'max_iter': 8}, transport)
    try:
        results = []
        for case in CASES:
            stage = f"eval/Q{case['id']}"
            result = cached_task(output, stage, lambda c=case, s=stage: run_agent(c['query'], run=run, stage=s))
            evaluation = check_case(case, result)
            results.append({'id': case['id'], 'query': case['query'], 'result': result, 'evaluation': evaluation})
            dump(output / 'eval_results.json', results)
            print(f"[{case['id']}/10] {'OK' if evaluation['ok'] else 'FAIL (сохранён для разбора)'}", flush=True)
        diagnostics = run_diagnostics(output / 'diagnostics')
        totals = api_totals(output)
        dump(output / 'metrics.json', {'run_kind': run.config['run_kind'], 'cases': 10, 'passed': sum(r['evaluation']['ok'] for r in results), **totals})
        write_report(output, results, diagnostics, totals, run.config['run_kind'])
        run.finish('completed')
        seal_output(output)
    except Exception:
        run.finish('interrupted')
        raise
    finally:
        run.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='output')
    args = p.parse_args()
    with experiment_lock(args.output):
        execute(args.output)

if __name__ == '__main__':
    main()
