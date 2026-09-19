"""90 сравнений + 100 независимых вызовов критика + парные замеры времени."""
import argparse
import json
import statistics
from pathlib import Path
from agent_s5 import run_agent
from benchmarks import measure
from critic import critic
from critic_cases import broken_cases
from eval_pwc import CASES, CONFIGS, check_case
from llm_client import ModelOutputError, dump
from orchestrator import run_pwc
from suite_common import make_run, cached_task, api_totals, seal_output, experiment_lock
from validator_control import run_control

N_EVAL, N_CRITIC, N_BENCH = 5, 10, 3

def evaluate_one(case, config, run, stage):
    try:
        if config == 'single':
            result = run_agent(case['query'], run=run, stage=stage)
        else:
            result = run_pwc(case['query'], run=run, stage=stage, validator=config == 'pwc_validator')
    except ModelOutputError as exc:
        result = {'stage': stage, 'query': case['query'], 'answer': None, 'error': str(exc), 'error_type': 'model_output',
                  'trace': [], 'run_kind': run.config['run_kind']}
    return {'qid': case['id'], 'config': config, 'result': result, 'evaluation': check_case(case, result)}

def critic_one(case, temp, rep, run, stage):
    try:
        verdict = critic(case['question'], case['plan'], case['answers'], run=run, stage=stage, temperature=temp, max_retries=0)
        value, error = verdict.model_dump(), None
    except ModelOutputError as exc:
        value, error = None, str(exc)
    return {'stage': stage, 'case': case['id'], 'temperature': temp, 'rep': rep, 'verdict': value,
            'format_error': error, 'false_accept': bool(value and value['ok']),
            'consistent': verdict.consistent() if value else None}

def summaries(evaluations, trials, benchmarks):
    eval_table = {case['id']: {config: {'passed': sum(r['evaluation']['ok'] for r in evaluations if r['qid'] == case['id'] and r['config'] == config),
                                      'n': sum(r['qid'] == case['id'] and r['config'] == config for r in evaluations)} for config in CONFIGS} for case in CASES}
    critic_table = {}
    for case in broken_cases():
        critic_table[case['id']] = {}
        for temp in (0.0, 0.7):
            selected = [t for t in trials if t['case'] == case['id'] and t['temperature'] == temp]
            critic_table[case['id']][str(temp)] = {'n': len(selected), 'false_accepts': sum(t['false_accept'] for t in selected),
                                                  'format_errors': sum(t['verdict'] is None for t in selected),
                                                  'inconsistent': sum(t['consistent'] is False for t in selected)}
    timing = {}
    for qid in ('Q1', 'Q5'):
        pairs = [i for i in (1, 2, 3) if len([r for r in benchmarks if r['qid'] == qid and r['pair'] == i and r['valid_answers']]) == 2]
        seq = [r['seconds'] for r in benchmarks if r['qid'] == qid and not r['parallel'] and r['pair'] in pairs]
        par = [r['seconds'] for r in benchmarks if r['qid'] == qid and r['parallel'] and r['pair'] in pairs]
        a, b = (statistics.median(seq), statistics.median(par)) if seq else (None, None)
        timing[qid] = {'sequential_seconds': seq, 'parallel_seconds': par,
                       'valid_pairs': pairs, 'median_sequential': a, 'median_parallel': b, 'speedup': a / b if b else None}
    return {'eval': eval_table, 'critic': critic_table, 'timing': timing}

def write_report(output, metrics, evaluations, controls):
    m = metrics
    lines = ['# Лабораторная № 5. Планировщик — Исполнитель — Критик', '',
             '**Режим: ' + ('реальный API-прогон' if m['run_kind'] == 'llm_api' else 'ЛОКАЛЬНЫЙ ТЕСТ, НЕ API-ПРОГОН') + '**', '',
             '## 1. Валидатор схемы', '',
             '`validate_plan(plan: Plan) -> list[str]` проверяет имена инструментов, пустой список инструментов, повтор id/зависимостей, несуществующие зависимости и циклы. Ошибка вызывает перепланировку до запуска любого исполнителя; максимум три попытки. Существуют только get_fx_rate, get_key_rate, get_inflation, calculate.', '']
    for case in controls['caught_plans']:
        lines += ['```json', json.dumps(case, ensure_ascii=False, indent=2), '```', '']
    lines += ['Контролируемая проверка Q4: одиночный диспетчер и PWC без валидатора отвергают инъекцию; PWC с валидатором получает обратную связь и исполняет исправленный план. Планировщик в этой проверке скриптовый, инструменты настоящие локальные, API-вызовов нет. Гарантия относится именно к фиксированной инъекции. В обычном API-eval Q4 все три конфигурации могут справиться: вероятностную галлюцинацию нельзя гарантировать промптом.', '',
              '## 2. Параллельность', '',
              'Для Q1 и Q5 использован один и тот же заранее заданный валидный план в последовательном и параллельном режимах. Три пары измерений; порядок чередуется. Измеряется wall time только исполнителей, без планировщика, критика и синтеза. Ответы API имеют новые ключи этапов; локальный кэш не ускоряет эти замеры.', '',
              '| Вопрос | Последовательно, медиана с | Параллельно, медиана с | Ускорение |', '|---|---:|---:|---:|']
    for qid, t in m['timing'].items():
        if t['speedup'] is not None:
            lines.append(f"| {qid} ({len(t['valid_pairs'])} валидных пар) | {t['median_sequential']:.3f} | {t['median_parallel']:.3f} | {t['speedup']:.3f}× |")
        else:
            lines.append(f'| {qid} | — | — | Нет пары с корректными ответами |')
    lines += ['', 'Сравниваются только пары, где обе стороны дали нужные числа и вызвали ожидаемые инструменты. Все исходные замеры, включая неуспешные, сохранены в timings.json. Значение менее 1 означает замедление, оно сохраняется без исправления. Малое число пар и изменение задержек API ограничивают вывод; кэш провайдера может влиять на время.', '',
              '## 3. Замер критики', '',
              'Пять заведомо повреждённых входов × две температуры × 10 отдельных API-вызовов. Внутри одного испытания нет автоматического повтора JSON-валидации. raw_trace критик не получает. Любое ok=true считается ложным принятием, в том числе при противоречивом action; ошибки формата отмечены отдельно и не засчитываются как правильное отклонение.', '',
              '| Битый кейс | T=0.0, ложных принятий | T=0.7, ложных принятий | Ошибки формата 0.0 / 0.7 |', '|---|---:|---:|---:|']
    for case in broken_cases():
        a, b = m['critic'][case['id']]['0.0'], m['critic'][case['id']]['0.7']
        lines.append(f"| {case['label']} | {a['false_accepts']}/{a['n']} | {b['false_accepts']}/{b['n']} | {a['format_errors']} / {b['format_errors']} |")
    a = sum(x['0.0']['false_accepts'] for x in m['critic'].values())
    b = sum(x['0.7']['false_accepts'] for x in m['critic'].values())
    if b < a:
        conclusion = 'На этих пяти входах при T=0.7 ложных принятий меньше; это ограниченное наблюдение в пользу гипотезы, не универсальное доказательство.'
    elif b == a:
        conclusion = 'Число ложных принятий одинаково: этот опыт не показывает, что повышение температуры лечит соглашательство.'
    else:
        conclusion = 'При T=0.7 ложных принятий больше: на этом наборе гипотеза улучшения не подтвердилась.'
    lines += ['', conclusion, '', 'T=0 также вызывается десять раз: одинаковые ответы не подменяются одной копией. Это повторяемость поведения API, не гарантия статистической независимости. Число исходных кейсов мало, доверительные выводы по качеству в целом не делаются.', '',
              '## 4. Eval 6 × 3', '',
              '| id | query | Одиночный N=5 | PWC N=5 | PWC + валидатор N=5 |', '|---|---|---:|---:|---:|']
    for case in CASES:
        row = m['eval'][case['id']]
        values = [f"{row[c]['passed']}/{row[c]['n']}" for c in CONFIGS]
        lines.append(f"| {case['id']} | {case['query']} | " + ' | '.join(values) + ' |')
    lines += ['', 'Для всех трёх конфигураций одинаковые правила оценки: завершённый ответ, успешные требуемые вызовы, ожидаемые числа и единицы, раскрытие источника. Это проверка заданных признаков, не полная смысловая экспертиза.', '',
              'Q3: CSV содержит cpi_yoy, а не месячные приросты. Поэтому мотивированный отказ с указанием годовой природы данных считается корректным. Произведение 51 годового значения не используется. Q4 в таблице — обычный API-вопрос; гарантированный контроль выше публикуется отдельно, его нельзя выдавать за эти пять повторов.', '', '### Неудачные прогоны', '']
    for case in CASES:
        fails = [r for r in evaluations if r['qid'] == case['id'] and not r['evaluation']['ok']]
        if fails:
            reasons = sorted({k for r in fails for k, v in r['evaluation']['checks'].items() if not v})
            lines.append(f"- {case['id']}: не прошло {len(fails)}/15; причины: {', '.join(reasons)}. Полные ответы и трассы сохранены в eval_results.json и tasks/.")
    if all(r['evaluation']['ok'] for r in evaluations):
        lines.append('Все заданные проверки пройдены; искусственно ухудшать результаты для сравнительной таблицы нельзя.')
    lines += ['', '## 5. Вывод', '',
              'PWC оправдан, когда вопрос действительно раскладывается на независимые источники и вычисление, а проверяемые ограничения плана позволяют остановить ошибку до исполнения. Для простого одного запроса дополнительный планировщик и критик добавляют стоимость и задержку; принимать решение о выгоде следует по таблицам этого прогона, а не по названию архитектуры.', '',
              'Проверка структуры не создаёт отсутствующие данные и не доказывает правильность чисел. Критик без сырых наблюдений ограничен доступными финальными ответами; использованные учебные CSV включают синтетические значения.', '',
              f"Ответов API: {m['api_responses']}; токенов вход/выход: {m['prompt_tokens']}/{m['completion_tokens']}. Расчётная стоимость ${m['estimated_usd_interval'][0]:.5f}–${m['estimated_usd_interval'][1]:.5f}; фактический биллинг отдельно не проверен.", '']
    (Path(output) / 'report.md').write_text('\n'.join(lines), encoding='utf-8')

def execute(output='output', transport=None):
    output = Path(output)
    settings = {'cases': CASES, 'configs': list(CONFIGS), 'n_eval': N_EVAL, 'n_critic': N_CRITIC, 'n_bench': N_BENCH}
    run = make_run(output, 'lab5', settings, transport)
    try:
        evaluations = []
        for case in CASES:
            for rep in range(1, N_EVAL + 1):
                order = CONFIGS[rep % 3:] + CONFIGS[:rep % 3]
                for config in order:
                    stage = f"eval/{case['id']}/{config}/rep{rep}"
                    row = cached_task(output, stage, lambda c=case, co=config, s=stage: evaluate_one(c, co, run, s))
                    evaluations.append({**row, 'rep': rep})
                    dump(output / 'eval_results.json', evaluations)
                    print(f"Eval {len(evaluations)}/90: {case['id']} {config} #{rep} {'OK' if row['evaluation']['ok'] else 'FAIL сохранён'}", flush=True)
        trials = []
        for case in broken_cases():
            for rep in range(1, N_CRITIC + 1):
                for temp in ((0.0, 0.7) if rep % 2 else (0.7, 0.0)):
                    stage = f"critic_probe/{case['id']}/T{temp}/rep{rep}"
                    row = cached_task(output, stage, lambda c=case, t=temp, n=rep, s=stage: critic_one(c, t, n, run, s))
                    trials.append(row)
                    dump(output / 'critic_trials.json', trials)
                    print(f"Критик {len(trials)}/100: {case['id']} T={temp} #{rep}", flush=True)
        measurements = []
        for qid in ('Q1', 'Q5'):
            for pair in range(1, N_BENCH + 1):
                for parallel in ((False, True) if pair % 2 else (True, False)):
                    stage = f'benchmark/{qid}/pair{pair}/parallel{parallel}'
                    row = cached_task(output, stage, lambda q=qid, p=parallel, s=stage: measure(q, p, run=run, stage=s))
                    assert row['cache_replays'] == 0, 'Замер времени попал в локальный кэш'
                    measurements.append({**row, 'pair': pair})
                    dump(output / 'timings.json', measurements)
                    print(f'Замер {len(measurements)}/12: {qid}, parallel={parallel}, {row["seconds"]:.2f} с', flush=True)
        controls = run_control(output / 'validator_control')
        metrics = {'run_kind': run.config['run_kind'], **summaries(evaluations, trials, measurements), **api_totals(output)}
        dump(output / 'metrics.json', metrics)
        write_report(output, metrics, evaluations, controls)
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
    a = p.parse_args()
    with experiment_lock(a.output):
        execute(a.output)

if __name__ == '__main__':
    main()
