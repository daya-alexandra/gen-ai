"""4 стартовых + 6 новых вопросов. Оцениваются инструменты И числа."""
from suite_common import has_number

CASES = [
    {'id': 1, 'query': 'Какая сегодня ключевая ставка ЦБ?', 'expected_tools': ['get_key_rate'], 'must_have': [], 'expected_values': [16], 'comment': 'Исходный Q1; сегодня означает дату учебного снимка.'},
    {'id': 2, 'query': 'Сколько стоит доллар сегодня и сколько стоил 1 января 2022?', 'expected_tools': ['get_fx_rate'], 'must_have': ['2021-12-15'], 'expected_values': [82.5, 73.6], 'comment': 'Исходный Q2. На 2022-01-01 берётся последняя запись 2021-12-15, НЕ будущая 2022-01-15.'},
    {'id': 3, 'query': 'Какая сейчас реальная ключевая ставка? (номинальная минус инфляция г/г)', 'expected_tools': ['get_key_rate', 'get_inflation', 'calculate'], 'must_have': ['%'], 'expected_values': [9.2], 'comment': 'Исходный Q3; годовая инфляция из последнего доступного месяца.'},
    {'id': 4, 'query': 'Посчитай, за сколько лет удвоится вклад 100 тыс руб при текущей ключевой ставке (формула 72).', 'expected_tools': ['get_key_rate', 'calculate'], 'must_have': ['год'], 'expected_values': [4.5], 'comment': 'Исходный Q4, формула 72.'},
    {'id': 5, 'query': 'Во сколько раз вырос курс USD с января 2022 по апрель 2026? Используй compare_periods и укажи даты фактических записей.', 'expected_tools': ['compare_periods'], 'must_have': ['раз'], 'expected_values': [82.5/76.2], 'compare_metrics': ['fx_USD'], 'comment': 'Новый: обязательный compare_periods. Период — последнее значение месяца, не среднее.'},
    {'id': 6, 'query': 'На сколько процентных пунктов изменилась безработица с января 2022 по март 2026? Сравни через compare_periods.', 'expected_tools': ['compare_periods'], 'must_have': [], 'expected_values': [-2], 'compare_metrics': ['unemployment'], 'comment': 'Новый: второй обязательный compare_periods, дельта в п.п.'},
    {'id': 7, 'query': 'Нужен курс USD строго на 2020-01-01. Если архив начинается позже, не подменяй дату ближайшей будущей записью, объясни отсутствие данных.', 'expected_tools': ['get_fx_rate'], 'must_have': [], 'expected_values': [], 'expect_refusal': True, 'comment': 'Трудный: запрос раньше самой ранней записи; запрещена утечка будущего.'},
    {'id': 8, 'query': 'Сравни январь 2022 и апрель 2026 по ключевой ставке; затем январь 2022 и март 2026 по инфляции г/г. На сколько п.п. изменился каждый показатель? Используй compare_periods.', 'expected_tools': ['compare_periods'], 'must_have': [], 'expected_values': [7.5, -1.93], 'compare_metrics': ['key_rate', 'cpi'], 'comment': 'Трудный: разные метрики и конечные месяцы; нельзя путать дельту в п.п. с отношением.'},
    {'id': 9, 'query': 'Сколько рублей нужно на 1000 USD по последнему доступному курсу учебного снимка, без банковской комиссии?', 'expected_tools': ['get_fx_rate', 'calculate'], 'must_have': ['руб'], 'expected_values': [82500], 'comment': 'Реальный бытовой вопрос: бюджет валютной покупки, только в условиях снимка.'},
    {'id': 10, 'query': 'Какова реальная годовая доходность вклада под ключевую ставку по формуле Фишера: ((1+ставка/100)/(1+инфляция_гг/100)-1)*100? Возьми последнюю доступную инфляцию и оговори учебный источник.', 'expected_tools': ['get_key_rate', 'get_inflation', 'calculate'], 'must_have': ['%'], 'expected_values': [(1.16/1.068-1)*100], 'comment': 'Реальный вопрос о покупательной способности вклада; это не прогноз доходности.'},
]

def check_case(case, result):
    text = result.get('answer') or ''
    low = text.lower()
    trace = [e for e in result.get('trace', []) if 'call' in e]
    successful = {e['call'] for e in trace if 'error' not in e['obs']}
    attempted = {e['call'] for e in trace}
    used = attempted if case.get('expect_refusal') else successful
    checks = {'finished': bool(text) and not result.get('error'),
              'tools': set(case['expected_tools']) <= used,
              'keywords': all(any(x in low for x in {'год': ('год', 'лет'), 'руб': ('руб', '₽', 'rub'), '%': ('%', 'процент')}.get(s, (s.lower(),))) for s in case['must_have']),
              'numbers': all(has_number(text, x) for x in case['expected_values']),
              'source_disclosed': any(s in low for s in ('учеб', 'сним', 'csv', 'архив', 'fallback'))}
    if case.get('expect_refusal'):
        checks['refusal'] = any(s in low for s in ('нет данных', 'отсутств', 'недостат', 'нет запис', 'невозможно', 'недоступ'))
    metrics = {e['args'].get('metric') for e in trace if e['call'] == 'compare_periods' and isinstance(e['args'], dict) and 'error' not in e['obs']}
    checks['compare_metrics'] = set(case.get('compare_metrics', [])) <= metrics
    return {'ok': all(checks.values()), 'checks': checks, 'tools_used': sorted(attempted), 'steps': result['steps']}

if __name__ == '__main__':
    from run_lab import main
    main()
