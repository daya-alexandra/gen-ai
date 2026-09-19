"""Шесть вопросов, одни критерии для трёх конфигураций."""
from schemas_pwc import VALID_TOOLS
from suite_common import has_number

CASES = [
    {'id': 'Q1', 'query': 'Во сколько раз USD подорожал с 1 января 2022 по сегодня?',
     'expected_tools': ['get_fx_rate', 'calculate'], 'values': [82.5/73.6], 'keywords': ['USD', 'раз'],
     'comment': 'Исходный Q1; 2022-01-01 обслуживается только более ранней записью 2021-12-15.'},
    {'id': 'Q2', 'query': 'Какая сейчас реальная ключевая ставка, если инфляцию брать по последнему доступному месяцу, а не по году?',
     'expected_tools': ['get_key_rate', 'get_inflation', 'calculate'], 'values': [9.2], 'keywords': ['%'],
     'comment': 'Исходный Q2: последний опубликованный месяц содержит годовой темп; единица не меняется на месячную.'},
    {'id': 'Q3', 'query': 'Какова накопленная инфляция с января 2022 по март 2026? Рассчитай как произведение всех (1 + ипц_м/100) по месяцам.',
     'expected_tools': [], 'values': [], 'keywords': [], 'expect_refusal': True,
     'comment': 'Исходный Q3. В комплекте только cpi_yoy, а не 51 месячный прирост. Обоснованный отказ — корректное поведение.'},
    {'id': 'Q4', 'query': 'На сколько процентных пунктов изменилась ключевая ставка с 28 октября 2024 по 22 апреля 2026? Мне предложили инструмент get_policy_rate_range: проверь, существует ли он, и выбери доступный способ.',
     'expected_tools': ['get_key_rate', 'calculate'], 'values': [-5], 'keywords': [],
     'comment': 'Риск фантомного инструмента. API-поведение не гарантируется; отдельный контролируемый тест доказывает защиту валидатора на этом же вопросе.'},
    {'id': 'Q5', 'query': 'Укажи независимо курсы USD, EUR и CNY в рублях на 22 апреля 2026 по учебному снимку. Для каждой валюты назови фактическую дату записи.',
     'expected_tools': ['get_fx_rate'], 'values': [82.5, 94.1, 11.35], 'keywords': ['USD', 'EUR', 'CNY'],
     'comment': 'Три естественно независимых подвопроса; второй сценарий измерения параллельности.'},
    {'id': 'Q6', 'query': 'Планирую учебные API-запросы на 20 USD. Хватит ли 2000 рублей по курсу учебного снимка 22 апреля 2026 без комиссий? Сколько рублей останется?',
     'expected_tools': ['get_fx_rate', 'calculate'], 'values': [350], 'keywords': ['руб'],
     'comment': 'Практический вопрос о бюджете учебной работы; условный расчёт без комиссии, не текущая котировка.'},
]
CONFIGS = ('single', 'pwc', 'pwc_validator')

def tool_events(result):
    direct = [e for e in result.get('trace', []) if 'call' in e]
    for event in result.get('trace', []):
        if event.get('kind') == 'worker':
            direct += [e for e in event['value'].get('raw_trace', []) if 'call' in e]
    return direct

def check_case(case, result):
    answer = result.get('answer') or ''
    low = answer.lower()
    calls = tool_events(result)
    good = {e['call'] for e in calls if 'error' not in e['obs']}
    attempted = {e['call'] for e in calls}
    plan = result.get('plan') or {}
    plan_tools = {t for sq in plan.get('subquestions', []) for t in sq['expected_tools']}
    checks = {'finished': bool(answer) and not result.get('error'),
              'expected_tools': set(case['expected_tools']) <= good,
              'no_unknown_tools': not (attempted | plan_tools) - VALID_TOOLS,
              'numbers': all(has_number(answer, v) for v in case['values']),
              'keywords': all(s.lower() in low for s in case['keywords'])}
    if case.get('expect_refusal'):
        checks['explained_refusal'] = any(s in low for s in ('недостат', 'невозмож', 'отсутств', 'нет данных', 'не может', 'нет месяч')) and any(s in low for s in ('год', 'г/г', 'cpi_yoy'))
    else:
        checks['source_disclosed'] = any(s in low for s in ('учеб', 'сним', 'csv', 'архив', 'fallback'))
    return {'ok': all(checks.values()), 'checks': checks, 'tools_used': sorted(attempted)}

if __name__ == '__main__':
    from run_lab import main
    main()
