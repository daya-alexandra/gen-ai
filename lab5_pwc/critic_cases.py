"""Пять фиксированных повреждённых входов, не полученных от API."""
from schemas_pwc import Plan, SubQuestion, WorkerAnswer

def sq(i, question, tools, deps=()):
    return SubQuestion(id=i, question=question, expected_tools=tools, depends_on=list(deps))

def answer(i, text, tools):
    return WorkerAnswer(subquestion_id=i, question_snippet='Контрольный пример', answer=text, used_tools=tools,
                        raw_trace=[{'private_test_marker': 'MUST_NOT_REACH_CRITIC'}])

def broken_cases():
    return [
        {'id': 'arithmetic_without_calculate', 'label': 'Арифметика без calculate', 'question': 'Какова разность EUR и USD?',
         'plan': Plan(reasoning='Два курса и разность.', subquestions=[sq(1, 'Курсы и разность', ['get_fx_rate', 'calculate'])]),
         'answers': {1: answer(1, 'USD=82.5, EUR=94.1, разность=11.6 RUB; учебный CSV.', ['get_fx_rate'])},
         'defect': 'Производное число 11.6 есть, calculate не вызван.'},
        {'id': 'invented_number', 'label': 'Выдуманное число', 'question': 'Какой курс USD по полученным данным?',
         'plan': Plan(reasoning='Получить курс, затем пересказать без изменения.', subquestions=[sq(1, 'Источник USD', ['get_fx_rate']), sq(2, 'Сообщи тот же USD', ['get_fx_rate'], [1])]),
         'answers': {1: answer(1, 'Источник учебный CSV: USD=82.5 RUB.', ['get_fx_rate']), 2: answer(2, 'По данным подвопроса 1 курс USD=999 RUB.', [])},
         'defect': '999 не получено инструментом и противоречит 82.5 в зависимости.'},
        {'id': 'dependency_mismatch', 'label': 'Несогласованные зависимости', 'question': 'Какова реальная ставка: номинальная минус инфляция?',
         'plan': Plan(reasoning='Две ставки, затем calculate.', subquestions=[sq(1, 'Номинальная ставка', ['get_key_rate']), sq(2, 'Инфляция', ['get_inflation']), sq(3, 'Разность', ['calculate'], [1, 2])]),
         'answers': {1: answer(1, 'Номинальная=16% (учебный CSV).', ['get_key_rate']), 2: answer(2, 'Инфляция=6.8% г/г (учебный CSV).', ['get_inflation']), 3: answer(3, 'Использованы 21 и 9.5, реальная ставка=11.5%.', ['calculate'])},
         'defect': 'calculate заявлен, но исходные числа заменены.'},
        {'id': 'missing_answer', 'label': 'Нет одного обязательного ответа', 'question': 'Укажи курсы USD, EUR, CNY.',
         'plan': Plan(reasoning='Независимые валюты.', subquestions=[sq(1, 'USD', ['get_fx_rate']), sq(2, 'EUR', ['get_fx_rate']), sq(3, 'CNY', ['get_fx_rate'])]),
         'answers': {1: answer(1, 'USD=82.5 RUB (CSV).', ['get_fx_rate']), 2: answer(2, 'EUR=94.1 RUB (CSV).', ['get_fx_rate'])},
         'defect': 'Ответ id=3 отсутствует.'},
        {'id': 'tool_error', 'label': 'Ошибка инструмента вместо данных', 'question': 'Какая ключевая ставка?',
         'plan': Plan(reasoning='Один источник.', subquestions=[sq(1, 'Ставка', ['get_key_rate'])]),
         'answers': {1: answer(1, '(ошибка: архив недоступен, данных нет)', [])},
         'defect': 'Фактического ответа нет, принимать нельзя.'},
    ]
