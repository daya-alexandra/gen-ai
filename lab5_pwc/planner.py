from schemas_pwc import Plan

SYSTEM_PROMPT = '''Ты планировщик. Разложи вопрос на минимальный DAG из 1–6 подвопросов.
Инструменты: get_fx_rate(currency,on_date), get_key_rate(on_date),
get_inflation(year,month), calculate(expression). НЕ придумывай другие имена.
На каждый арифметический результат выдели отдельный calculate-подвопрос,
зависящий от всех исходных чисел. Независимые источники получают depends_on=[].
expected_tools — только реально доступные имена, один узкий вопрос на узел.
«Сегодня» — учебный снимок 2026-04-22. Последняя инфляция — 2026-03, % ГОД К ГОДУ.
get_inflation без year/month возвращает последний доступный месяц.
Месячных приростов ИПЦ НЕТ: накопленную инфляцию нельзя получить произведением
годовых темпов. Для такой задачи верни subquestions=[] и объясни отсутствие данных.
Во всех данных есть учебные/синтетические значения, не скрывай это.
Обратная связь требует нового плана, а не повторения ошибочного инструмента.
Для задачи вне возможностей верни пустой план с конкретной причиной.
Ответ — JSON по схеме Plan.'''

def planner(question, *, run, stage, feedback=None):
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': question}]
    if feedback:
        messages.append({'role': 'user', 'content': 'Предыдущая попытка не прошла проверку. Замечание: ' + feedback})
    return run.create(stage=stage, messages=messages, response_model=Plan, temperature=0.0, max_tokens=2600, max_retries=2)
