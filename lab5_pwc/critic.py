import json
from schemas_pwc import Verdict

SYSTEM = '''Ты независимый критик ответов. Верни JSON Verdict.
Проверь покрытие всех подвопросов; согласованность чисел зависимостей; обязательный
calculate для производных чисел; отсутствие ошибок, выдуманных инструментов и
подмены г/г на месячную инфляцию. Не используй свои знания вместо наблюдений.
Первичные числа допустимы от get_fx_rate/get_key_rate/get_inflation без calculate.
Производная арифметика должна иметь calculate в used_tools соответствующего ответа.
Ты видишь только финальные ответы и имена инструментов, не их сырую трассу.
Если происхождение числа нельзя проверить по этим данным, отметь ограничение;
не считай само название инструмента доказательством правильного вычисления.
ok=true совместим только с action=accept и пустым rework_ids.
Для конкретных исправлений: ok=false, action=rework, непустые существующие id.
Если план не охватывает вопрос: ok=false, action=replan, rework_ids=[].
Не считай рекомендации Планировщика доказательством собственной правильности.'''

def critic_payload(question, plan, answers):
    return {'question': question, 'plan': plan.model_dump(),
            'answers': {str(i): {'answer': a.answer, 'used_tools': a.used_tools} for i, a in sorted(answers.items())}}

def critic(question, plan, answers, *, run, stage, temperature=.7, max_retries=2):
    return run.create(stage=stage, messages=[{'role': 'system', 'content': SYSTEM},
                      {'role': 'user', 'content': json.dumps(critic_payload(question, plan, answers), ensure_ascii=False)}],
                      response_model=Verdict, temperature=temperature, max_tokens=1600, max_retries=max_retries)
