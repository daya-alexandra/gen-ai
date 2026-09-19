"""Парные замеры на одинаковых заранее заданных планах; без повторного кэша ответов."""
import time
import uuid
from schemas_pwc import Plan, SubQuestion, _topological_levels, validate_plan
from worker import execute_level
from suite_common import has_number

def benchmark_plan(qid):
    if qid == 'Q1':
        sub = [SubQuestion(id=1, question='Курс USD на 2022-01-01; назови фактическую дату записи.', expected_tools=['get_fx_rate']),
               SubQuestion(id=2, question='Курс USD на 2026-04-22; назови фактическую дату записи.', expected_tools=['get_fx_rate']),
               SubQuestion(id=3, question='Во сколько раз курс USD из ответа 2 больше курса USD из ответа 1?', expected_tools=['calculate'], depends_on=[1, 2])]
    else:
        sub = [SubQuestion(id=i, question=f'Курс {currency} на 2026-04-22; назови фактическую дату записи.', expected_tools=['get_fx_rate'])
               for i, currency in enumerate(('USD', 'EUR', 'CNY'), 1)]
    return Plan(reasoning='Одинаковый фиксированный план для изоляции времени исполнения.', subquestions=sub)

def measure(qid, parallel, *, run, stage):
    plan = benchmark_plan(qid)
    assert not validate_plan(plan)
    # При прерывании измерение начинается заново с новым nonce; неполный кэш не ускоряет замер.
    nonce = uuid.uuid4().hex
    answers = {}
    started = time.perf_counter()
    for i, level in enumerate(_topological_levels(plan.subquestions)):
        answers.update(execute_level(level, answers, run=run, stage=f'{stage}/{nonce}/level{i}', parallel=parallel))
    seconds = time.perf_counter() - started
    cache_replays = sum(e.get('cache_replay', False) for a in answers.values() for e in a.raw_trace if 'call' in e or 'final' in e)
    expected = {1: 73.6, 2: 82.5, 3: 82.5/73.6} if qid == 'Q1' else {1: 82.5, 2: 94.1, 3: 11.35}
    valid = all(not answers[sq.id].agent_result.get('error') and set(sq.expected_tools) <= set(answers[sq.id].used_tools)
                and has_number(answers[sq.id].answer, expected[sq.id]) for sq in plan.subquestions)
    return {'qid': qid, 'parallel': parallel, 'nonce': nonce, 'seconds': seconds, 'cache_replays': cache_replays,
            'valid_answers': valid,
            'plan': plan.model_dump(), 'answers': {str(i): a.model_dump() for i, a in answers.items()},
            'scope': 'Только wall time исполнителей, без планировщика, критика и синтеза. Кэш провайдера может влиять на латентность.'}
