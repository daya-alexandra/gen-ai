import time
import uuid
from pathlib import Path
from agent import append_trace
from llm_client import now
from planner import planner
from worker import execute_level
from critic import critic
from schemas_pwc import FinalAnswer, _topological_levels, validate_plan

def dependent_closure(plan, ids):
    todo = set(ids)
    if not todo <= {sq.id for sq in plan.subquestions}:
        raise ValueError('Критик указал несуществующий id')
    while True:
        expanded = todo | {sq.id for sq in plan.subquestions if set(sq.depends_on) & todo}
        if expanded == todo:
            return todo
        todo = expanded

def _synthesize(question, plan, answers, *, run, stage):
    import json
    return run.create(stage=stage, messages=[{'role': 'system', 'content':
                      'Собери короткий итоговый ответ на русском ТОЛЬКО из готовых ответов. Не считай заново. '
                      'Сохрани числа, единицы, даты и оговорки об учебном CSV/синтетике. JSON: {"answer":"..."}.'},
                      {'role': 'user', 'content': json.dumps({'question': question, 'answers': {i: a.answer for i, a in answers.items()}}, ensure_ascii=False)}],
                      response_model=FinalAnswer, temperature=0.0, max_tokens=1600).answer

def run_pwc(question, *, run, stage, validator=True, parallel=True, max_iter=3,
            planner_fn=planner, critic_fn=critic, execute_fn=execute_level, synthesize_fn=_synthesize):
    started, run_id, trace = time.perf_counter(), str(uuid.uuid4()), []
    def log(data):
        row = {'run_id': run_id, 'ts': now(), 'stage': stage, 'run_kind': run.config['run_kind'], **data}
        trace.append(row)
        append_trace(run.output.parent / 'pwc_trace.jsonl', row)
    plan_no = 0
    def plan_with_feedback(feedback=None):
        nonlocal plan_no
        for _ in range(3):
            api_stage = f'{stage}/planner/{plan_no}'
            plan_no += 1
            p = planner_fn(question, run=run, stage=api_stage, feedback=feedback)
            log({'kind': 'plan', 'api_stage': api_stage, 'value': p.model_dump(), 'feedback': feedback})
            if not validator:
                return p
            errors = validate_plan(p)
            log({'kind': 'validation', 'errors': errors, 'plan_index': plan_no - 1})
            if not errors:
                return p
            feedback = 'Инструменты не существуют или нарушена схема: ' + '; '.join(errors)
        raise ValueError('Валидатор отклонил три последовательных плана')
    plan, answers, final, error, rounds = None, {}, None, None, 0
    try:
        plan = plan_with_feedback()
        pending = {sq.id for sq in plan.subquestions}
        worker_feedback = None
        if not plan.subquestions:
            final = 'Недостаточно данных: ' + plan.reasoning
        else:
            for rounds in range(1, max_iter + 1):
                for level_index, level in enumerate(_topological_levels(plan.subquestions)):
                    active = [sq for sq in level if sq.id in pending]
                    if not active:
                        continue
                    updated = execute_fn(active, answers, run=run, stage=f'{stage}/round{rounds}/level{level_index}', parallel=parallel, feedback=worker_feedback)
                    answers.update(updated)
                    for i, a in updated.items():
                        log({'kind': 'worker', 'round': rounds, 'sq_id': i, 'value': a.model_dump()})
                api_stage = f'{stage}/critic/{rounds}'
                verdict = critic_fn(question, plan, answers, run=run, stage=api_stage)
                log({'kind': 'verdict', 'api_stage': api_stage, 'value': verdict.model_dump()})
                if not verdict.consistent():
                    raise ValueError('Противоречивый вердикт критика')
                if verdict.ok:
                    final_stage = f'{stage}/synthesis'
                    final = synthesize_fn(question, plan, answers, run=run, stage=final_stage)
                    log({'kind': 'synthesis', 'api_stage': final_stage, 'answer': final})
                    break
                if rounds == max_iter:
                    error = 'critic_rejected_after_limit'
                    break
                if verdict.action == 'replan':
                    plan = plan_with_feedback(verdict.reason)
                    answers = {}
                    pending = {sq.id for sq in plan.subquestions}
                    worker_feedback = None
                    if not pending:
                        final = 'Недостаточно данных: ' + plan.reasoning
                        break
                else:
                    pending = dependent_closure(plan, verdict.rework_ids)
                    answers = {i: a for i, a in answers.items() if i not in pending}
                    worker_feedback = verdict.reason
    except ValueError as exc:
        error = str(exc)
        log({'kind': 'structural_error', 'error': error})
    return {'run_id': run_id, 'stage': stage, 'query': question, 'answer': final, 'error': error,
            'plan': plan.model_dump() if plan else None, 'answers': {str(i): a.model_dump() for i, a in answers.items()},
            'trace': trace, 'iterations': rounds, 'validator': validator, 'parallel': parallel,
            'run_kind': run.config['run_kind'], 'seconds': time.perf_counter() - started}

if __name__ == '__main__':
    import argparse
    from suite_common import make_run
    from llm_client import dump
    p = argparse.ArgumentParser()
    p.add_argument('query')
    p.add_argument('--output', default='interactive_pwc')
    p.add_argument('--no-validator', action='store_true')
    p.add_argument('--sequential', action='store_true')
    a = p.parse_args()
    r = make_run(a.output, 'interactive', {})
    try:
        result = run_pwc(a.query, run=r, stage=uuid.uuid4().hex, validator=not a.no_validator, parallel=not a.sequential)
        dump(Path(a.output) / 'last_answer.json', result)
        print(result['answer'] or result['error'])
        r.finish('completed')
    finally:
        r.close()
