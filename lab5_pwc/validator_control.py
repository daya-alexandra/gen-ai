"""Контроль с фиксированной поломкой, гарантия относится ТОЛЬКО к нему."""
from schemas import dispatch
from schemas_pwc import Plan, SubQuestion, WorkerAnswer, Verdict, validate_plan
from orchestrator import run_pwc
from suite_common import make_run
from llm_client import dump

def plans():
    bad = Plan(reasoning='Контролируемая инъекция неизвестного имени.', subquestions=[
        SubQuestion(id=1, question='Ставка 2024-10-28', expected_tools=['get_policy_rate_range'])])
    good = Plan(reasoning='Два разрешённых источника и calculate.', subquestions=[
        SubQuestion(id=1, question='Ставка 2024-10-28', expected_tools=['get_key_rate']),
        SubQuestion(id=2, question='Ставка 2026-04-22', expected_tools=['get_key_rate']),
        SubQuestion(id=3, question='Изменение b-a', expected_tools=['calculate'], depends_on=[1, 2])])
    return bad, good

def run_control(output):
    bad, good = plans()
    state = {'feedback': [], 'executed': []}
    def p(question, *, feedback=None, **kwargs):
        state['feedback'].append(feedback)
        return good if feedback and 'Инструменты не существуют' in feedback else bad
    def execute(level, prev_answers, **kwargs):
        result = {}
        for sq in level:
            if sq.expected_tools == ['get_policy_rate_range']:
                obs = dispatch('get_policy_rate_range', {})
                text, used = '(ошибка: ' + obs['error_type'] + ')', []
            else:
                name = sq.expected_tools[0]
                args = {'on_date': '2024-10-28' if sq.id == 1 else '2026-04-22'} if name == 'get_key_rate' else {'expression': '16-21'}
                obs = dispatch(name, args)
                state['executed'].append(name)
                text, used = 'Учебный CSV: ' + str(obs.get('rate', obs.get('result'))), [name]
            result[sq.id] = WorkerAnswer(subquestion_id=sq.id, question_snippet=sq.question, answer=text, used_tools=used, raw_trace=[{'obs': obs}])
        return result
    def c(question, plan, answers, **kwargs):
        bad_answer = any('ошибка' in a.answer for a in answers.values())
        return Verdict(ok=not bad_answer, reason='Контролируемая проверка', action='replan' if bad_answer else 'accept')
    def synthesis(question, plan, answers, **kwargs):
        return answers[3].answer
    r = make_run(output, 'validator_control', {}, transport=lambda _: (_ for _ in ()).throw(AssertionError('API forbidden')))
    question = 'На сколько п.п. изменилась ставка с 2024-10-28 по 2026-04-22?'
    try:
        single = dispatch('get_policy_rate_range', {})
        plain = run_pwc(question, run=r, stage='controlled/plain', validator=False, max_iter=1, planner_fn=p, execute_fn=execute, critic_fn=c, synthesize_fn=synthesis)
        checked = run_pwc(question, run=r, stage='controlled/validated', validator=True, max_iter=1, planner_fn=p, execute_fn=execute, critic_fn=c, synthesize_fn=synthesis)
        phantom2 = Plan(reasoning='Вторая фантомная функция.', subquestions=[SubQuestion(id=1, question='ИПЦ', expected_tools=['get_cumulative_inflation'])])
        result = {'run_kind': 'controlled_test', 'api_calls': 0, 'single_failed': single.get('error_type') == 'unknown_tool',
                  'pwc_failed': plain['answer'] is None, 'validated_passed': checked['answer'] == 'Учебный CSV: -5.0',
                  'caught_plans': [{'plan': x.model_dump(), 'errors': validate_plan(x)} for x in (bad, phantom2)],
                  'feedback': state['feedback'], 'executed': state['executed'], 'plain': plain, 'validated': checked,
                  'scope': 'Гарантия — на фиксированной инъекции и скриптовой перепланировке. Это НЕ измерение качества DeepSeek.'}
        assert result['single_failed'] and result['pwc_failed'] and result['validated_passed']
        dump(r.output.parent / 'control_results.json', result)
        r.finish('completed')
        return result
    finally:
        r.close()
