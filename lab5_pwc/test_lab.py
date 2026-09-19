"""Тесты инфраструктуры PWC без API и без измерения качества модели."""
import io
import json
import os
import tempfile
import threading
import unittest
import uuid
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from llm_client import read, dump, events
from suite_common import make_run
from schemas_pwc import Plan, SubQuestion, WorkerAnswer, Verdict, validate_plan, _topological_levels
from worker import execute_level, worker
from critic import critic_payload
from critic_cases import broken_cases
from orchestrator import run_pwc, dependent_closure

def node(i, deps=(), tools=('calculate',)):
    return SubQuestion(id=i, question='Вычисли 5.75', expected_tools=list(tools), depends_on=list(deps))

def wa(i):
    return WorkerAnswer(subquestion_id=i, question_snippet='Тест', answer='Учебный CSV: 5.75%', used_tools=['calculate'])

class ScriptedAPI:
    def __init__(self):
        self.calls = 0
        self.lock = threading.Lock()
    def __call__(self, request):
        with self.lock:
            self.calls += 1
        finish = 'stop'
        if 'response_format' in request:
            schema = json.loads(request['messages'][0]['content'].split('схеме: ', 1)[1])['title']
            if schema == 'Plan':
                value = Plan(reasoning='Скриптовый тест конвейера.', subquestions=[node(1)]).model_dump()
            elif schema == 'Verdict':
                # Намеренно принимает всё, чтобы проверить, что false_accept не обнуляется кодом.
                value = {'ok': True, 'reason': 'Скриптовый ответ, не вывод модели', 'action': 'accept', 'rework_ids': []}
            else:
                value = {'answer': 'Учебный CSV: 5.75%.'}
            message = {'role': 'assistant', 'content': json.dumps(value, ensure_ascii=False)}
        elif any(m['role'] == 'tool' for m in request['messages']):
            message = {'role': 'assistant', 'content': 'Учебный CSV: 5.75%.'}
        else:
            message = {'role': 'assistant', 'tool_calls': [{'id': 'scripted_call', 'type': 'function',
                'function': {'name': 'calculate', 'arguments': '{"expression":"(21-9.5)/2"}'}}]}
            finish = 'tool_calls'
        return {'id': 'offline-' + uuid.uuid4().hex, 'model': 'scripted_test',
                'choices': [{'message': message, 'finish_reason': finish}], 'usage': {'prompt_tokens': 20, 'completion_tokens': 10}}

class PWCTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'MACRO_DATA_MODE': 'snapshot', 'LLM_BUDGET_USD': '100', 'LLM_MAX_CALLS': '10000'})
        self.env.start()
    def tearDown(self):
        self.env.stop()
    def test_phantom_tools_and_invalid_dependencies(self):
        for name in ('get_policy_rate_range', 'get_cumulative_inflation'):
            self.assertTrue(validate_plan(Plan(reasoning='x', subquestions=[node(1, tools=(name,))])))
        for nodes in ([node(1, [9])], [node(1, [1])], [node(1, [2]), node(2, [1])], [node(1), node(1)]):
            self.assertTrue(validate_plan(Plan(reasoning='x', subquestions=nodes)))
    def test_levels_preserve_dependencies(self):
        levels = _topological_levels([node(4, [3]), node(2), node(3, [1, 2]), node(1)])
        self.assertEqual([[s.id for s in level] for level in levels], [[1, 2], [3], [4]])
    def test_workers_concurrent_and_only_see_declared_dependencies(self):
        barrier = threading.Barrier(2)
        seen = {}
        def fake(sq, prev, **kwargs):
            seen[sq.id] = set(prev)
            barrier.wait(timeout=2)
            return wa(sq.id)
        result = execute_level([node(3, [1]), node(4, [2])], {1: wa(1), 2: wa(2), 99: wa(99)}, worker_fn=fake)
        self.assertEqual(seen, {3: {1}, 4: {2}})
        self.assertEqual(set(result), {3, 4})
    def test_critic_receives_no_worker_trace(self):
        for case in broken_cases():
            payload = critic_payload(case['question'], case['plan'], case['answers'])
            self.assertNotIn('MUST_NOT_REACH_CRITIC', json.dumps(payload))
            self.assertTrue(all(set(x) == {'answer', 'used_tools'} for x in payload['answers'].values()))
    def test_validator_replans_before_any_worker(self):
        calls = []
        def p(question, feedback=None, **kwargs):
            calls.append('plan')
            if feedback is None:
                return Plan(reasoning='x', subquestions=[node(1, tools=('get_fake',))])
            self.assertIn('get_fake', feedback)
            return Plan(reasoning='x', subquestions=[node(1)])
        def execute(level, prev, **kwargs):
            calls.append('worker')
            return {sq.id: wa(sq.id) for sq in level}
        with tempfile.TemporaryDirectory() as d:
            r = make_run(d, 'test', {}, ScriptedAPI())
            result = run_pwc('x', run=r, stage='v', planner_fn=p, execute_fn=execute,
                             critic_fn=lambda *a, **k: Verdict(ok=True, reason='x', action='accept'), synthesize_fn=lambda *a, **k: '5.75')
        self.assertEqual(calls, ['plan', 'plan', 'worker'])
        self.assertEqual(result['answer'], '5.75')
    def test_rework_invalidates_downstream_answers_only(self):
        plan = Plan(reasoning='x', subquestions=[node(1), node(2), node(3, [1]), node(4, [3])])
        counts, crits = Counter(), []
        def execute(level, prev, **kwargs):
            for sq in level:
                counts[sq.id] += 1
                self.assertTrue(set(sq.depends_on) <= prev.keys())
            return {sq.id: wa(sq.id) for sq in level}
        def c(*args, **kwargs):
            crits.append(1)
            return Verdict(ok=False, reason='Повторить', action='rework', rework_ids=[1]) if len(crits) == 1 else Verdict(ok=True, reason='x', action='accept')
        with tempfile.TemporaryDirectory() as d:
            r = make_run(d, 'test', {}, ScriptedAPI())
            result = run_pwc('x', run=r, stage='r', planner_fn=lambda *a, **k: plan, execute_fn=execute, critic_fn=c, synthesize_fn=lambda *a, **k: 'x')
        self.assertEqual(counts, {1: 2, 2: 1, 3: 2, 4: 2})
        self.assertEqual(result['answer'], 'x')
    def test_inconsistent_verdict_not_accepted_by_orchestrator(self):
        self.assertFalse(Verdict(ok=True, reason='x', action='rework', rework_ids=[1]).consistent())
        self.assertFalse(Verdict(ok=False, reason='x', action='accept').consistent())
    def test_false_accept_metric_keeps_raw_true_even_if_inconsistent(self):
        from run_lab import critic_one
        class FakeRun:
            def create(self, **kwargs):
                return Verdict(ok=True, reason='x', action='rework', rework_ids=[1])
        r = critic_one(broken_cases()[0], .7, 1, FakeRun(), 'probe')
        self.assertTrue(r['false_accept'])
        self.assertFalse(r['consistent'])
    def test_controlled_validator_experiment(self):
        from validator_control import run_control
        with tempfile.TemporaryDirectory() as d:
            result = run_control(d)
        self.assertTrue(result['validated_passed'] and result['single_failed'] and result['pwc_failed'])
        self.assertEqual(result['api_calls'], 0)
    def test_unsupported_cumulative_inflation_is_valid_refusal(self):
        from eval_pwc import CASES, check_case
        good = {'answer': 'Недостаточно данных: есть только инфляция г/г, нет месячных приростов.', 'error': None, 'trace': [], 'plan': {'subquestions': []}}
        self.assertTrue(check_case(CASES[2], good)['ok'])
        good['answer'] = 'Учебная накопленная инфляция 650%.'
        self.assertFalse(check_case(CASES[2], good)['ok'])
    def test_full_counts_real_repeats_resume_and_verification(self):
        from run_lab import execute
        from verify_results import verify
        with tempfile.TemporaryDirectory() as d:
            fake = ScriptedAPI()
            with redirect_stdout(io.StringIO()):
                execute(d, fake)
            result = verify(d, False)
            self.assertEqual(result['eval_runs'], 90)
            self.assertEqual(result['critic_trials'], 100)
            metrics = read(Path(d)/'metrics.json')
            self.assertTrue(all(x['0.0']['false_accepts'] == 10 and x['0.7']['false_accepts'] == 10 for x in metrics['critic'].values()))
            n = fake.calls
            with redirect_stdout(io.StringIO()):
                execute(d, fake)
            self.assertEqual(fake.calls, n)
            verify(d, False)
            with self.assertRaises(AssertionError):
                verify(d, True)
            path = Path(d)/'critic_trials.json'
            trials = read(path)
            trials[0]['false_accept'] = False
            dump(path, trials)
            with self.assertRaises(AssertionError):
                verify(d, False)

if __name__ == '__main__':
    unittest.main()
