"""Регрессии: бюджет, реальные инструменты, границы дат и целостность результатов."""
import json
import io
from contextlib import redirect_stdout
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
import tools
from agent import run_agent
from llm_client import Run, dump, read
from schemas import dispatch
from suite_common import make_run, verify_agent

def native_response(message, finish='stop'):
    return {'id': 'offline-' + uuid.uuid4().hex, 'model': 'scripted_test', 'choices': [{'message': message, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 20, 'completion_tokens': 10}}

class ScriptedAPI:
    def __init__(self):
        self.calls = 0
    def __call__(self, request):
        self.calls += 1
        if any(m['role'] == 'tool' for m in request['messages']):
            return native_response({'role': 'assistant', 'content': 'Учебный CSV: результат 5.75%.'})
        return native_response({'role': 'assistant', 'tool_calls': [{'id': 'test_call', 'type': 'function',
            'function': {'name': 'calculate', 'arguments': '{"expression":"(21-9.5)/2"}'}}]}, 'tool_calls')

class LabTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'MACRO_DATA_MODE': 'snapshot', 'LLM_BUDGET_USD': '100', 'LLM_MAX_CALLS': '10000'})
        self.env.start()
    def tearDown(self):
        self.env.stop()
    def test_calculator_blocks_code_and_extreme_work(self):
        self.assertEqual(tools.calculate('(21-9.5)/2')['result'], 5.75)
        for expr in ("__import__('os').getcwd()", '(1).__class__', '2**1000000000', 'exp(10000)', '1/0'):
            self.assertIn('error', tools.calculate(expr))
    def test_all_six_comparison_metrics_and_delegation(self):
        for metric in ('key_rate', 'fx_USD', 'fx_EUR', 'fx_CNY', 'cpi', 'unemployment'):
            r = tools.compare_periods(metric, '2022-01', '2026-03')
            self.assertAlmostEqual(r['delta'], r['b']['value'] - r['a']['value'])
            self.assertAlmostEqual(r['ratio'], r['b']['value'] / r['a']['value'])
        with patch.object(tools, 'get_fx_rate', wraps=tools.get_fx_rate) as f:
            tools.compare_periods('fx_USD', '2022-01', '2026-04')
            self.assertEqual(f.call_count, 2)
    def test_no_future_fallback_and_month_end(self):
        self.assertEqual(tools.get_fx_rate('USD', '2022-01-01')['date'], '2021-12-15')
        self.assertEqual(tools.compare_periods('fx_USD', '2022-01', '2026-04')['a']['date'], '2022-01-15')
        self.assertIn('error', tools.get_fx_rate('USD', '2020-01-01'))
    def test_synthetic_marker_propagates(self):
        self.assertTrue(tools.get_inflation(2026, 3)['synthetic'])
        self.assertFalse(tools.get_inflation(2025, 4)['synthetic'])
        self.assertTrue(tools.compare_periods('cpi', '2022-01', '2026-03')['synthetic'])
        self.assertEqual(tools.get_inflation()['frequency'], 'year_over_year')
    def test_ratio_zero_is_explicit(self):
        with patch.object(tools, 'get_key_rate', return_value={'value': 0, 'date': '2022-01-01', 'source': 'test', 'fixture': True, 'synthetic': False, 'unit': '%' }):
            self.assertIsNone(tools.compare_periods('key_rate', '2022-01', '2026-03')['ratio'])
    def test_schema_and_tool_errors_remain_observations(self):
        self.assertEqual(dispatch('phantom', {})['error_type'], 'unknown_tool')
        self.assertEqual(dispatch('get_inflation', {'month': 13, 'year': 2025})['error_type'], 'bad_arguments')
        self.assertEqual(dispatch('calculate', {'expression': '1/0'})['error_type'], 'tool_error')
    def test_budget_blocks_before_transport_and_env_is_ignored_for_test(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'LLM_BUDGET_USD': '0'}):
            fake = ScriptedAPI()
            r = make_run(d, 'test', {}, fake)
            with self.assertRaises(RuntimeError):
                run_agent('Считай', run=r, stage='budget')
            self.assertEqual(fake.calls, 0)
    def test_auth_errors_are_not_leaked_or_retried(self):
        def fail(_):
            raise RuntimeError('SECRET_DO_NOT_PRINT')
        with tempfile.TemporaryDirectory() as d:
            r = make_run(d, 'test', {}, fail)
            with self.assertRaises(RuntimeError) as cm:
                run_agent('Считай', run=r, stage='auth')
            self.assertNotIn('SECRET_DO_NOT_PRINT', str(cm.exception))
            self.assertNotIn('SECRET_DO_NOT_PRINT', (Path(d)/'generation/api_trace.jsonl').read_text(encoding='utf-8'))
    def test_resume_reuses_api_but_repeat_id_is_fresh(self):
        with tempfile.TemporaryDirectory() as d:
            fake = ScriptedAPI()
            r = make_run(d, 'test', {}, fake)
            a = run_agent('Считай', run=r, stage='rep1')
            n = fake.calls
            b = run_agent('Считай', run=r, stage='rep1')
            self.assertEqual(n, fake.calls)
            self.assertEqual(a['run_id'], b['run_id'])
            self.assertEqual(uuid.UUID(a['run_id']).version, 4)
            run_agent('Считай', run=r, stage='rep2')
            self.assertEqual(fake.calls, n * 2)
    def test_config_changes_block_resume(self):
        with tempfile.TemporaryDirectory() as d:
            make_run(d, 'test', {'n': 5}, ScriptedAPI())
            with self.assertRaises(ValueError):
                make_run(d, 'test', {'n': 6}, ScriptedAPI())
    def test_interrupted_agent_reuses_first_api_response(self):
        with tempfile.TemporaryDirectory() as d:
            fake = ScriptedAPI()
            def unreliable(request):
                if fake.calls == 1:
                    fake.calls += 1
                    raise TimeoutError('test timeout')
                return fake(request)
            r = make_run(d, 'test', {}, unreliable)
            with self.assertRaises(RuntimeError):
                run_agent('Считай', run=r, stage='resume')
            answer = run_agent('Считай', run=r, stage='resume')
            self.assertTrue(answer['answer'])
            self.assertEqual(fake.calls, 3)
    def test_experiment_lock_blocks_second_writer(self):
        from suite_common import experiment_lock
        with tempfile.TemporaryDirectory() as d:
            with experiment_lock(Path(d) / 'output'):
                with self.assertRaises(RuntimeError):
                    with experiment_lock(Path(d) / 'output'):
                        pass
    def test_wrong_answer_is_not_passed_for_tool_presence(self):
        from eval import CASES, check_case
        result = {'answer': 'Учебный CSV: 999%', 'error': None, 'steps': 2,
                  'trace': [{'call': 'get_key_rate', 'args': {}, 'obs': {'rate': 16}}]}
        self.assertFalse(check_case(CASES[0], result)['ok'])
    def test_bootstrap_copies_key_but_preserves_new_budget(self):
        from bootstrap import prepare
        from dotenv import dotenv_values
        with tempfile.TemporaryDirectory() as d:
            parent = Path(d)
            old, new = parent / 'lab3_rag', parent / 'lab4_agent'
            old.mkdir(); new.mkdir()
            (old / '.env').write_text('LLM_AUTH_TOKEN=local-test-value\nLLM_BUDGET_USD=0.01\nLLM_MAX_CALLS=1\n', encoding='utf-8')
            (new / '.env.example').write_text('LLM_AUTH_TOKEN=PASTE_YOUR_KEY_HERE\nLLM_BUDGET_USD=2\nLLM_MAX_CALLS=300\n', encoding='utf-8')
            self.assertTrue(prepare(new))
            values = dotenv_values(new / '.env')
            self.assertEqual(values['LLM_AUTH_TOKEN'], 'local-test-value')
            self.assertEqual(values['LLM_BUDGET_USD'], '2')
            self.assertEqual(values['LLM_MAX_CALLS'], '300')
    def test_bootstrap_never_overwrites_existing_env(self):
        from bootstrap import prepare
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / '.env'
            p.write_text('LLM_AUTH_TOKEN=keep-local-test-value\n', encoding='utf-8')
            original = p.read_bytes()
            self.assertTrue(prepare(Path(d)))
            self.assertEqual(p.read_bytes(), original)
    def test_complete_pipeline_and_tamper_detection(self):
        from run_lab import execute
        from verify_results import verify
        with tempfile.TemporaryDirectory() as d:
            fake = ScriptedAPI()
            with redirect_stdout(io.StringIO()):
                execute(d, fake)
            self.assertEqual(verify(d, False)['questions'], 10)
            n = fake.calls
            with redirect_stdout(io.StringIO()):
                execute(d, fake)
            self.assertEqual(fake.calls, n)
            with self.assertRaises(AssertionError):
                verify(d, True)
            p = Path(d)/'eval_results.json'
            content = read(p)
            content[0]['result']['answer'] = 'Подмена'
            dump(p, content)
            with self.assertRaises(AssertionError):
                verify(d, False)

if __name__ == '__main__':
    unittest.main()
