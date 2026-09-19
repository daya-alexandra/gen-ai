"""Локальные тесты. Тестовые ответы не являются результатами DeepSeek."""
import copy
import json
import os
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError
from llm_client import Run, read, events, usage_cost
from pipeline import ROOT, analyze, load_input, check_quotes, validate_result, split_chunks
from schema import ASPECTS, Review, Reviews, ReviewAspects, AspectScore, JudgeReport
from verify_results import verify


def claim(row):
    return {'text': 'В отзыве описан опыт использования приложения',
            'evidence': [{'review_id': row['review_id'], 'quote': row['text'][-80:]}]}


class FakeAPI:
    def __init__(self):
        self.calls = 0
        self.judges = 0

    def __call__(self, request):
        self.calls += 1
        call_id = self.calls
        schema = json.loads(request['messages'][0]['content'].split('по схеме: ', 1)[1])
        title = schema['title']
        payload = json.loads(next(m['content'] for m in request['messages'] if m['role'] == 'user'))
        if title == 'Reviews':
            value = {'reviews': [{'review_id': r['review_id'], 'author': None, 'published_at': None,
                     'rating': None, 'platform': 'не указана', 'app_version': None,
                     'issues': [claim(r)], 'praise': []} for r in payload]}
        elif title == 'Aspects':
            value = {'reviews': [{'review_id': r['review_id'], 'aspects': [
                {'aspect': a, 'score': -1 if i == 0 else None, 'explanation': 'Тестовая оценка по заглушке',
                 'evidence': claim(r)['evidence'] if i == 0 else []} for i, a in enumerate(ASPECTS)]} for r in payload]}
        elif title == 'MapSummary':
            value = {'source_ids': [r['review_id'] for r in payload], 'facts': [claim(r) for r in payload[:3]]}
        elif title == 'Summary':
            maps = payload['map_summaries'] if isinstance(payload, dict) else payload
            facts = [c for m in maps for c in m['facts']][:3]
            value = {'title': 'Тестовая сводка отзывов', 'findings': facts,
                     'action_items': [{**copy.deepcopy(c), 'action_id': f'A{i+1}', 'priority': 'средний'} for i, c in enumerate(facts[:2])],
                     'limitations': ['Синтетический тестовый ответ, не является API-результатом']}
        elif title == 'JudgeReport':
            self.judges += 1
            score = .5 if self.judges == 1 else .9
            actions = payload['summary']['action_items']
            value = {'faithfulness': score, 'coverage': score, 'actionability': score, 'overall_score': score,
                     'verdicts': [{'action_id': a['action_id'], 'support': 'not_supported' if a['action_id'] == 'A99' else 'supported',
                                  'explanation': 'Контролируемый вердикт тестовой заглушки',
                                  'evidence': [] if a['action_id'] == 'A99' else a['evidence']} for a in actions],
                     'distortions': [], 'feedback': ['Уточнить формулировки по исходным цитатам']}
        elif title == 'Discovery':
            value = {'aspects': [{'name': 'topic_'+str(i), 'description': 'Учебная тема для проверки конвейера',
                                  'related_fixed_aspect': ASPECTS[i] if i < 2 else None,
                                  'evidence': claim(payload[i])['evidence']} for i in range(3)]}
        else:
            raise AssertionError(title)
        return {'id': f'offline-{call_id}', 'model': 'offline-fake',
                'usage': {'prompt_tokens': 100, 'completion_tokens': 50, 'prompt_cache_hit_tokens': 20},
                'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value, ensure_ascii=False)}}]}


class Lab2Tests(unittest.TestCase):
    def setUp(self):
        self.sources = load_input(ROOT / 'input')

    def test_future_date_rejected(self):
        with self.assertRaises(ValidationError):
            Review(review_id='R01', published_at=date.today()+timedelta(days=1), platform='Android', issues=[], praise=[])

    def test_absent_aspect_is_not_positive(self):
        with self.assertRaises(ValidationError):
            AspectScore(aspect='цена', score=2, explanation='Не упоминается', evidence=[])

    def test_exactly_six_different_aspects(self):
        cell = {'aspect': ASPECTS[0], 'score': None, 'explanation': 'Нет упоминания', 'evidence': []}
        with self.assertRaises(ValidationError):
            ReviewAspects(review_id='R01', aspects=[cell]*6)

    def test_quote_cannot_switch_source(self):
        q = {'review_id': 'R02', 'quote': self.sources[0].text[-80:]}
        self.assertFalse(check_quotes(q, self.sources)[0]['valid'])

    def test_changed_negation_and_numbers_are_ghosts(self):
        q = {'review_id': 'R01', 'quote': 'Задачи исчезли'}
        self.assertFalse(check_quotes(q, self.sources)[0]['valid'])
        q['quote'] = 'список открывается примерно за пятьдесят секунд'
        self.assertFalse(check_quotes(q, self.sources)[0]['valid'])

    def test_chunks_cover_input_once(self):
        chunks = split_chunks(self.sources, max_chars=900)
        self.assertEqual([r.review_id for c in chunks for r in c], [r.review_id for r in self.sources])

    def test_cache_tariff(self):
        self.assertAlmostEqual(usage_cost({'prompt_tokens': 1000, 'completion_tokens': 500, 'prompt_cache_hit_tokens': 400}), .0007824)
        self.assertIsNone(usage_cost(None))

    def test_budget_blocks_before_request(self):
        fake = FakeAPI()
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'LLM_BUDGET_USD': '0'}):
            project = Path(d) / 'project'
            project.mkdir()
            (project / '.env').write_text('LLM_BUDGET_USD=1\n', encoding='utf-8')
            with patch('llm_client.ROOT', project):
                run = Run(Path(d) / 'output', {}, transport=fake)
            with self.assertRaises(RuntimeError):
                run.create(stage='unit', messages=[{'role':'user','content':'[]'}], response_model=Reviews)
            self.assertEqual(fake.calls, 0)
            self.assertEqual(os.environ['LLM_BUDGET_USD'], '0')

    def test_live_run_still_loads_dotenv_without_request(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'LLM_BUDGET_USD': '0'}):
            project = Path(d) / 'project'
            project.mkdir()
            (project / '.env').write_text('LLM_BUDGET_USD=0.25\n', encoding='utf-8')
            with patch('llm_client.ROOT', project):
                run = Run(Path(d) / 'output', {})
            self.assertEqual(os.environ['LLM_BUDGET_USD'], '0.25')
            self.assertEqual(run.config['run_kind'], 'llm_api')
            self.assertIsNone(run.raw)

    def test_api_error_is_not_retried_or_leaked(self):
        calls = []
        def bad(request):
            calls.append(request)
            raise RuntimeError('SECRET_VALUE_MUST_NOT_BE_LOGGED')
        with tempfile.TemporaryDirectory() as d:
            run = Run(d, {}, transport=bad)
            with self.assertRaises(RuntimeError):
                run.create(stage='unit', messages=[{'role':'user','content':'[]'}], response_model=Reviews)
            self.assertEqual(len(calls), 1)
            self.assertNotIn('SECRET_VALUE', run.trace.read_text(encoding='utf-8'))

    def test_three_retries_means_four_attempts(self):
        calls = []
        def malformed(request):
            calls.append(1)
            return {'id': str(len(calls)), 'model':'fake', 'usage':{'prompt_tokens':1,'completion_tokens':1},
                    'choices':[{'finish_reason':'stop','message':{'content':'not JSON'}}]}
        with tempfile.TemporaryDirectory() as d:
            run = Run(d, {}, transport=malformed)
            with self.assertRaises(RuntimeError):
                run.create(stage='unit', messages=[{'role':'user','content':'[]'}], response_model=Reviews, max_retries=3)
            self.assertEqual(len(calls), 4)
            self.assertEqual(len(events(run.trace)), 4)

    def test_config_change_refuses_resume(self):
        with tempfile.TemporaryDirectory() as d:
            Run(d, {'input':'one'}, transport=FakeAPI())
            with self.assertRaises(ValueError):
                Run(d, {'input':'two'}, transport=FakeAPI())

    def test_whole_pipeline_repair_resume_and_audit(self):
        fake = FakeAPI()
        with tempfile.TemporaryDirectory() as d:
            result = analyze(ROOT / 'input', d, transport=fake)
            self.assertEqual(result['run_kind'], 'offline_test')
            self.assertTrue((Path(d) / 'judge_v2.json').exists())
            self.assertEqual(read(Path(d) / 'diagnostic_expected.json')['judge_detected_action'], 'not_supported')
            self.assertTrue(verify(d, ROOT / 'input', allow_test=True)['verified'])
            with self.assertRaises(ValueError):
                verify(d, ROOT / 'input')
            count = fake.calls
            analyze(ROOT / 'input', d, transport=fake)
            self.assertEqual(fake.calls, count)
            self.assertIn('ТЕСТОВЫЙ ПРОГОН', (Path(d) / 'выводы.md').read_text(encoding='utf-8'))
            edited = read(Path(d) / 'reviews.json')
            edited['reviews'][0]['author'] = 'Изменённый автор'
            (Path(d) / 'reviews.json').write_text(json.dumps(edited, ensure_ascii=False), encoding='utf-8')
            with self.assertRaises(AssertionError):
                verify(d, ROOT / 'input', allow_test=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
