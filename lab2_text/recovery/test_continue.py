"""Регрессионные тесты продолжения. Все ответы здесь — явные заглушки."""
import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import continue_lab2 as recovery
from llm_client import events, read, dump
from test_lab2 import FakeAPI
from pydantic import ValidationError


class LimitedAPI(FakeAPI):
    def __init__(self):
        super().__init__()
        self.fail_summary=True
        self.titles=[]
        self.limits=[]

    def __call__(self,request):
        schema=json.loads(request['messages'][0]['content'].split('по схеме: ',1)[1])
        title=schema['title']
        self.titles.append(title);self.limits.append(request['max_tokens'])
        if title=='Summary' and self.fail_summary:
            self.calls+=1
            return {'id':f'offline-{self.calls}','model':'offline-test',
                    'usage':{'prompt_tokens':100,'completion_tokens':5000},
                    'choices':[{'finish_reason':'length','message':{'content':'{"title":"Оборванный тестовый JSON"'}}]}
        return super().__call__(request)


class ContinueTests(unittest.TestCase):
    def test_fresh_run_and_profile_change_guard(self):
        fake=LimitedAPI();fake.fail_summary=False
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            recovery.continue_run(recovery.ROOT/'input',out,transport=fake)
            result=recovery.verify_continuation(out,recovery.ROOT/'input',allow_test=True)
            self.assertEqual(result['previously_cached_stages'],0)
            before=(out/'run.json').read_bytes();calls=fake.calls
            manifest=read(out/'continuation.json');manifest['policy']['name']='modified'
            dump(out/'continuation.json',manifest)
            with self.assertRaises(ValueError):
                recovery.continue_run(recovery.ROOT/'input',out,transport=fake)
            self.assertEqual(fake.calls,calls)
            self.assertEqual((out/'run.json').read_bytes(),before)

    def test_preserves_prefix_then_reuses_completed_run(self):
        fake=LimitedAPI()
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{'LLM_BUDGET_USD':'1','LLM_MAX_CALLS':'120'}):
            out=Path(d)
            with self.assertRaises(RuntimeError):
                recovery.pipeline.analyze(recovery.ROOT/'input',out,transport=fake)
            self.assertEqual(read(out/'run.json')['status'],'interrupted')
            prefix=(out/'api_trace.jsonl').read_bytes()
            self.assertEqual(sum(e['status']=='valid' for e in events(out/'api_trace.jsonl')),17)
            old_cache={p.name:p.read_bytes() for p in (out/'cache').glob('*.json')}
            old_calls=fake.calls
            fake.fail_summary=False
            metrics=recovery.continue_run(recovery.ROOT/'input',out,transport=fake)
            self.assertEqual(metrics['run_kind'],'offline_test')
            self.assertEqual(fake.titles[old_calls:],['Summary','JudgeReport','Summary','JudgeReport','Discovery','JudgeReport'])
            self.assertEqual(fake.limits[old_calls:],[12000,10000,12000,10000,8000,10000])
            self.assertTrue((out/'api_trace.jsonl').read_bytes().startswith(prefix))
            self.assertTrue(all((out/'cache'/name).read_bytes()==b for name,b in old_cache.items()))
            result=recovery.verify_continuation(out,recovery.ROOT/'input',allow_test=True)
            self.assertEqual(result['previously_cached_stages'],17)
            with self.assertRaises(ValueError):
                recovery.verify_continuation(out,recovery.ROOT/'input')
            calls=fake.calls
            recovery.continue_run(recovery.ROOT/'input',out,transport=fake)
            self.assertEqual(fake.calls,calls)
            self.assertEqual(metrics['api_responses_including_diagnostics'],27)
            manifest=read(out/'continuation.json')
            manifest['policy']['profiles']['Summary']['max_tokens']=1
            dump(out/'continuation.json',manifest)
            with self.assertRaises(ValueError):
                recovery.verify_continuation(out,recovery.ROOT/'input',allow_test=True)

    def test_summary_limits_are_enforced(self):
        evidence=[{'review_id':'R01','quote':'Подходящая тестовая цитата'}]
        claim={'text':'Короткий контролируемый факт','evidence':evidence}
        value={'title':'Краткая тестовая сводка','findings':[copy.deepcopy(claim) for _ in range(5)],
               'action_items':[{**claim,'action_id':'A'+str(i),'priority':'средний'} for i in (1,2,3)],
               'limitations':['Синтетическая проверка формата']}
        recovery.CompactSummary.model_validate(value)
        value['findings'].append(copy.deepcopy(claim))
        with self.assertRaises(ValidationError):recovery.CompactSummary.model_validate(value)
        value['findings'].pop();value['findings'][0]['evidence']=evidence*3
        with self.assertRaises(ValidationError):recovery.CompactSummary.model_validate(value)

    def test_changed_baseline_rejected_before_calls(self):
        fake=LimitedAPI()
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            with self.assertRaises(RuntimeError):
                recovery.pipeline.analyze(recovery.ROOT/'input',out,transport=fake)
            meta=read(out/'run.json');meta['config']['code_sha256']['pipeline.py']='changed'
            dump(out/'run.json',meta);calls=fake.calls
            with self.assertRaises(ValueError):
                recovery.continue_run(recovery.ROOT/'input',out,transport=fake)
            self.assertEqual(fake.calls,calls)

    def test_modified_saved_cache_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'cache').mkdir()
            cache=out/'cache'/'value.json';cache.write_bytes(b'original')
            trace=out/'api_trace.jsonl';trace.write_bytes(b'old log\r\nnew log\r\n')
            m={'preserved_cache_sha256':{'cache/value.json':hashlib.sha256(b'original').hexdigest()},
               'trace_prefix_bytes':9,'trace_prefix_sha256':hashlib.sha256(b'old log\r\n').hexdigest()}
            recovery.verify_preserved_prefix(out,m)
            cache.write_bytes(b'edited')
            with self.assertRaises(ValueError):recovery.verify_preserved_prefix(out,m)
            cache.write_bytes(b'original');trace.write_bytes(b'EDITED!\r\nnew log\r\n')
            with self.assertRaises(ValueError):recovery.verify_preserved_prefix(out,m)


if __name__=='__main__':unittest.main(verbosity=2)
