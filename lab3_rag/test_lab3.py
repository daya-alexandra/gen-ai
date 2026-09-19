"""Офлайн-проверки; ответы FakeAPI не являются экспериментом DeepSeek."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError
from eval import run_eval
from pipeline import generate, check_citations
from rag_core import (ROOT, BM25, Chunk, boundary_spans, fixed_spans, hit_rate,
                      load_corpus, load_gold, read, dump)
from schema import RAGAnswer
from verify_results import verify


class FakeAPI:
    def __init__(self):
        self.calls=0

    def __call__(self,request):
        self.calls+=1
        payload=json.loads(next(m['content'] for m in request['messages'] if m['role']=='user'))
        c=payload['context'][0]
        answer={'status':'answered','answer':'Контролируемый тестовый ответ; он не оценивает смысл вопроса.',
                'confidence':.6,'citations':[{'chunk_id':'unknown' if self.calls==1 else c['chunk_id'],
                                            'quote':c['text'][:80]}]}
        return {'id':f'offline-{self.calls}','model':'offline-test',
                'usage':{'prompt_tokens':100,'completion_tokens':30,'prompt_cache_hit_tokens':20},
                'choices':[{'finish_reason':'stop','message':{'content':json.dumps(answer,ensure_ascii=False)}}]}


class RagTests(unittest.TestCase):
    def test_corpus_and_gold_requirements(self):
        docs=load_corpus(ROOT/'data');gold=load_gold(ROOT/'gold.json',docs)
        self.assertEqual(len(docs),5)
        self.assertEqual(len(gold),30)
        self.assertGreaterEqual(sum(len(t) for t in docs.values()),30000)

    def test_fixed_is_exact_slicing(self):
        t='0123456789'*531
        spans=fixed_spans(t)
        self.assertEqual([t[a:b] for a,b in spans],[t[i:i+2000] for i in range(0,len(t),2000)])
        self.assertEqual(''.join(t[a:b] for a,b in spans),t)

    def test_boundary_no_loss_and_bounded_overlap(self):
        for text in ['x'*4567,('Абзац. Предложение!\n\n'*200),'я','']:
            spans=boundary_spans(text)
            coverage=[0]*len(text)
            previous=None
            for a,b in spans:
                self.assertTrue(0<=a<b<=len(text))
                self.assertLessEqual(b-a,800)
                if previous:
                    self.assertEqual(previous[1]-a,120)
                    self.assertGreater(a,previous[0])
                for i in range(a,b):coverage[i]+=1
                previous=(a,b)
            self.assertTrue(all(n>=1 for n in coverage))

    def test_boundary_prefers_paragraph(self):
        text='а'*550+'\n\n'+'б'*900
        self.assertEqual(boundary_spans(text)[0],(0,552))

    def test_invalid_overlap_rejected(self):
        with self.assertRaises(ValueError):boundary_spans('abc',10,10)

    def test_duplicate_source_does_not_inflate_metric(self):
        top=[{'source_id':'doc_a'},{'source_id':'doc_a'},{'source_id':'doc_other'}]
        self.assertEqual(hit_rate(top,['doc_a','doc_b']),.5)

    def test_bm25_prefers_relevant_chunk(self):
        chunks=[Chunk('a__0','a',0,23,'кэш сохраняет готовый ответ'),
                Chunk('b__0','b',0,22,'платёж карта подписка цена')]
        self.assertEqual(BM25(chunks).retrieve('подписка цена')[0]['source_id'],'b')

    def test_same_quote_in_wrong_chunk_rejected(self):
        answer=RAGAnswer(status='answered',answer='Подробный ответ с цитатой.',confidence=.7,
                         citations=[{'chunk_id':'b__0','quote':'точная цитата'}])
        with self.assertRaises(ValueError):
            check_citations(answer,[{'chunk_id':'a__0','text':'Здесь точная цитата источника.'},
                                    {'chunk_id':'b__0','text':'Совершенно другой текст.'}])

    def test_cannot_cite_unretrieved_document_part(self):
        answer=RAGAnswer(status='answered',answer='Подробный ответ с цитатой.',confidence=.7,
                         citations=[{'chunk_id':'a__0','quote':'другая часть документа'}])
        with self.assertRaises(ValueError):
            check_citations(answer,[{'chunk_id':'a__0','text':'Только видимый фрагмент.'}])

    def test_abstention_and_supported_answer_rules(self):
        with self.assertRaises(ValidationError):
            RAGAnswer(status='answered',answer='Неподкреплённый ответ',confidence=.9,citations=[])
        with self.assertRaises(ValidationError):
            RAGAnswer(status='insufficient_context',answer='Не хватает информации',confidence=.9,citations=[])
        a=RAGAnswer(status='insufficient_context',answer='В контексте нет сведений для ответа.',confidence=.1,citations=[])
        self.assertEqual(check_citations(a,[]),[])

    def test_offline_eval_and_tamper_detection(self):
        with tempfile.TemporaryDirectory(prefix='rag_eval_') as d:
            result=run_eval(output=d)
            self.assertTrue(verify(d)['retrieval_verified'])
            with self.assertRaises(ValueError):verify(d,require_api=True)
            result['A']['hit_rate_at_5']=.123
            dump(Path(d)/'evaluation.json',result)
            with self.assertRaises(AssertionError):verify(d)

    def test_complete_generation_retry_resume_and_provenance(self):
        with tempfile.TemporaryDirectory(prefix='rag_api_test_') as d:
            run_eval(output=d)
            fake=FakeAPI()
            m=generate(d,transport=fake)
            self.assertEqual(m['run_kind'],'offline_test')
            self.assertEqual(m['answer_count'],60)
            self.assertEqual(m['ghost_quote_checks'],1)
            self.assertTrue(verify(d,require_api=True,allow_test=True)['api_generation_verified'])
            with self.assertRaises(ValueError):verify(d,require_api=True)
            count=fake.calls
            generate(d,transport=fake)
            self.assertEqual(fake.calls,count)
            a=read(Path(d)/'answers_A.json')
            a[0]['answer']['answer']='Подменённый вручную окончательный ответ.'
            dump(Path(d)/'answers_A.json',a)
            with self.assertRaises(AssertionError):verify(d,allow_test=True)


if __name__=='__main__':
    unittest.main(verbosity=2)
