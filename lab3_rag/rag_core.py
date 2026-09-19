"""Детерминированный BM25 и две стратегии чанкинга с исходными смещениями."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = {'retriever':'BM25', 'k1':1.5, 'b':.75, 'top_k':5,
          'A':{'kind':'fixed', 'size':2000, 'overlap':0},
          'B':{'kind':'boundary', 'size':800, 'overlap':120}}
STOP = set('и в во на с со к ко у о об от до по за из без для при а но или что это как где когда ли не же то бы быть есть все всего'.split())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint():
    return {p.name:sha(p.read_bytes()) for p in sorted(ROOT.glob('*.py'))}


def load_corpus(data_dir):
    files = sorted(p for p in Path(data_dir).iterdir() if p.is_file() and p.suffix in {'.txt','.md'})
    if not 5 <= len(files) <= 15:
        raise ValueError('Нужны 5–15 документов .txt/.md в data/')
    docs = {p.stem:p.read_text(encoding='utf-8-sig') for p in files}
    if len(docs) != len(files) or any('__' in key for key in docs):
        raise ValueError('Имена источников должны быть уникальны и не содержать __')
    counts = {key:len(re.findall(r'\b\w+\b', text)) for key,text in docs.items()}
    bad = {key:n for key,n in counts.items() if not 500 <= n <= 5000}
    if bad:
        raise ValueError('В каждом документе нужны 500–5000 слов: '+str(bad))
    if sum(map(len, docs.values())) < 30000:
        raise ValueError('Общий корпус должен содержать не менее 30000 символов')
    return docs


def corpus_stats(docs):
    return {'documents':len(docs), 'characters':sum(map(len,docs.values())),
            'word_count_rule':r'Количество совпадений \b\w+\b; символы Python Unicode, включая пробелы',
            'files':{key:{'characters':len(text),'words':len(re.findall(r'\b\w+\b',text)),
                          'sha256':sha(text.encode())} for key,text in docs.items()}}


def load_gold(path, docs):
    rows = read(path)
    if not isinstance(rows,list) or len(rows) < 10:
        raise ValueError('В gold должны быть минимум 10 вопросов')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Повторяющиеся ID gold')
    allowed = {'прямой','синоним','точный артикул','multi-hop','позитив','перефраз'}
    for row in rows:
        if row['type'] not in allowed or not row['question'].strip():
            raise ValueError('Некорректный вопрос или тип')
        sources = row['gold_sources']
        if not sources or len(set(sources)) != len(sources) or not set(sources) <= set(docs):
            raise ValueError('Gold ссылается на отсутствующий или повторяющийся источник')
        if row['type'] == 'multi-hop' and len(sources) < 2:
            raise ValueError('Для multi-hop нужны как минимум два источника')
    if sum(r['type'] in {'multi-hop','синоним'} and r.get('hard',False) for r in rows) < 2:
        raise ValueError('Нужны минимум два отмеченных сложных вопроса')
    return rows


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_id: str
    start: int
    end: int
    text: str


def fixed_spans(text, size=2000):
    if size <= 0:
        raise ValueError('size должен быть положительным')
    return [(i,min(i+size,len(text))) for i in range(0,len(text),size)]


def boundary_spans(text, size=800, overlap=120):
    if not 0 <= overlap < size or size < 2:
        raise ValueError('Нужно 0 <= overlap < size')
    spans, start = [], 0
    while start < len(text):
        end = min(start+size,len(text))
        if end < len(text):
            low = start + max(size//2, overlap+1)
            # Приоритет абзаца, строки, предложения, затем слова.
            for separator in ('\n\n','\n','. ','? ','! ','; ',' '):
                point = text.rfind(separator,low,end)
                if point >= low:
                    end = point + len(separator)
                    break
        spans.append((start,end))
        if end == len(text):
            break
        start = end-overlap
    return spans


def build_chunks(docs, strategy):
    if strategy not in {'A','B'}:
        raise ValueError('Стратегия должна быть A или B')
    result = []
    for source,text in docs.items():
        spans = fixed_spans(text) if strategy == 'A' else boundary_spans(text)
        result += [Chunk(f'{source}__{i}',source,a,b,text[a:b]) for i,(a,b) in enumerate(spans)]
    return result


def tokens(text):
    return [w for w in re.findall(r'[а-яёa-z0-9]+(?:[_.-][а-яёa-z0-9]+)*',text.lower()) if len(w)>1 and w not in STOP]


class BM25:
    def __init__(self, chunks, k1=1.5, b=.75):
        if not chunks:
            raise ValueError('Пустой индекс')
        self.chunks = chunks
        self.k1,self.b = k1,b
        self.counts = [Counter(tokens(c.text)) for c in chunks]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avglen = sum(self.lengths)/len(chunks) or 1
        df = Counter(t for counts in self.counts for t in counts)
        self.idf = {t:math.log(1+(len(chunks)-n+.5)/(n+.5)) for t,n in df.items()}

    def rank(self, query):
        terms = sorted(set(tokens(query)))
        values = []
        for chunk,counts,length in zip(self.chunks,self.counts,self.lengths):
            score = sum(self.idf.get(t,0)*counts[t]*(self.k1+1)/
                        (counts[t]+self.k1*(1-self.b+self.b*length/self.avglen))
                        for t in terms if counts[t])
            values.append({**asdict(chunk),'score':score})
        values.sort(key=lambda r:(-r['score'],r['chunk_id']))
        return values

    def retrieve(self, query, k=5):
        return self.rank(query)[:k]


def hit_rate(retrieved, gold_sources):
    """Как в eval.py преподавателя: доля обязательных документов в TOP-5 чанков."""
    if not gold_sources:
        raise ValueError('Пустой gold_sources')
    present = {r['source_id'] for r in retrieved}
    return len(present & set(gold_sources))/len(set(gold_sources))
