"""Resumable real API calls, strict JSON validation and a shared expense ceiling."""
from __future__ import annotations
import hashlib
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
LOCK = threading.RLock()

def now(): return datetime.now(timezone.utc).isoformat()
def digest(obj): return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    temp.replace(path)
def append(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with LOCK, path.open('a', encoding='utf-8') as f: f.write(json.dumps(obj, ensure_ascii=False, default=str) + '\n')
def records(path):
    p=Path(path)
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()] if p.exists() else []
def get_model():
    model=os.getenv('LLM_MODEL','deepseek-flash')
    if model not in {'deepseek-flash','deepseek-v4-flash'}:
        raise ValueError('Расчёт бюджета настроен на DeepSeek Flash. Выбери LLM_MODEL=deepseek-flash.')
    return model

def api_options():
    # Non-thinking keeps temperature meaningful in the critic experiment.
    base=os.getenv('LLM_BASE_URL','https://api.deepseek.com').rstrip('/')
    if base not in {'https://api.deepseek.com', 'https://api.deepseek.com/v1'}:
        raise ValueError('Этот комплект настроен только на официальный DeepSeek API. Проверь LLM_BASE_URL.')
    return {'extra_body': {'thinking': {'type': 'disabled'}}}

def make_raw_client():
    api_options()
    key=os.getenv('LLM_AUTH_TOKEN','')
    if not key or key in {'PASTE_YOUR_KEY_HERE','your_token_here'}:
        raise RuntimeError('Не настроен LLM_AUTH_TOKEN. Заполни локальный .env; см. README.md.')
    return OpenAI(api_key=key, base_url=os.getenv('LLM_BASE_URL','https://api.deepseek.com'),
                  timeout=float(os.getenv('LLM_TIMEOUT','90')), max_retries=0)

def safe_error(exc):
    return f'{type(exc).__name__}' + (f' HTTP {exc.status_code}' if hasattr(exc,'status_code') else '')

def cost(usage):
    """Upper tariff estimate at 2026-09-18; this is not a bank statement."""
    inp=usage.get('prompt_tokens',0) or 0; out=usage.get('completion_tokens',0) or 0
    hit=min(inp, usage.get('prompt_cache_hit_tokens',0) or 0)
    return ((inp-hit)*0.30 + hit*0.006 + out*1.20)/1_000_000

class Budget:
    def __init__(self, path=None): self.path=Path(path or ROOT/'.runtime/budget.json')
    def reserve(self, request):
        # UTF-8 byte count overestimates token count; include protocol overhead.
        upper=(len(json.dumps(request,ensure_ascii=False).encode())+4096)*0.30/1e6 + request.get('max_tokens',2048)*1.20/1e6
        with LOCK:
            state=json.loads(self.path.read_text()) if self.path.exists() else {}
            limit=float(os.getenv('LLM_BUDGET_USD','5'))
            if sum(x['amount'] for x in state.values())+upper > limit:
                raise RuntimeError(f'Достигнут общий бюджет {limit} USD. Состояние сохранено.')
            if len(state)>=int(os.getenv('LLM_MAX_CALLS','3000')): raise RuntimeError('Достигнут общий лимит API-вызовов.')
            token=uuid.uuid4().hex;state[token]={'amount':upper,'status':'reserved','ts':now()};dump(self.path,state)
            return token
    def settle(self, token, usage):
        with LOCK:
            state=json.loads(self.path.read_text());state[token].update(amount=cost(usage),status='accounted');dump(self.path,state)

class Run:
    def __init__(self, output, config=None, raw=None, budget=None):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.raw=raw;self.budget=budget or Budget();self.mock=raw is not None
        files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*.py') if not any(s in p.parts for s in ('.venv','__pycache__'))}
        inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for pattern in ['lab2_text/input/*','lab3_rag/data/*','lab3_rag/gold.json','lab4_agent/data/*'] for p in ROOT.glob(pattern) if p.is_file()}
        self.config={'model':get_model(),'thinking':'disabled','code_sha256':digest(files),'data_sha256':digest(inputs),'run_kind':'offline_test' if self.mock else 'llm_api', **(config or {})}
        self.meta=self.output/'run.json'
        if self.meta.exists():
            old=json.loads(self.meta.read_text())
            if old['config']!=self.config: raise ValueError('Настройки изменились. Выбери новую папку output для нового эксперимента.')
            self.run_id=old['run_id']
        else:
            self.run_id=str(uuid.uuid4());dump(self.meta,{'run_id':self.run_id,'started':now(),'config':self.config,'status':'prepared'})
        self.trace=self.output/'api_trace.jsonl'
    def request(self, request, stage):
        with LOCK:
            if self.raw is None:self.raw=make_raw_client()
        token=self.budget.reserve(request)
        started=time.perf_counter()
        try:r=self.raw.chat.completions.create(**request,**api_options())
        except Exception as exc:
            append(self.trace,{'run_id':self.run_id,'stage':stage,'ts':now(),'status':'api_error','error':safe_error(exc)})
            # Keep reservation: a timeout may have incurred a charge server-side.
            raise
        usage=r.usage.model_dump() if hasattr(r.usage,'model_dump') else vars(r.usage)
        self.budget.settle(token,usage)
        append(self.trace,{'run_id':self.run_id,'stage':stage,'ts':now(),'status':'response','response_id':r.id,
            'model':getattr(r,'model',get_model()),'usage':usage,'cost_upper_usd':cost(usage),'seconds':time.perf_counter()-started,
            'message':r.choices[0].message.model_dump(exclude_none=True) if hasattr(r.choices[0].message,'model_dump') else vars(r.choices[0].message)})
        return r
    def json(self, stage, messages, response_model, max_retries=3, temperature=0.2, max_tokens=2048):
        schema=response_model.model_json_schema()
        msgs=[dict(m) for m in messages]
        msgs.insert(0,{'role':'system','content':'Верни один JSON-объект без Markdown, точно по JSON-схеме: '+json.dumps(schema,ensure_ascii=False)})
        cache=self.output/'cache'/(digest([stage,msgs,get_model(),temperature,max_tokens]) + '.json')
        if cache.exists(): return response_model.model_validate_json(cache.read_text())
        for attempt in range(max_retries+1):
            r=self.request({'model':get_model(),'messages':msgs,'temperature':temperature,'max_tokens':max_tokens,'response_format':{'type':'json_object'}},stage)
            raw=r.choices[0].message.content or ''
            try:
                if r.choices[0].finish_reason=='length': raise ValueError('JSON truncated at max_tokens')
                value=response_model.model_validate_json(raw)
            except (ValidationError,ValueError) as exc:
                errors=exc.errors(include_url=False,include_context=False,include_input=False) if isinstance(exc,ValidationError) else [{'msg':str(exc)}]
                append(self.trace,{'run_id':self.run_id,'stage':stage,'status':'validation_error','attempt':attempt+1,'errors':errors,'ts':now()})
                if attempt==max_retries:raise
                msgs.extend([{'role':'assistant','content':raw},{'role':'user','content':'Исправь JSON. Ошибки: '+json.dumps(errors,ensure_ascii=False)}])
            else:
                dump(cache,value.model_dump());return value
    def stats(self):
        rows=records(self.trace);responses=[r for r in rows if r['status']=='response']
        return {'run_kind':self.config['run_kind'],'api_calls':len(responses),'validation_errors':sum(r['status']=='validation_error' for r in rows),
          'prompt_tokens':sum(r['usage'].get('prompt_tokens',0) or 0 for r in responses),
          'completion_tokens':sum(r['usage'].get('completion_tokens',0) or 0 for r in responses),
          'cost_upper_usd':sum(r['cost_upper_usd'] for r in responses),'api_seconds_sum':sum(r['seconds'] for r in responses)}
    def finish(self, **extra):
        meta=json.loads(self.meta.read_text());meta.update(status='completed',finished=now(),stats=self.stats(),**extra);dump(self.meta,meta)

def make_client(run, stage):
    """Course-compatible make_client()/response_model/max_retries interface."""
    def create(*, model=None, messages, response_model, max_retries=3, **kwargs):
        return run.json(stage,messages,response_model,max_retries,**kwargs)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

def ask(run, stage, model, system, obj, **kw):
    return make_client(run,stage).chat.completions.create(model=get_model(),response_model=model,max_retries=3,
        messages=[{'role':'system','content':system},{'role':'user','content':json.dumps(obj,ensure_ascii=False,default=str)}],**kw)
