"""Один и тот же ReAct-цикл lab4; в С6 доступны исходные четыре инструмента."""
from agent import run_agent as _run_agent
from schemas_pwc import VALID_TOOLS

def run_agent(question, *, run, stage, **kwargs):
    return _run_agent(question, run=run, stage=stage, allowed=VALID_TOOLS, **kwargs)

if __name__ == '__main__':
    import argparse, uuid
    from pathlib import Path
    from suite_common import make_run
    from llm_client import dump
    p = argparse.ArgumentParser()
    p.add_argument('query')
    p.add_argument('--output', default='interactive_single')
    a = p.parse_args()
    r = make_run(a.output, 'interactive', {})
    try:
        result = run_agent(a.query, run=r, stage=uuid.uuid4().hex)
        print(result['answer'] or result['error'])
        dump(Path(a.output) / 'last_answer.json', result)
        r.finish('completed')
    finally:
        r.close()
