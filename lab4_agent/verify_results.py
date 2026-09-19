"""Локальная проверка сохранённых ответов; API не вызывается."""
import argparse
import os
from pathlib import Path
from llm_client import read
from eval import CASES, check_case
from suite_common import verify_api, verify_agent, api_totals

def verify(output='output', require_api=True):
    output = Path(output)
    meta, caches = verify_api(output, require_api)
    os.environ['MACRO_DATA_MODE'] = meta['config']['data_mode']
    results = read(output / 'eval_results.json')
    assert [r['id'] for r in results] == [c['id'] for c in CASES]
    for case, row in zip(CASES, results):
        assert row['query'] == case['query']
        verify_agent(row['result'], caches)
        assert row['evaluation'] == check_case(case, row['result'])
    metrics = read(output / 'metrics.json')
    assert metrics['passed'] == sum(r['evaluation']['ok'] for r in results)
    for k, v in api_totals(output).items():
        assert metrics[k] == v
    diagnostic = read(output / 'diagnostics/diagnostics.json')
    assert diagnostic['run_kind'] == 'controlled_test' and diagnostic['api_calls'] == 0
    assert {x['type'] for x in diagnostic['cases']} == {'invalid_arguments_json', 'unknown_tool', 'tool_error'}
    for d in diagnostic['cases']:
        assert any(e.get('obs', {}).get('error_type') == d['type'] for e in d['result']['trace'])
    return {'verified': True, 'run_kind': meta['config']['run_kind'], 'questions': len(results),
            'passed': metrics['passed'], 'controlled_fault_types': 3,
            'limits': 'Проверяется согласованность артефактов и заданных критериев, не биллинг и не полная семантика.'}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='output')
    p.add_argument('--allow-test', action='store_true')
    a = p.parse_args()
    import json
    print(json.dumps(verify(a.output, not a.allow_test), ensure_ascii=False, indent=2))
