"""Проверить 90 запусков, 100 независимых испытаний и 12 замеров без API."""
import argparse
import os
from pathlib import Path
from llm_client import read, events
from schemas_pwc import Plan, Verdict, FinalAnswer
from eval_pwc import CASES, CONFIGS, check_case
from critic_cases import broken_cases
from suite_common import verify_api, verify_agent, api_totals
from run_lab import summaries

def verify_pwc(result, caches):
    by_stage = {c['stage']: c for c in caches.values()}
    for e in result.get('trace', []):
        if e.get('kind') in ('plan', 'verdict'):
            assert by_stage[e['api_stage']]['value'] == e['value'], 'Изменён план/вердикт'
        elif e.get('kind') == 'synthesis':
            assert by_stage[e['api_stage']]['value']['answer'] == e['answer'] == result['answer']
        elif e.get('kind') == 'worker':
            a = e['value']
            if a['agent_result']:
                verify_agent(a['agent_result'], caches)
                assert a['raw_trace'] == a['agent_result']['trace']
                assert a['used_tools'] == a['agent_result']['used_tools']
                expected = a['agent_result']['answer'] or '(ошибка: ' + str(a['agent_result']['error']) + ')'
                assert a['answer'] == expected
    plans = [e['value'] for e in result.get('trace', []) if e.get('kind') == 'plan']
    if result.get('answer') and result.get('plan', {}).get('subquestions') == []:
        assert plans and result['plan'] == plans[-1]
        assert result['answer'] == 'Недостаточно данных: ' + plans[-1]['reasoning']

def verify(output='output', require_api=True):
    output = Path(output)
    meta, caches = verify_api(output, require_api, {'Plan': Plan, 'Verdict': Verdict, 'FinalAnswer': FinalAnswer})
    os.environ['MACRO_DATA_MODE'] = meta['config']['data_mode']
    results = read(output / 'eval_results.json')
    expected = {(c['id'], config, rep) for c in CASES for config in CONFIGS for rep in range(1, 6)}
    assert len(results) == 90 and {(r['qid'], r['config'], r['rep']) for r in results} == expected
    lookup = {c['id']: c for c in CASES}
    for r in results:
        result = r['result']
        assert result['query'] == lookup[r['qid']]['query']
        if result.get('error_type') != 'model_output':
            if r['config'] == 'single':
                verify_agent(result, caches)
            else:
                verify_pwc(result, caches)
        else:
            assert any(e['stage'].startswith(result['stage'] + '/') and e['status'] == 'validation_error'
                       for e in events(output / 'generation/api_trace.jsonl'))
        assert r['evaluation'] == check_case(lookup[r['qid']], result)
    trials = read(output / 'critic_trials.json')
    expected = {(c['id'], t, i) for c in broken_cases() for t in (0.0, 0.7) for i in range(1, 11)}
    assert len(trials) == 100 and {(t['case'], t['temperature'], t['rep']) for t in trials} == expected
    trace = events(output / 'generation/api_trace.jsonl')
    for trial in trials:
        calls = [e for e in trace if e['stage'] == trial['stage'] and e.get('response_id')]
        assert len(calls) == 1, 'Испытание должно иметь один отдельный ответ, без повторов JSON'
        event = calls[0]
        assert event['request']['temperature'] == trial['temperature']
        import json
        payload = json.loads(event['request']['messages'][-1]['content'])
        assert all(set(a) == {'answer', 'used_tools'} for a in payload['answers'].values()), 'Критику передана трасса'
        if trial['verdict'] is not None:
            assert caches[event['cache_key']]['value'] == trial['verdict']
            assert trial['false_accept'] == trial['verdict']['ok']
            assert trial['consistent'] == Verdict.model_validate(trial['verdict']).consistent()
        else:
            assert event['status'] == 'validation_error'
    timings = read(output / 'timings.json')
    assert len(timings) == 12
    assert {(x['qid'], x['parallel'], x['pair']) for x in timings} == {(q, p, n) for q in ('Q1', 'Q5') for p in (False, True) for n in (1, 2, 3)}
    nonces = {r['nonce'] for r in timings}
    assert len(nonces) == 12
    for r in timings:
        assert r['seconds'] > 0 and r['cache_replays'] == 0
        for a in r['answers'].values():
            verify_agent(a['agent_result'], caches)
            assert all(not e.get('cache_replay') for e in a['raw_trace'])
    metrics = read(output / 'metrics.json')
    assert metrics == {'run_kind': meta['config']['run_kind'], **summaries(results, trials, timings), **api_totals(output)}
    control = read(output / 'validator_control/control_results.json')
    assert control['run_kind'] == 'controlled_test' and control['api_calls'] == 0
    assert control['single_failed'] and control['pwc_failed'] and control['validated_passed']
    return {'verified': True, 'run_kind': meta['config']['run_kind'], 'eval_runs': 90, 'critic_trials': 100,
            'timing_runs': 12, 'limits': 'Согласованность файлов и критериев; не биллинг и не полная независимая оценка смысла.'}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='output')
    p.add_argument('--allow-test', action='store_true')
    a = p.parse_args()
    import json
    print(json.dumps(verify(a.output, not a.allow_test), ensure_ascii=False, indent=2))
