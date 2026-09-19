import json
from concurrent.futures import ThreadPoolExecutor
from agent import run_agent
from schemas_pwc import WorkerAnswer, VALID_TOOLS

def worker(sq, prev_answers, *, run, stage, feedback=None):
    if not set(sq.expected_tools) <= VALID_TOOLS or not sq.expected_tools:
        return WorkerAnswer(subquestion_id=sq.id, question_snippet=sq.question[:80],
                            answer='(ошибка: unknown_tool в ожидаемых инструментах)', used_tools=[])
    missing = set(sq.depends_on) - prev_answers.keys()
    if missing:
        raise ValueError('Ответы зависимостей отсутствуют: ' + str(sorted(missing)))
    context = json.dumps({i: {'answer': prev_answers[i].answer, 'used_tools': prev_answers[i].used_tools}
                          for i in sq.depends_on}, ensure_ascii=False)
    result = run_agent(sq.question, run=run, stage=stage, allowed=sq.expected_tools,
                       prev_context=context if sq.depends_on else None, feedback=feedback, max_iter=6)
    return WorkerAnswer(subquestion_id=sq.id, question_snippet=sq.question[:80],
                        answer=result['answer'] or '(ошибка: ' + str(result['error']) + ')',
                        used_tools=result['used_tools'], raw_trace=result['trace'], agent_result=result)

def execute_level(level, prev_answers, *, run=None, stage='workers', parallel=True, feedback=None, worker_fn=worker):
    """Только снимок предыдущих уровней; результаты соседей недоступны."""
    snapshot = dict(prev_answers)
    def one(sq):
        deps = {i: snapshot[i] for i in sq.depends_on}
        return worker_fn(sq, deps, run=run, stage=f'{stage}/sq{sq.id}', feedback=feedback)
    if parallel and len(level) > 1:
        with ThreadPoolExecutor(max_workers=min(3, len(level))) as pool:
            futures = {sq.id: pool.submit(one, sq) for sq in level}
            return {i: future.result() for i, future in futures.items()}
    return {sq.id: one(sq) for sq in level}
