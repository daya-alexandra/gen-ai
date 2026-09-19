from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

class SubQuestion(StrictModel):
    id: int = Field(ge=1)
    question: str = Field(min_length=1)
    expected_tools: list[str]
    depends_on: list[int] = Field(default_factory=list)

class Plan(StrictModel):
    reasoning: str = Field(min_length=1)
    subquestions: list[SubQuestion] = Field(max_length=12)

class WorkerAnswer(StrictModel):
    subquestion_id: int
    question_snippet: str
    answer: str
    used_tools: list[str] = Field(default_factory=list)
    raw_trace: list[dict] = Field(default_factory=list)
    agent_result: dict = Field(default_factory=dict)

class Verdict(StrictModel):
    ok: bool
    reason: str = Field(min_length=1)
    action: Literal['accept', 'replan', 'rework']
    rework_ids: list[int] = Field(default_factory=list)

    def consistent(self):
        return (self.ok and self.action == 'accept' and not self.rework_ids) or (
            not self.ok and self.action == 'replan' and not self.rework_ids) or (
            not self.ok and self.action == 'rework' and bool(self.rework_ids))

class FinalAnswer(StrictModel):
    answer: str = Field(min_length=1)

VALID_TOOLS = {'get_fx_rate', 'get_key_rate', 'get_inflation', 'calculate'}

def _topological_levels(subqs):
    ids = [sq.id for sq in subqs]
    if len(ids) != len(set(ids)):
        raise ValueError('Повторяющиеся id')
    by_id = {sq.id: sq for sq in subqs}
    for sq in subqs:
        if not set(sq.depends_on) <= by_id.keys():
            raise ValueError('Несуществующая зависимость')
    left, done, levels = set(ids), set(), []
    while left:
        ready = sorted(i for i in left if set(by_id[i].depends_on) <= done)
        if not ready:
            raise ValueError('Цикл зависимостей')
        levels.append([by_id[i] for i in ready])
        done.update(ready)
        left.difference_update(ready)
    return levels

def validate_plan(plan: Plan) -> list[str]:
    errors = []
    for sq in plan.subquestions:
        missing = sorted(set(sq.expected_tools) - VALID_TOOLS)
        if missing:
            errors.append(f'{sq.id}: инструменты не существуют: {missing}')
        if not sq.expected_tools:
            errors.append(f'{sq.id}: отсутствуют инструменты')
        if len(sq.depends_on) != len(set(sq.depends_on)):
            errors.append(f'{sq.id}: повтор зависимости')
    try:
        _topological_levels(plan.subquestions)
    except ValueError as exc:
        errors.append(str(exc))
    return errors
