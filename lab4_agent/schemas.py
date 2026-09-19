"""Шесть инструментов и строгая проверка фактических аргументов."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
import tools

class Args(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

class FxArgs(Args):
    currency: str = 'USD'
    on_date: str | None = None

class RateArgs(Args):
    on_date: str | None = None

class MonthArgs(Args):
    year: int | None = None
    month: int | None = Field(default=None, ge=1, le=12)

class CalcArgs(Args):
    expression: str = Field(min_length=1, max_length=2000)

class CompareArgs(Args):
    metric: Literal['key_rate', 'fx_USD', 'fx_EUR', 'fx_CNY', 'cpi', 'unemployment']
    period_a: str
    period_b: str

REGISTRY = {
    'get_fx_rate': (FxArgs, tools.get_fx_rate, 'Рубли за единицу валюты на YYYY-MM-DD. Учебный CSV: последняя запись НЕ ПОЗЖЕ даты; фактическая дата может отличаться.'),
    'get_key_rate': (RateArgs, tools.get_key_rate, 'Ключевая ставка в % годовых на YYYY-MM-DD. Без даты — снимок 2026-04-22. Источник учебный CSV, поздние значения синтетические.'),
    'get_inflation': (MonthArgs, tools.get_inflation, 'Инфляция ТОЛЬКО % год к году, НЕ месячный прирост. Без year/month — последний доступный месяц, 2026-03. Помесячного ИПЦ нет.'),
    'get_unemployment': (MonthArgs, tools.get_unemployment, 'Учебный уровень безработицы в % рабочей силы. Без year/month — последний доступный месяц.'),
    'calculate': (CalcArgs, tools.calculate, 'Вычислить арифметику: + - * / ** sqrt log ln exp abs; числа из полученных данных. Любое производное число считай здесь.'),
    'compare_periods': (CompareArgs, tools.compare_periods, 'Сравнить шесть разрешённых метрик в периодах YYYY-MM либо YYYY-MM-DD: delta=b-a, ratio=b/a. Месяц означает последнее доступное значение на конец месяца, НЕ среднее. cpi — г/г; delta ставок — п.п.'),
}
TOOL_SCHEMAS = [{'type': 'function', 'function': {'name': name, 'description': desc, 'parameters': model.model_json_schema()}}
                for name, (model, _, desc) in REGISTRY.items()]

def dispatch(name, args, allowed=None):
    from pydantic import ValidationError
    if name not in REGISTRY or (allowed is not None and name not in allowed):
        return {'error': 'Инструмент не существует или не разрешён', 'error_type': 'unknown_tool'}
    model, fn, _ = REGISTRY[name]
    try:
        parsed = model.model_validate(args)
    except ValidationError as exc:
        return {'error': 'Аргументы не соответствуют схеме', 'error_type': 'bad_arguments',
                'details': exc.errors(include_input=False, include_url=False, include_context=False)}
    try:
        result = fn(**parsed.model_dump())
        if 'error' in result:
            result.setdefault('error_type', 'tool_error')
        return result
    except (ValueError, TypeError, OSError) as exc:
        return {'error': str(exc), 'error_type': 'tool_error'}
