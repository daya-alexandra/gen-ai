"""Инструменты на неизменном учебном снимке; live FX включается явно."""
from __future__ import annotations
import ast
import calendar
import csv
import math
import operator
import os
import re
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

DATA_DIR = Path(__file__).resolve().parent / 'data'
AS_OF = '2026-04-22'


def rows(name):
    with (DATA_DIR / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def parse_date(value=None):
    result = date.fromisoformat(value or AS_OF)
    if os.getenv('MACRO_DATA_MODE', 'snapshot') == 'snapshot' and result > date.fromisoformat(AS_OF):
        raise ValueError('Дата позже фиксированного учебного снимка ' + AS_OF)
    return result


def get_fx_rate(currency='USD', on_date=None):
    d = parse_date(on_date)
    currency = currency.upper()
    if not re.fullmatch('[A-Z]{3}', currency):
        raise ValueError('Код валюты: три латинские буквы')
    reason = 'Детерминированный учебный снимок, не текущая котировка'
    if os.getenv('MACRO_DATA_MODE', 'snapshot') == 'live':
        try:
            url = 'https://www.cbr.ru/scripts/XML_daily.asp?' + urlencode({'date_req': d.strftime('%d/%m/%Y')})
            with urlopen(Request(url, headers={'User-Agent': 'GenAI-course-lab'}), timeout=8) as response:
                root = ET.fromstring(response.read())
            actual = date.fromisoformat('-'.join(reversed(root.attrib['Date'].split('.'))))
            if actual > d:
                raise ValueError('Источник вернул будущую дату')
            for node in root.findall('Valute'):
                if node.findtext('CharCode') == currency:
                    value = float(node.findtext('Value').replace(',', '.')) / int(node.findtext('Nominal'))
                    return {'currency': currency, 'date': actual.isoformat(), 'requested_date': d.isoformat(),
                            'rate': value, 'value': value, 'unit': 'RUB/' + currency, 'source': 'cbr_live',
                            'fixture': False, 'synthetic': False}
            raise ValueError('Валюта отсутствует')
        except Exception as exc:
            reason = 'Живой запрос недоступен: ' + type(exc).__name__
    eligible = [r for r in rows('fx_benchmark.csv') if r['currency'] == currency and date.fromisoformat(r['date']) <= d]
    if not eligible:
        return {'error': 'Нет записи на дату или раньше', 'source': 'fallback_csv'}
    r = max(eligible, key=lambda x: x['date'])
    value = float(r['rate'])
    return {'currency': currency, 'date': r['date'], 'requested_date': d.isoformat(), 'rate': value,
            'value': value, 'unit': 'RUB/' + currency, 'source': 'fallback_csv', 'fixture': True,
            'synthetic': 'mock' in r['note'], 'stale_days': (d - date.fromisoformat(r['date'])).days,
            'reason': reason, 'note': r['note']}


def get_key_rate(on_date=None):
    d = parse_date(on_date)
    candidates = [r for r in rows('key_rate_history.csv') if date.fromisoformat(r['valid_from']) <= d]
    if not candidates:
        return {'error': 'Нет ставки на дату или раньше', 'source': 'fallback_csv'}
    r = max(candidates, key=lambda x: x['valid_from'])
    return {'date': d.isoformat(), 'valid_from': r['valid_from'], 'rate': float(r['rate']),
            'value': float(r['rate']), 'unit': '% годовых', 'source': 'fallback_csv', 'fixture': True,
            'synthetic': 'mock' in r['note'], 'note': r['note']}


def monthly(filename, column, year=None, month=None):
    if (year is None) != (month is None):
        raise ValueError('Укажи вместе year и month либо пропусти оба')
    data = rows(filename)
    if year is None:
        eligible = [r for r in data if (int(r['year']), int(r['month'])) <= (2026, 4)]
        r = max(eligible, key=lambda x: (int(x['year']), int(x['month'])))
    else:
        if isinstance(year, bool) or isinstance(month, bool) or not 1 <= month <= 12:
            raise ValueError('Неверный месяц')
        r = next((r for r in data if int(r['year']) == year and int(r['month']) == month), None)
        if r is None:
            return {'error': f'Нет данных за {year}-{month:02d}', 'source': 'fallback_csv'}
    period = f"{int(r['year']):04d}-{int(r['month']):02d}"
    # mock_from_here относится и ко всем последующим строкам, не только к первой.
    cutoff = next((f"{int(x['year']):04d}-{int(x['month']):02d}" for x in data if 'mock_from_here' in x['note']), None)
    return {'year': int(r['year']), 'month': int(r['month']), 'date': period, column: float(r[column]),
            'value': float(r[column]), 'unit': '% г/г' if column == 'cpi_yoy' else '% рабочей силы',
            'source': 'fallback_csv', 'fixture': True, 'synthetic': bool(cutoff and period >= cutoff),
            'note': r['note'], 'frequency': 'year_over_year' if column == 'cpi_yoy' else 'monthly_level'}


def get_inflation(year=None, month=None):
    return monthly('cpi_ru_monthly.csv', 'cpi_yoy', year, month)


def get_unemployment(year=None, month=None):
    return monthly('unemployment_ru_monthly.csv', 'unemployment', year, month)


def calculate(expression):
    """Ограниченный AST: без eval, атрибутов, индексов и произвольного кода."""
    try:
        if not isinstance(expression, str) or len(expression) > 2000:
            raise ValueError('Слишком длинное выражение')
        tree = ast.parse(expression.replace('^', '**'), mode='eval')
        if len(list(ast.walk(tree))) > 250:
            raise ValueError('Слишком сложное выражение')
        funcs = {'sqrt': math.sqrt, 'log': math.log, 'ln': math.log, 'exp': math.exp, 'abs': abs}
        binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
        def visit(n):
            if isinstance(n, ast.Constant) and type(n.value) in (int, float):
                value = float(n.value)
            elif isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
                value = visit(n.operand) * (-1 if isinstance(n.op, ast.USub) else 1)
            elif isinstance(n, ast.BinOp):
                a, b = visit(n.left), visit(n.right)
                if isinstance(n.op, ast.Pow):
                    if abs(b) > 100 or abs(a) > 1e12:
                        raise ValueError('Степень вне лимита')
                    value = a ** b
                elif type(n.op) in binary:
                    value = binary[type(n.op)](a, b)
                else:
                    raise ValueError('Оператор запрещён')
            elif isinstance(n, ast.Name) and n.id in ('pi', 'e'):
                value = getattr(math, n.id)
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in funcs and len(n.args) == 1 and not n.keywords:
                arg = visit(n.args[0])
                if n.func.id == 'exp' and abs(arg) > 100:
                    raise ValueError('exp вне лимита')
                value = funcs[n.func.id](arg)
            else:
                raise ValueError('Конструкция запрещена')
            if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e100:
                raise ValueError('Нефинитный или слишком большой результат')
            return float(value)
        return {'expression': expression, 'result': visit(tree.body), 'source': 'local_calculator'}
    except (ValueError, SyntaxError, ArithmeticError, RecursionError, TypeError):
        return {'expression': str(expression)[:2000], 'error': 'Недопустимое арифметическое выражение', 'source': 'local_calculator'}


def period_date(period):
    if re.fullmatch(r'\d{4}-\d{2}', period):
        y, m = map(int, period.split('-'))
        d = date(y, m, calendar.monthrange(y, m)[1])
        if period == AS_OF[:7]:
            d = min(d, date.fromisoformat(AS_OF))
    else:
        d = date.fromisoformat(period)
    return parse_date(d.isoformat())


def compare_periods(metric, period_a, period_b):
    """Дельта b-a и отношение b/a через существующие функции получения данных."""
    def one(period):
        d = period_date(period)
        if metric == 'key_rate':
            return get_key_rate(d.isoformat())
        if metric in ('fx_USD', 'fx_EUR', 'fx_CNY'):
            return get_fx_rate(metric[3:], d.isoformat())
        if metric == 'cpi':
            return get_inflation(d.year, d.month)
        if metric == 'unemployment':
            return get_unemployment(d.year, d.month)
        raise ValueError('Неизвестная метрика')
    a, b = one(period_a), one(period_b)
    if 'error' in a or 'error' in b:
        return {'metric': metric, 'a': a, 'b': b, 'error': 'Недостаточно данных для сравнения', 'source': 'compare_periods'}
    va, vb = a['value'], b['value']
    return {'metric': metric, 'a': a, 'b': b, 'delta': vb - va, 'ratio': vb / va if va else None,
            'ratio_note': None if va else 'Деление на ноль не определено',
            'source': a['source'] + ' / ' + b['source'], 'fixture': a['fixture'] or b['fixture'],
            'synthetic': a['synthetic'] or b['synthetic'], 'period_rule': 'Последняя запись не позже конца месяца или указанной даты; не среднее за месяц',
            'delta_unit': 'процентные пункты' if metric in ('key_rate', 'cpi', 'unemployment') else a['unit']}
