"""Локальные инструменты агента: поиск, контакт и календарный интервал."""
from __future__ import annotations

from datetime import date

from rag import BM25Index


CONTACTS = {
    "schedule": {"unit": "учебный офис программы", "channel": "раздел «Обращения» учебного портала"},
    "documents": {"unit": "служба учебных документов", "channel": "раздел «Документы» учебного портала"},
    "tuition": {"unit": "финансовый отдел", "channel": "раздел «Оплата обучения» учебного портала"},
    "academic_leave": {"unit": "учебный офис программы", "channel": "защищённая форма «Академический отпуск»"},
    "technical": {"unit": "техническая поддержка", "channel": "форма «Сообщить о проблеме» в портале"},
}


def execute_tool(name: str, arguments: dict[str, str], index: BM25Index) -> dict:
    if name == "search_rules":
        query = arguments.get("query", "").strip()
        if not query:
            raise ValueError("search_rules требует непустой query")
        return {"name": name, "arguments": {"query": query}, "result": index.search(query, top_k=3)}
    if name == "get_contact":
        category = arguments.get("category", "")
        if category not in CONTACTS:
            raise ValueError("Неизвестная категория контакта")
        return {"name": name, "arguments": {"category": category}, "result": CONTACTS[category]}
    if name == "calculate_days":
        start = date.fromisoformat(arguments["start"])
        end = date.fromisoformat(arguments["end"])
        if end < start:
            raise ValueError("Конец интервала раньше начала")
        return {
            "name": name,
            "arguments": {"start": start.isoformat(), "end": end.isoformat()},
            "result": {"calendar_days": (end - start).days},
        }
    raise ValueError(f"Инструмент {name!r} не разрешён")
