import json
import os
import tempfile
import unittest
import uuid
from datetime import date
from pathlib import Path
from unittest.mock import patch

from pipeline import controlled_diagnostics, execute, load_inputs, validate_resolution
from rag import BM25Index
from runtime import ApiRun
from schemas import RequestExtraction, Resolution
from verify_results import verify


ROOT = Path(__file__).resolve().parent


class FakeTransport:
    def __init__(self):
        self.calls = 0
        cases, gold = load_inputs()
        self.cases = {item["id"]: item for item in cases}
        self.gold = gold

    def __call__(self, request, stage):
        self.calls += 1
        parts = stage.split("/")
        request_id, action = parts[1], parts[2]
        case = self.cases[request_id]
        expected = self.gold[request_id]
        if action == "extract":
            lo, hi = expected["urgency"]
            payload = {
                "request_id": request_id,
                "message_date": case["message_date"],
                "category": expected["category"],
                "intent": "Разобрать обращение и дать проверяемый маршрут решения",
                "urgency": (lo + hi) // 2,
                "contains_sensitive_data": expected["sensitive"],
                "requested_deadline": expected.get("requested_deadline"),
                "evidence": [{"quote": case["message"], "meaning": "Исходная формулировка запроса"}],
            }
        elif action == "plan":
            tools = [
                {"name": "search_rules", "arguments": {"query": case["message"]}},
                {"name": "get_contact", "arguments": {"category": expected["category"]}},
            ]
            if "calculate_days" in expected["required_tools"]:
                start, end = expected["date_range"]
                tools.append({"name": "calculate_days", "arguments": {"start": start, "end": end}})
            payload = {
                "route": expected["category"],
                "short_rationale": "Сначала найти правило, затем дать проверяемый контакт",
                "tools": tools,
            }
        elif action == "respond":
            context = json.loads(request["messages"][-1]["content"])
            observations = context["observations"]
            chunks = next(item["result"] for item in observations if item["name"] == "search_rules")
            chunk = chunks[0]
            contact = next(item["result"] for item in observations if item["name"] == "get_contact")
            payload = {
                "answer": (
                    f"По найденному учебному правилу обратитесь в {contact['unit']} через {contact['channel']}. "
                    "База правил синтетическая, поэтому официальный ответ нужно подтвердить в указанном канале."
                ),
                "message_quotes": [case["message"]],
                "citations": [{"source_id": chunk["source_id"], "quote": chunk["text"][:180]}],
                "abstained": request_id == "R12",
                "escalation_needed": expected["sensitive"] or expected["urgency"][0] >= 4,
            }
        else:
            payload = {
                "correctness": 4,
                "groundedness": 4,
                "path_quality": 2,
                "approved": True,
                "problems": [],
                "short_reason": "Ответ опирается на найденный источник и корректный маршрут",
            }
        return {
            "id": "fake-" + str(uuid.uuid4()),
            "model": "deepseek-flash",
            "choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 60, "prompt_cache_hit_tokens": 0},
        }


class ProjectTests(unittest.TestCase):
    def test_deadline_field_validator(self):
        with self.assertRaises(ValueError):
            RequestExtraction.model_validate(
                {
                    "request_id": "R01",
                    "message_date": date(2026, 9, 19),
                    "category": "schedule",
                    "intent": "Уточнить дату экзамена",
                    "urgency": 3,
                    "contains_sensitive_data": False,
                    "requested_deadline": date(2026, 9, 18),
                    "evidence": [{"quote": "Дата экзамена", "meaning": "Срок"}],
                }
            )

    def test_rag_hits_expected_source_for_all_cases(self):
        cases, gold = load_inputs()
        index = BM25Index(ROOT / "input/knowledge_base")
        for case in cases:
            sources = {item["source_id"] for item in index.search(case["message"], 3)}
            self.assertTrue(sources.intersection(gold[case["id"]]["expected_sources"]), case["id"])

    def test_controlled_failures_are_blocked(self):
        index = BM25Index(ROOT / "input/knowledge_base")
        checks = controlled_diagnostics(index)
        self.assertEqual(len(checks), 3)
        self.assertTrue(all(item["blocked"] for item in checks))

    def test_ghost_citation_is_rejected(self):
        value = Resolution.model_validate(
            {
                "answer": "Используйте официальный учебный портал для обращения.",
                "message_quotes": ["Нужна справка"],
                "citations": [{"source_id": "documents", "quote": "Выдуманная цитата из документа"}],
                "abstained": False,
                "escalation_needed": False,
            }
        )
        with self.assertRaises(ValueError):
            validate_resolution(
                value,
                {"message": "Нужна справка"},
                [{"name": "search_rules", "result": [{"source_id": "documents", "text": "Настоящий текст"}]}],
            )

    def test_full_pipeline_and_verifier(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            fake = FakeTransport()
            result = execute(directory, transport=fake)
            self.assertEqual(fake.calls, 80)
            self.assertEqual(result["metrics"]["passed"], 20)
            checked = verify(directory, require_api=False)
            self.assertTrue(checked["verified"])

    def test_resume_uses_cache(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            fake = FakeTransport()
            execute(directory, transport=fake)
            execute(directory, transport=fake)
            self.assertEqual(fake.calls, 80)

    def test_budget_blocks_before_transport(self):
        with tempfile.TemporaryDirectory(dir=".") as directory, patch.dict(os.environ, {"LLM_MAX_CALLS": "0"}, clear=False):
            fake = FakeTransport()
            run = ApiRun(directory, transport=fake)
            with self.assertRaises(RuntimeError):
                run.call_json(
                    "case/R01/extract",
                    [{"role": "user", "content": "x"}],
                    RequestExtraction,
                    attempts=1,
                )
            self.assertEqual(fake.calls, 0)


if __name__ == "__main__":
    unittest.main()
