"""Локальные тесты. Все ответы здесь — заглушки, ни одного API-запроса."""

import contextlib
import copy
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from analysis import analyze_run, compare_runs
from generator import make_schedule, run_generation
from llm_client import JsonClient, explain_error, make_client
from schema import Application, CITIES, COURSES, SPECIALITIES, flatten, from_flat


def fixture(city="Казань", speciality="Инженер"):
    return dict(full_name="Тестов Иван Тестович", age=35,
                address={"city": city, "district": "Тестовый район"},
                speciality=speciality, desired_course="Анализ данных",
                years_of_experience=10, graduation_year=2013)


def response(data, index=1):
    return SimpleNamespace(
        id=f"offline-fixture-{index}",
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=80),
    )


class ScriptedRaw:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        item = next(self.outputs)
        if isinstance(item, Exception):
            raise item
        return response(item, len(self.calls))

    def close(self):
        pass


class FixtureRaw:
    def __init__(self, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise ConnectionError("Simulated interruption: no network was used")
        self.calls += 1
        text = "\n".join(m["content"] for m in kwargs["messages"] if m["role"] == "user")
        city = re.search(r"seed_city = ([^.]+)\.", text).group(1)
        return response(fixture(city, SPECIALITIES[(self.calls - 1) % 8]), self.calls)

    def close(self):
        pass


class SchemaTests(unittest.TestCase):
    def test_valid_csv_round_trip(self):
        value = Application.model_validate(fixture())
        csv_row = {k: str(v) for k, v in flatten(value).items()}
        self.assertEqual(from_flat(csv_row), value)
        self.assertNotIn("address", csv_row)
        self.assertIn("district", csv_row)

    def test_field_validator_business_errors(self):
        for patch_values, code in [
            ({"years_of_experience": 30}, "experience_exceeds_age"),
            ({"graduation_year": 1985}, "graduation_too_early"),
            ({"full_name": "Иван"}, "string_too_short"),
            ({"full_name": "Иван Петров"}, "invalid_full_name"),
        ]:
            with self.subTest(code=code):
                with self.assertRaises(ValidationError) as ctx:
                    Application.model_validate({**fixture(), **patch_values})
                self.assertIn(code, {e["type"] for e in ctx.exception.errors()})

    def test_invalid_city(self):
        value = fixture("Неизвестный город")
        with self.assertRaises(ValidationError) as ctx:
            Application.model_validate(value)
        self.assertEqual(ctx.exception.errors()[0]["type"], "city_not_allowed")

    def test_ranges_types_and_literals(self):
        for key, value in [("age", 21), ("age", 66), ("age", "35"), ("age", True),
                           ("graduation_year", 2025), ("years_of_experience", -1),
                           ("years_of_experience", 41), ("speciality", "Маг"),
                           ("desired_course", "Курс, которого нет")]:
            with self.subTest(key=key, value=value), self.assertRaises(ValidationError):
                Application.model_validate({**fixture(), key: value})

    def test_schedules(self):
        random_schedule = make_schedule("random", 42)
        self.assertEqual(len(random_schedule), 50)
        self.assertEqual(random_schedule, make_schedule("random", 42))
        self.assertTrue(set(random_schedule).issubset(CITIES))
        stratified = make_schedule("stratified", 42)
        self.assertTrue(all(stratified.count(city) == 5 for city in CITIES))
        self.assertNotEqual(stratified, make_schedule("stratified", 43))


class ClientTests(unittest.TestCase):
    def test_auth_error_diagnostic_does_not_include_sensitive_response(self):
        error = RuntimeError("secret error body must never be shown")
        error.status_code = 401
        message = explain_error(error)
        self.assertIn("401", message)
        self.assertIn("действующий ключ", message)
        self.assertNotIn("secret", message)

    def test_retry_and_audit_custom_validation(self):
        invalid = {**fixture(), "years_of_experience": 30}
        raw = ScriptedRaw([invalid, fixture()])
        events = []
        client = JsonClient(raw, events.append)
        messages = [{"role": "user", "content": "Создай заявку в JSON"}]
        result = client.chat.completions.create(
            model="offline", messages=messages, response_model=Application,
            max_retries=3, max_tokens=777,
        )
        self.assertEqual(result.years_of_experience, 10)
        self.assertEqual([e["status"] for e in events], ["validation_error", "valid"])
        self.assertEqual(events[0]["errors"][0]["type"], "experience_exceeds_age")
        self.assertEqual(raw.calls[0]["max_tokens"], 777)
        self.assertEqual(len(messages), 1, "Входной промпт не должен мутировать")
        self.assertIn("Исправь ошибки", raw.calls[1]["messages"][-1]["content"])

    def test_retry_budget_is_four_calls(self):
        raw = ScriptedRaw(["не JSON"] * 5)
        events = []
        client = JsonClient(raw, events.append)
        with self.assertRaises(ValueError):
            client.chat.completions.create(model="offline", messages=[],
                                           response_model=Application, max_retries=3)
        self.assertEqual(len(raw.calls), 4)
        self.assertEqual(len(events), 4)

    def test_api_error_is_not_retried_or_logged_as_validation(self):
        raw = ScriptedRaw([ConnectionError("a sensitive server error"), fixture()])
        events = []
        with self.assertRaises(ConnectionError):
            JsonClient(raw, events.append).chat.completions.create(
                model="offline", messages=[], response_model=Application, max_retries=3)
        self.assertEqual(len(raw.calls), 1)
        self.assertEqual(events[0]["status"], "api_error")
        self.assertNotIn("sensitive", json.dumps(events))

    def test_no_key_does_not_create_network_client(self):
        with patch.dict("os.environ", {"LLM_AUTH_TOKEN": ""}), patch("llm_client.OpenAI") as sdk:
            with self.assertRaises(RuntimeError):
                make_client()
            sdk.assert_not_called()


class PipelineTests(unittest.TestCase):
    def factory(self, **kwargs):
        return JsonClient(FixtureRaw(), kwargs["on_attempt"])

    def test_end_to_end_comparison_and_honest_test_label(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            for strategy in ("random", "stratified"):
                run_generation(root / strategy, strategy=strategy, client_factory=self.factory,
                               run_kind="offline_test")
            result = compare_runs(root / "random", root / "stratified")
            self.assertEqual(result["count"], 50)
            self.assertEqual(result["max_city_share"], 0.1)
            self.assertTrue(result["meets_good_criteria"])
            self.assertEqual(result["api_calls"], 50)
            for file in ("applications.csv", "cities.png", "specialities.png", "выводы.md", "comparison.md"):
                self.assertTrue((root / "stratified" / file).is_file())
            conclusions = (root / "stratified" / "выводы.md").read_text(encoding="utf-8")
            self.assertIn("ТЕСТОВЫЙ ПРОГОН БЕЗ LLM", conclusions)
            self.assertIn("не обнаружили", conclusions)
            self.assertIn("При случайном выборе", conclusions)

    def test_interruption_resume_preserves_existing_records(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            def failing_factory(**kwargs):
                return JsonClient(FixtureRaw(fail_after=3), kwargs["on_attempt"])
            with self.assertRaises(ConnectionError):
                run_generation(root, client_factory=failing_factory, run_kind="offline_test")
            before = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(len(before), 3)
            self.assertFalse((root / "applications.csv").exists())
            result = run_generation(root, resume=True, client_factory=self.factory, run_kind="offline_test")
            after = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(after[:3], before)
            self.assertEqual(len(after), 50)
            self.assertEqual(result["api_errors"], 1)
            self.assertEqual(result["api_calls"], 51)

    def test_resume_complete_does_not_call_api(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            run_generation(root, client_factory=self.factory, run_kind="offline_test")
            with patch("llm_client.OpenAI") as sdk:
                run_generation(root, resume=True, run_kind="offline_test")
                sdk.assert_not_called()

    def test_analysis_refuses_partial_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run.json").write_text(json.dumps({"status": "running"}))
            (root / "checkpoint.json").write_text(json.dumps([fixture()]))
            with self.assertRaises(ValueError):
                analyze_run(root)
            self.assertFalse((root / "applications.csv").exists())

    def test_resume_refuses_changed_seed(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            run_generation(root, client_factory=self.factory, run_kind="offline_test")
            with self.assertRaises(RuntimeError):
                run_generation(root, seed=43, resume=True, client_factory=self.factory,
                               run_kind="offline_test")


if __name__ == "__main__":
    unittest.main()
