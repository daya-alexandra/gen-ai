import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from experiments import execute, score_from_text
from runtime import (
    ApiRun,
    LEGACY_RELEASE_FILES,
    TOKEN_LIMIT_FIX_FILES,
    TARIFF,
    compatible_legacy_config,
    digest,
    dump,
    file_hashes,
    read,
)
from verify_results import verify


class FakeTransport:
    def __init__(self):
        self.calls = 0
        self.requests = []

    def __call__(self, request, stage):
        self.calls += 1
        self.requests.append((stage, request))
        if stage.startswith("scorer/") or stage.startswith("prompt/golf/"):
            text = "5"
        elif stage.startswith("temperature/t1/"):
            text = f"вариант-{self.calls % 4}"
        else:
            text = "Учебный ответ модели."
        return {
            "id": "fake-" + str(uuid.uuid4()),
            "model": "deepseek-flash",
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "prompt_cache_hit_tokens": 0},
        }


class Lab1Tests(unittest.TestCase):
    def test_score_parser(self):
        self.assertEqual(score_from_text("10"), 10)
        self.assertEqual(score_from_text("Оценка: 7 из 10"), 7)
        self.assertEqual(score_from_text("семь"), -1)

    def test_full_pipeline_and_verifier(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            fake = FakeTransport()
            execute(directory, transport=fake)
            self.assertEqual(fake.calls, 234)
            checked = verify(directory, require_api=False)
            self.assertTrue(checked["verified"])
            self.assertEqual(read(Path(directory) / "metrics.json")["prompt_golf_clean"], 5)
            zero = next(request for stage, request in fake.requests if stage == "prompt/ladder/zero")
            role = next(request for stage, request in fake.requests if stage == "prompt/ladder/role")
            professor = next(request for stage, request in fake.requests if stage == "prompt/persona/professor")
            self.assertEqual(zero["max_tokens"], 800)
            self.assertEqual(role["max_tokens"], 800)
            self.assertEqual(professor["max_tokens"], 1200)

    def test_resume_uses_cache(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            fake = FakeTransport()
            execute(directory, transport=fake)
            execute(directory, transport=fake)
            self.assertEqual(fake.calls, 234)

    def test_budget_blocks_before_transport(self):
        with tempfile.TemporaryDirectory(dir=".") as directory, patch.dict(os.environ, {"LLM_MAX_CALLS": "0"}, clear=False):
            fake = FakeTransport()
            run = ApiRun(directory, transport=fake)
            with self.assertRaises(RuntimeError):
                run.call("blocked", [{"role": "user", "content": "x"}], 0)
            self.assertEqual(fake.calls, 0)

    def test_known_legacy_run_reuses_cache(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            output = root / "legacy"
            output.mkdir()
            fake = FakeTransport()
            current = {
                "suite": "lab1_api_basics",
                "model": "deepseek-flash",
                "base_url": "https://api.deepseek.com",
                "run_kind": "offline_test",
                "files_sha256": file_hashes(),
                "tariff": TARIFF,
            }
            previous = {**current, "files_sha256": LEGACY_RELEASE_FILES}
            dump(
                output / "run.json",
                {
                    "run_id": "legacy-test",
                    "created_at": "2026-09-20T00:00:00+00:00",
                    "updated_at": "2026-09-20T00:01:00+00:00",
                    "active_seconds": 1.0,
                    "status": "interrupted",
                    "config": previous,
                },
            )
            request = {
                "model": "deepseek-flash",
                "messages": [{"role": "user", "content": "cached"}],
                "temperature": 0,
                "max_tokens": 180,
            }
            cache_key = digest(["cached-stage", request, previous])
            value = {
                "stage": "cached-stage",
                "text": "Сохранённый ответ",
                "response_id": "legacy-response",
                "server_model": "deepseek-flash",
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "prompt_cache_hit_tokens": 0},
                "seconds": 0.1,
            }
            dump(output / "cache" / f"{cache_key}.json", {"stage": "cached-stage", "response_id": "legacy-response", "value": value})

            run = ApiRun(output, transport=fake)
            answer = run.call("cached-stage", request["messages"], 0, 180)
            run.finish("interrupted")
            run.close()

            self.assertEqual(fake.calls, 0)
            self.assertTrue(answer["cache_replay"])
            self.assertEqual(answer["cache_key"], cache_key)
            migrated = read(output / "run.json")
            self.assertEqual(migrated["config"], current)
            self.assertIn(previous, migrated["compatible_cache_configs"])

    def test_first_fix_is_accepted_for_resume(self):
        current = {
            "suite": "lab1_api_basics",
            "model": "deepseek-flash",
            "base_url": "https://api.deepseek.com",
            "run_kind": "llm_api",
            "files_sha256": file_hashes(),
            "tariff": TARIFF,
        }
        previous = {**current, "files_sha256": TOKEN_LIMIT_FIX_FILES}
        self.assertTrue(compatible_legacy_config(previous, current))


if __name__ == "__main__":
    unittest.main()
