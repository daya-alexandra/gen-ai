import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from experiments import execute, score_from_text
from runtime import ApiRun, read
from verify_results import verify


class FakeTransport:
    def __init__(self):
        self.calls = 0

    def __call__(self, request, stage):
        self.calls += 1
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


if __name__ == "__main__":
    unittest.main()
