"""Проверка сохранённого результата без API-запросов."""
import argparse
import hashlib
from pathlib import Path

from runtime import digest, events, file_hashes, read


def verify(output="output", require_api=True):
    output = Path(output)
    meta = read(output / "run.json")
    assert meta["status"] == "completed", "Прогон не завершён"
    assert meta["config"]["files_sha256"] == file_hashes(), "Код или вход изменён"
    if require_api:
        assert meta["config"]["run_kind"] == "llm_api", "Это локальный тест, а не API-прогон"
    log = [item for item in events(output / "api_trace.jsonl") if item.get("response_id")]
    assert len(log) == 234, f"Ожидалось 234 ответа API, получено {len(log)}"
    assert len({item["response_id"] for item in log}) == len(log), "Повтор response_id"
    assert all(item["request_sha256"] == digest(item["request"]) for item in log)
    valid = {item["cache_key"]: item for item in log if item["status"] == "valid"}
    caches = list((output / "cache").glob("*.json"))
    assert len(caches) == 234
    for path in caches:
        cached = read(path)
        assert path.stem in valid
        assert cached["response_id"] == valid[path.stem]["response_id"]
        assert cached["value"]["text"] == valid[path.stem]["text"]
    metrics = read(output / "metrics.json")
    assert metrics["headline_count"] == 20 and metrics["headline_repeats"] == 10
    assert metrics["api_responses"] == 234
    assert len(read(output / "temperature.json")["t0"]) == 10
    assert len(read(output / "temperature.json")["t1"]) == 10
    for name in ["report.md", "headlines_scored.csv", "temperature_experiment.png", "headline_scores.png"]:
        assert (output / name).exists() and (output / name).stat().st_size > 0, name
    return {"verified": True, "run_kind": meta["config"]["run_kind"], "api_responses": len(log), "headline_scores": 200}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output")
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    import json

    print(json.dumps(verify(args.output, not args.allow_test), ensure_ascii=False, indent=2))
