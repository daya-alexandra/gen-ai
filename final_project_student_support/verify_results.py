"""Независимая проверка происхождения и согласованности сохранённого результата."""
import argparse
import hashlib
import json
from pathlib import Path

from pipeline import validate_resolution
from runtime import digest, events, file_hashes, read
from schemas import JudgeVerdict, Plan, RequestExtraction, Resolution


def verify(output="output", require_api=True):
    output = Path(output)
    meta = read(output / "run.json")
    assert meta["status"] == "completed", "Прогон не завершён"
    assert meta["config"]["files_sha256"] == file_hashes(), "Код или входные данные изменены после прогона"
    if require_api:
        assert meta["config"]["run_kind"] == "llm_api", "Это локальный тест, а не API-прогон"

    trace = [item for item in events(output / "api_trace.jsonl") if item.get("response_id")]
    assert 80 <= len(trace) <= 160, f"Ожидалось 80–160 ответов API, получено {len(trace)}"
    ids = [item["response_id"] for item in trace]
    assert len(ids) == len(set(ids)), "response_id должны быть уникальными"
    assert all(item["request_sha256"] == digest(item["request"]) for item in trace)
    by_cache = {item["cache_key"]: item for item in trace if item["status"] == "api_response"}
    cache_files = list((output / "cache").glob("*.json"))
    assert len(cache_files) == len(trace)
    for path in cache_files:
        cached = read(path)
        assert path.stem in by_cache
        assert cached["response_id"] == by_cache[path.stem]["response_id"]
        assert cached["value"]["text"] == by_cache[path.stem]["text"]

    results = read(output / "results.json")
    rows = read(output / "evaluation.json")
    metrics = read(output / "metrics.json")
    diagnostics = read(output / "diagnostics.json")
    assert len(results) == len(rows) == metrics["cases"] == 20
    assert metrics["api_responses"] == len(trace)
    assert metrics["ghost_citations"] == 0
    assert len(diagnostics) == 3 and all(item["blocked"] for item in diagnostics)

    for result in results:
        RequestExtraction.model_validate(result["extraction"])
        Plan.model_validate(result["plan"])
        resolution = Resolution.model_validate(result["resolution"])
        JudgeVerdict.model_validate(result["judge"])
        validate_resolution(resolution, result["request"], result["observations"])

    manifest = read(output / "artifact_manifest.json")
    for relative, expected in manifest["files"].items():
        path = output / relative
        assert path.exists(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
    for name in ["report.md", "results.json", "evaluation.json", "pipeline_trace.jsonl"]:
        assert (output / name).exists() and (output / name).stat().st_size > 0, name

    return {
        "verified": True,
        "run_kind": meta["config"]["run_kind"],
        "cases": len(results),
        "api_responses": len(trace),
        "passed": metrics["passed"],
        "ghost_citations": metrics["ghost_citations"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output")
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.output, not args.allow_test), ensure_ascii=False, indent=2))
