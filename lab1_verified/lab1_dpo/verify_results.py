"""Read-only verification of saved CSV, raw API answers, checkpoints and metrics.

No clients are created and no network calls are made. Original results are never
rewritten. Assertions are explicit so validation also runs under python -O.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from generator import code_fingerprint, make_schedule
from schema import Application, CITIES, COURSES, SPECIALITIES, from_flat

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_run(folder, strategy):
    meta = read_json(folder / "run.json")
    metrics = read_json(folder / "metrics.json")
    checkpoint = read_json(folder / "checkpoint.json")
    require(meta["status"] == "complete", f"{strategy}: incomplete run")
    require(meta["run_kind"] == "llm_api", f"{strategy}: not marked as an API run")
    require(meta["config"]["strategy"] == strategy, "Unexpected strategy")
    require(meta["config"]["code"] == code_fingerprint(), "Generation source hashes differ")
    with (folder / "applications.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        require(set(reader.fieldnames or []) == {
            "full_name", "age", "city", "district", "speciality", "desired_course",
            "years_of_experience", "graduation_year",
        }, "Incorrect CSV columns")
        apps = [from_flat(row) for row in reader]
    values = [a.model_dump() for a in apps]
    require(len(apps) == 50 == meta["accepted"], "Expected 50 accepted applications")
    require(values == checkpoint, "CSV and checkpoint differ")
    trace = [json.loads(line) for line in (folder / "trace.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    started = datetime.fromisoformat(meta["started_at"])
    finished = datetime.fromisoformat(meta["finished_at"])
    previous = started
    for event in trace:
        current = datetime.fromisoformat(event["ts"])
        require(event["run_id"] == meta["run_id"], "Mismatched run ID")
        require(previous <= current <= finished, "Invalid event order/time")
        previous = current
    attempts = [e for e in trace if e["event"] == "llm_attempt"]
    accepted = [e for e in trace if e["event"] == "record_accepted"]
    # This supplied experiment completed every record at the first attempt.
    require(len(attempts) == len(accepted) == 50 and len(trace) == 100,
            "Unexpected event count; review retries/errors separately")
    require([e["record"] for e in accepted] == list(range(1, 51)), "Missing/repeated accepted records")
    schedule = make_schedule(strategy, meta["config"]["seed"])
    ids = []
    for i, (app, attempt, event) in enumerate(zip(apps, attempts, accepted)):
        require(attempt["record"] == i + 1 and attempt["status"] == "valid", "Invalid attempt")
        require(attempt["attempt"] == 1 and attempt["resample"] == 0, "Unexpected retry")
        require(attempt["model"] == meta["config"]["model"], "Mismatched requested model")
        require(attempt["seed_city"] == schedule[i] == app.address.city, "City schedule mismatch")
        require(event["application"] == app.model_dump(), "Accepted event differs from CSV")
        raw = Application.model_validate_json(attempt["raw_response"])
        require(raw == app, "Raw model answer differs from CSV")
        require(attempt.get("response_id") and not attempt["response_id"].startswith("offline-"), "Missing API response ID")
        ids.append(attempt["response_id"])
        require(all(isinstance(attempt[k], int) and attempt[k] > 0 for k in ("prompt_tokens", "completion_tokens")), "Missing usage")
    require(len(set(ids)) == 50, "Repeated response IDs")
    cities = Counter(a.address.city for a in apps)
    specialities = Counter(a.speciality for a in apps)
    names = Counter(a.full_name for a in apps)
    ages = Counter(a.age for a in apps)
    calculated = {
        "run_kind": meta["run_kind"], "model": meta["config"]["model"],
        "strategy": strategy, "count": len(apps),
        "city_counts": {c: cities[c] for c in CITIES},
        "speciality_counts": {s: specialities[s] for s in SPECIALITIES},
        "max_city_share": max(cities.values()) / 50,
        "max_speciality_share": max(specialities.values()) / 50,
        "specialities_present": len(specialities),
        "duplicate_names": sum(v - 1 for v in names.values()),
        "validation_error_events": 0, "custom_validator_error_events": 0,
        "custom_error_types": {}, "seed_mismatches": 0, "api_calls": 50,
        "api_errors": 0,
        "prompt_tokens_observed": sum(e["prompt_tokens"] for e in attempts),
        "completion_tokens_observed": sum(e["completion_tokens"] for e in attempts),
        "responses_without_usage": 0, "execution_seconds": meta["execution_seconds"],
        "meets_good_criteria": max(cities.values()) / 50 <= .40 and max(specialities.values()) / 50 <= .35,
    }
    require(calculated == metrics, "Saved metrics differ from recomputed values")
    require(calculated["meets_good_criteria"], "Assignment thresholds failed")
    if strategy == "stratified":
        require(all(cities[c] == 5 for c in CITIES), "Incorrect city quotas")
    return {
        "status": "passed", "run_id": meta["run_id"], "metrics": calculated,
        "checks": ["Pydantic schema", "CSV/checkpoint/raw-answer equality",
                   "source SHA-256", "city schedule", "event chronology",
                   "response ID uniqueness", "usage totals", "saved metrics"],
        "age_counts": {str(a): ages[a] for a in sorted(ages)},
        "courses": {c: sum(a.desired_course == c for a in apps) for c in COURSES},
        "duplicate_full_records": len(values) - len({json.dumps(v, sort_keys=True, ensure_ascii=False) for v in values}),
        "missing_specialities": [s for s in SPECIALITIES if not specialities[s]],
        "response_ids": ids,
        "original_result_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.iterdir()) if p.suffix in {".csv", ".json", ".jsonl"}},
    }


def verify(output):
    results = {s: verify_run(output / s, s) for s in ("random", "stratified")}
    ids = [i for r in results.values() for i in r.pop("response_ids")]
    require(len(ids) == len(set(ids)) == 100, "Response IDs repeat between runs")
    a = read_json(output / "random/run.json")["config"]
    b = read_json(output / "stratified/run.json")["config"]
    require({k: v for k, v in a.items() if k != "strategy"} ==
            {k: v for k, v in b.items() if k != "strategy"}, "Experiment configurations differ")
    return {
        "status": "passed", "scope": "offline audit of supplied experiment files",
        "api_calls_made_by_this_verification": 0, "runs": results,
        "limits": ["Provider billing and server-side authenticity are not independently verified.",
                   "District geography and profession/course plausibility are not validated by the schema.",
                   "Only one run per strategy; no causal/statistical significance claim."],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    print(json.dumps(verify(args.output), ensure_ascii=False, indent=2))
