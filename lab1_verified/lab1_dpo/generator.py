"""ДЗ № 1: настоящая LLM-генерация с сохранением прогресса и аудитом попыток."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from llm_client import explain_error, get_model, make_client
from prompts import build_messages
from schema import Application, CITIES, REFERENCE_YEAR, SPECIALITIES

COUNT = 50


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def make_schedule(strategy: str, seed: int) -> list[str]:
    rng = random.Random(seed)
    if strategy == "stratified":
        cities = list(CITIES) * (COUNT // len(CITIES))
        rng.shuffle(cities)
        return cities
    if strategy == "random":
        return [rng.choice(CITIES) for _ in range(COUNT)]
    raise ValueError("Неизвестная стратегия")


def code_fingerprint() -> dict:
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("schema.py", "prompts.py", "generator.py", "llm_client.py")}


def run_generation(output: Path, strategy="stratified", seed=42, temperature=0.8,
                   resume=False, client_factory=make_client, run_kind="llm_api") -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run.json"
    checkpoint_path = output / "checkpoint.json"
    model = get_model()
    config = dict(strategy=strategy, seed=seed, temperature=temperature,
                  model=model, count=COUNT, reference_year=REFERENCE_YEAR,
                  max_retries=3, max_tokens=1200, code=code_fingerprint())
    schedule = make_schedule(strategy, seed)
    if manifest_path.exists():
        if not resume:
            raise RuntimeError("Прогон уже существует: используй --resume или другую папку --output")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config"] != config or manifest["run_kind"] != run_kind:
            raise RuntimeError("Настройки/код изменились: нужен новый каталог прогона")
        records = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if len(records) > COUNT:
            raise ValueError("В checkpoint больше 50 заявок")
        applications = [Application.model_validate(row) for row in records]
        if any(app.address.city != schedule[i] for i, app in enumerate(applications)):
            raise ValueError("Города в checkpoint не соответствуют плану")
        if manifest["status"] == "complete" and len(applications) != COUNT:
            raise ValueError("Завершённый прогон содержит не 50 заявок")
    else:
        if checkpoint_path.exists() or (output / "trace.jsonl").exists():
            raise RuntimeError("В папке есть данные без run.json; выбери пустую папку")
        manifest = dict(run_id=str(uuid.uuid4()), run_kind=run_kind, config=config,
                        started_at=now(), status="created", source="LLM API" if run_kind == "llm_api"
                        else "OFFLINE TEST FIXTURES — не ответы модели")
        applications = []
        write_json(checkpoint_path, [])
        write_json(manifest_path, manifest)

    if len(applications) == COUNT:
        from analysis import analyze_run
        return analyze_run(output)

    current = {}

    def log(event: dict) -> None:
        row = {"run_id": manifest["run_id"], "ts": now(), **current, **event}
        with (output / "trace.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def observer(event):
        log({"event": "llm_attempt", **event})

    client = None
    started = time.perf_counter()
    manifest["status"] = "running"
    write_json(manifest_path, manifest)
    try:
        client = client_factory(on_attempt=observer)
        for index in range(len(applications), COUNT):
            city = schedule[index]
            current.update(record=index + 1, seed_city=city)
            order = list(SPECIALITIES)
            random.Random(seed + index + 1000).shuffle(order)
            messages = build_messages(city, order, [app.full_name for app in applications[-5:]])
            # Схема проверяет принадлежность городу из списка; здесь отдельно проверяется
            # совпадение с назначенной квотой. Данные никогда не исправляются вручную.
            for resample in range(3):
                current["resample"] = resample
                app = client.chat.completions.create(
                    model=model, messages=messages, response_model=Application,
                    max_retries=3, temperature=temperature, max_tokens=1200,
                )
                if app.address.city == city:
                    break
                log({"event": "seed_mismatch", "returned_city": app.address.city})
                messages += [
                    {"role": "assistant", "content": app.model_dump_json()},
                    {"role": "user", "content": f"Нужен именно город {city}; создай заявку для него."},
                ]
            else:
                raise ValueError("Модель трижды не соблюла назначенный город")
            applications.append(app)
            write_json(checkpoint_path, [value.model_dump() for value in applications])
            log({"event": "record_accepted", "application": app.model_dump()})
            print(f"[{index + 1:02d}/{COUNT}] {city}: заявка принята", flush=True)
        manifest.update(status="complete", finished_at=now(), accepted=COUNT)
        manifest.pop("last_error", None)
    except Exception as exc:
        manifest.update(status="interrupted", accepted=len(applications), last_error={
            "type": type(exc).__name__, "status_code": getattr(exc, "status_code", None)})
        raise
    finally:
        manifest["execution_seconds"] = round(
            manifest.get("execution_seconds", 0) + time.perf_counter() - started, 3)
        write_json(manifest_path, manifest)
        if client is not None:
            client.close()
    from analysis import analyze_run
    return analyze_run(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/stratified"))
    parser.add_argument("--strategy", choices=["random", "stratified"], default="stratified")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        metrics = run_generation(args.output, args.strategy, args.seed, args.temperature, args.resume)
    except Exception as exc:
        print(explain_error(exc))
        print("Продолжение после устранения ошибки — с --resume, если каталог прогона уже создан.")
        return 1
    print(f"Готово: {metrics['count']} заявок. Критерии: {metrics['meets_good_criteria']}")
    return 0 if metrics["meets_good_criteria"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
