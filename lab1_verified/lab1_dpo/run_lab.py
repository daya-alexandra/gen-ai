"""Одна команда: случайный baseline → стратификация → сравнительные выводы."""

import argparse
from pathlib import Path

from analysis import compare_runs
from generator import run_generation
from llm_client import explain_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        for strategy in ("random", "stratified"):
            print(f"\nСтратегия: {strategy}", flush=True)
            run_generation(args.output / strategy, strategy=strategy, seed=args.seed, resume=args.resume)
        result = compare_runs(args.output / "random", args.output / "stratified")
    except Exception as exc:
        print(explain_error(exc))
        print("После устранения ошибки: python run_lab.py --resume")
        return 1
    print(f"\nИтоговые файлы: {(args.output / 'stratified').resolve()}")
    print("Пороговые критерии: " + ("соблюдены" if result["meets_good_criteria"] else "НЕ соблюдены"))
    return 0 if result["meets_good_criteria"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
