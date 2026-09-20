import argparse
from experiments import execute


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output")
    args = parser.parse_args()
    execute(args.output)
    print(f"Готово: {args.output}")
