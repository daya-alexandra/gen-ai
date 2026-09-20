import argparse

from pipeline import execute


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output")
    args = parser.parse_args()
    result = execute(args.output)
    print(
        f"Готово: {result['metrics']['passed']}/{result['metrics']['cases']}; "
        f"API-ответов: {result['metrics']['api_responses']}"
    )
