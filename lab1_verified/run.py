"""Commands for the first lab. check and verify never call the API."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["check", "verify", "lab1"])
    args = parser.parse_args()
    # Child interpreters use UTF-8 on Windows as well as Linux.
    env = dict(os.environ, PYTHONUTF8="1")
    commands = {
        "check": ["-m", "unittest", "discover", "-s", "tests", "-v"],
        "verify": ["verify_results.py"],
        "lab1": ["run_lab.py", "--resume"],
    }
    return subprocess.run(
        [sys.executable, *commands[args.task]],
        cwd=ROOT / "lab1_dpo", env=env,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
