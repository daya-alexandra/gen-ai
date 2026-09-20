import argparse
import zipfile
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output")
    parser.add_argument("--archive", default="lab1_results.zip")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = Path(args.output)
    archive = root / args.archive
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                zf.write(path, Path("output") / path.relative_to(output))
        for name in ["README.md", "REQUIREMENTS.md"]:
            zf.write(root / name, name)
    print(f"Архив: {archive}")
