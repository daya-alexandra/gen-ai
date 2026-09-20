import argparse
import zipfile
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output")
    parser.add_argument("--archive", default="final_project_results.zip")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = Path(args.output)
    archive = root / args.archive
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                zipped.write(path, Path("output") / path.relative_to(output))
        for name in ["README.md", "REQUIREMENTS.md"]:
            zipped.write(root / name, name)
        for path in sorted((root / "input").rglob("*")):
            if path.is_file():
                zipped.write(path, path.relative_to(root))
    print(f"Архив: {archive}")
