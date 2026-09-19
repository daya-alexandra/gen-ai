"""Собрать для проверки только результаты, без .env и виртуального окружения."""
from pathlib import Path
import zipfile
from llm_client import ROOT
from verify_results import verify


def collect(output=None, destination=None):
    output = Path(output or ROOT / 'output')
    verify(output, ROOT / 'input')
    destination = Path(destination or ROOT / 'lab2_results.zip')
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(output.rglob('*')):
            if p.is_file() and p.suffix in {'.json', '.jsonl', '.png', '.md'}:
                archive.write(p, Path('output') / p.relative_to(output))
    print(f'Архив результатов: {destination.resolve()}')
    return destination


if __name__ == '__main__':
    collect()
