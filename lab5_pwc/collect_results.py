"""Архив только проверенного output, без ключей и виртуального окружения."""
import argparse
import zipfile
from pathlib import Path
from verify_results import verify

def collect(output='output', destination=None):
    output = Path(output)
    verify(output, require_api=True)
    number = '5' if (Path(__file__).parent / 'schemas_pwc.py').exists() else '4'
    destination = Path(destination or f'lab{number}_results.zip')
    temp = destination.with_suffix('.tmp.zip')
    files = sorted(p for p in output.rglob('*') if p.is_file())
    if any(p.name == '.env' or p.suffix not in ('.json', '.jsonl', '.md') for p in files):
        raise ValueError('В output обнаружен посторонний файл; архив не создан')
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, 'output/' + p.relative_to(output).as_posix())
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None:
            raise ValueError('Архив повреждён')
    temp.replace(destination)
    print('Готово: ' + str(destination.resolve()))
    return destination

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='output')
    p.add_argument('--destination')
    a = p.parse_args()
    collect(a.output, a.destination)
