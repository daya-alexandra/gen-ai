"""Архив результатов третьей лабораторной без ключей и окружения."""
import argparse
import zipfile
from pathlib import Path
from rag_core import ROOT
from verify_results import verify


def collect(output=None,destination=None):
    out=Path(output or ROOT/'output')
    verify(out,require_api=True)
    target=Path(destination or ROOT/'lab3_results.zip')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in {'.json','.jsonl','.md','.png'}:
                archive.write(p,Path('output')/p.relative_to(out))
    print('Архив результатов:',target.resolve())
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT/'output'))
    args=parser.parse_args()
    collect(args.output)
