#!/usr/bin/env python3
"""Verify curated tracked research files against the generated SHA-256 index.

The index excludes itself; it authenticates consistency against this snapshot,
not the identity or trustworthiness of the original research.
"""
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / 'checks' / 'SHA256SUMS.txt'

def verify() -> int:
    if not INDEX.is_file():
        print(f'ERROR: missing {INDEX}', file=sys.stderr)
        return 1
    failures=[]
    count=0
    for lineno, line in enumerate(INDEX.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip() or line.startswith('#'): continue
        if '  ' not in line:
            failures.append(f'malformed record at line {lineno}')
            continue
        digest, name=line.split('  ',1)
        if len(digest)!=64 or any(ch not in '0123456789abcdef' for ch in digest):
            failures.append(f'invalid SHA256 line {lineno}')
            continue
        rel=Path(name)
        target=(ROOT/rel).resolve()
        if rel.is_absolute() or '..' in rel.parts or not target.is_relative_to(ROOT.resolve()):
            failures.append(f'unsafe path line {lineno}: {name}')
            continue
        if not target.is_file():
            failures.append(f'missing file: {name}')
            continue
        actual=hashlib.sha256(target.read_bytes()).hexdigest()
        if actual!=digest: failures.append(f'digest mismatch: {name}')
        count+=1
    print(f'Checked {count} files against {INDEX.relative_to(ROOT)}.')
    for error in failures: print('ERROR:',error,file=sys.stderr)
    return 1 if failures else 0

if __name__=='__main__': sys.exit(verify())
