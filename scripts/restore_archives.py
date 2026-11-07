#!/usr/bin/env python3
"""Verify and safely restore the nine original research archives.

Requires the original ZIPs to be supplied by the user; they are not tracked by git.
Files are restored to isolated per-archive folders and never overwritten.
"""
from pathlib import Path, PurePosixPath
from argparse import ArgumentParser
import csv, hashlib, shutil, stat, sys, zipfile

ROOT=Path(__file__).resolve().parents[1]
SOURCE_INDEX=ROOT/'archives'/'source_archives.csv'

def sha256_file(path:Path)->str:
    d=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):d.update(chunk)
    return d.hexdigest()

def safe_relative(name:str)->Path:
    if '\\' in name or not name or name.startswith('/'):
        raise ValueError(f'Unsafe ZIP filename: {name!r}')
    pp=PurePosixPath(name)
    if any(x in ('..','.','') for x in pp.parts) or pp.is_absolute() or ':' in pp.parts[0]:
        raise ValueError(f'Unsafe ZIP filename: {name!r}')
    return Path(*pp.parts)

def extract_verified(archive:Path, destination:Path, expected_size:int):
    with zipfile.ZipFile(archive,'r') as z:
        contents=[m for m in z.infolist() if not m.is_dir()]
        actual_size=sum(m.file_size for m in contents)
        if actual_size!=expected_size:raise ValueError(f'Unexpected unpacked size: {archive.name}')
        # First validate ALL members, then write any contents.
        for m in z.infolist():
            safe_relative(m.filename)
            if stat.S_ISLNK(m.external_attr >> 16):
                raise ValueError(f'Refusing ZIP symlink: {m.filename}')
        corrupt=z.testzip()
        if corrupt is not None:raise ValueError(f'CRC failed {archive.name}:{corrupt}')
        if destination.exists():raise FileExistsError(f'Refusing overwrite: {destination}')
        destination.mkdir(parents=True)
        for m in z.infolist():
            dest=destination/safe_relative(m.filename)
            if not dest.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f'ZIP traversal: {m.filename}')
            if m.is_dir():dest.mkdir(parents=True,exist_ok=True)
            else:
                dest.parent.mkdir(parents=True,exist_ok=True)
                with z.open(m) as inp, dest.open('xb') as out:shutil.copyfileobj(inp,out)
        return len(contents)

def main()->int:
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir',required=True,type=Path)
    parser.add_argument('--output',type=Path,help='Directory into which original archive trees are restored')
    parser.add_argument('--verify-only',action='store_true')
    a=parser.parse_args()
    if not a.verify_only and a.output is None:parser.error('--output is required unless --verify-only is used')
    rows=list(csv.DictReader(SOURCE_INDEX.open(encoding='utf-8')))
    for row in rows:
        inp=a.archive_dir/row['archive']
        if not inp.is_file():raise FileNotFoundError(inp)
        if inp.stat().st_size!=int(row['bytes']) or sha256_file(inp)!=row['sha256']:
            raise ValueError(f'Original archive SHA256/size mismatch: {inp.name}')
        if a.verify_only:
            with zipfile.ZipFile(inp) as z:
                if z.testzip() is not None:raise ValueError(f'CRC failure: {inp.name}')
            print(f'VERIFIED {inp.name}')
        else:
            dst=a.output/row['stage']
            count=extract_verified(inp,dst,int(row['uncompressed_bytes']))
            print(f'RESTORED {inp.name} -> {dst} ({count} files)')
    return 0

if __name__=='__main__':sys.exit(main())
