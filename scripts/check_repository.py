#!/usr/bin/env python3
"""Check links in authored documentation and stage indexes (not legacy reports)."""
from pathlib import Path
import re,sys,urllib.parse
ROOT=Path(__file__).resolve().parents[1]
FILES=[ROOT/'README.md',ROOT/'README.en.md',ROOT/'CONTRIBUTING.md']
FILES+=list((ROOT/'docs').glob('*.md'))
FILES+=list((ROOT/'experiments').glob('*/INDEX.md'))
FILES+=list((ROOT/'experiments').glob('stage-09/README.md'))
link_re=re.compile(r'(?<!\!)\[[^\]]+\]\(([^)]+)\)')
failed=[];links=0
for f in FILES:
    for link in link_re.findall(f.read_text(encoding='utf8')):
        link=link.split(' ',1)[0].strip('<>')
        if link.startswith(('https://','http://','mailto:','#')):continue
        rel=urllib.parse.unquote(link.split('#')[0].split('?')[0])
        if not rel:continue
        dst=(f.parent/rel).resolve()
        if not dst.is_relative_to(ROOT.resolve()) or not dst.exists():
            failed.append(f'{f.relative_to(ROOT)} -> {link}')
        else: links+=1
for m in failed:print('BROKEN:',m,file=sys.stderr)
print(f'Checked {links} local authored links across {len(FILES)} documents.')
sys.exit(1 if failed else 0)
