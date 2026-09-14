#!/usr/bin/env python3
"""Check retained Calliope parity receipts and recompute the warm ABBA result."""
import json
from pathlib import Path
from statistics import mean

r = json.loads((Path(__file__).parent / 'depth-corpus/depth-sync-probe.json').read_text())
assert [x['label'] for x in r['runs']] == [
    'baseline-a', 'async-a', 'async-b', 'baseline-b', 'async-tags']
groups = {'baseline': [], 'async': []}
for run in r['runs']:
    tagged = run['label'] == 'async-tags'
    expected = r['tags_approved_sha256' if tagged else 'ordinary_approved_sha256']
    count = 2 if tagged else 3
    assert run['returncode'] == 0 and len(run['runs']) == count
    hashes = [line.split()[0] for line in run['wav_sha256'].splitlines()]
    assert hashes == [expected] * count
    assert all(x['frames'] == (151 if tagged else 114) for x in run['runs'])
    if not tagged:
        groups[run['label'].split('-')[0]].extend(run['runs'][1:])
for mode, runs in groups.items():
    assert len(runs) == 4
    print(f"{mode}: warm RTF {mean(x['wall_rtf'] for x in runs):.6f}, "
          f"depth {mean(x['stages_ms']['depth decode']/x['frames'] for x in runs):.3f} ms/frame")
print('14 WAVs match reviewed Q3 audio, including two CFG=2 delivery-tag runs')
