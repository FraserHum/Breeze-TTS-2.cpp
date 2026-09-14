#!/usr/bin/env python3
"""Validate the frozen IQ2 model selection and paired Calliope receipts."""
import json
from pathlib import Path
import statistics
r = json.loads((Path(__file__).parent / 'depth-corpus/iq2-ffn-refresh.json').read_text())
m = r['model']
assert m['metadata_identical'] and m['other_tensors_byte_identical']
assert len(m['tensors']) == 1161
ffn = [t for t in m['tensors'] if t['source'] == 'calibrated_ffn']
assert len(ffn) == 36 and all(t['type'] == 'IQ2_S' for t in ffn)
assert sum(t['bytes'] for t in ffn) == 96731136
assert len(r['speech']) == 4
for row in r['speech']:
    assert row['returncode'] == 0 and len(row['runs']) == 3
    hashes = [x.split()[0] for x in row['wav_sha256'].splitlines()]
    assert len(hashes) == 3 and len(set(hashes)) == 1
    assert all(0 < run['frames'] < 750 for run in row['runs'])
    for opt in ('--voice', '--instruction', '--text', '--seed'):
        c, first = row['command'], r['speech'][0]['command']
        assert c[c.index(opt)+1] == first[first.index(opt)+1]
for mode in ('q3', 'iq2'):
    runs = [run for row in r['speech'] if f'-{mode}-' in row['label'] for run in row['runs'][1:]]
    assert len(runs) == 4
    print(mode, 'warmed mean RTF', statistics.mean(run['wall_rtf'] for run in runs))
print('36 IQ2 FFNs and twelve deterministic completed generations verified')
