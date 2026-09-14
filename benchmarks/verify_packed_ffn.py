#!/usr/bin/env python3
"""Verify packed FFN measurements and the rejected dispatch probe receipts."""
import hashlib
import json
from pathlib import Path
import re

d = json.loads((Path(__file__).parent / 'depth-corpus/packed-ffn-probe.json').read_text())
assert len(d['microbenchmarks']) == 5
for block in d['microbenchmarks']:
    assert [case['n'] for case in block['cases']] == [1, 2]
    for case in block['cases']:
        packed = case['variants']['packed']
        assert packed['mean_us'] > 0 and packed['submissions'] == 1
        assert packed['checksum'] == packed['repeat_checksum']
        assert case['normal_vs_packed']['max_abs'] <= 1e-5
        assert case['normal_vs_packed']['max_rel'] <= 1e-5
        value = re.search(r'variant=packed n=' + str(case['n']) + r'.*?mean_us=([\d.]+)', block['log'])
        assert abs(float(value[1]) - packed['mean_us']) < 0.001
control, candidate = d['generation']
for block in [control, candidate]:
    assert block['returncode'] == 0 and len(block['runs']) == 3
    assert hashlib.sha256(block['log'].encode()).hexdigest() == block['log_sha256']
    hashes = [line.split()[0] for line in block['wav_sha256'].splitlines()]
    assert len(hashes) == 3 and len(set(hashes)) == 1
    for row, part in zip(block['runs'], block['log'].split('generation wall')[1:]):
        assert row['wall_ms'] == float(re.search(r'^\s+([\d.]+) ms', part)[1])
        frames, seconds = re.search(r'(\d+) frames, ([\d.]+) s audio', part).groups()
        assert (row['frames'], row['audio_s']) == (int(frames), float(seconds))
        assert abs(row['wall_rtf'] - row['wall_ms'] / (1000 * row['audio_s'])) < 1e-12
assert control['runs'][0]['frames'] == 213
assert candidate['runs'][0]['frames'] == 235
assert control['wav_sha256'].split()[0] != candidate['wav_sha256'].split()[0]
print('PASS: packed FFN equivalence, timing receipts, and rejected speech-parity gate')
