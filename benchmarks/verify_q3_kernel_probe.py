#!/usr/bin/env python3
"""Verify Q3 kernel probe equivalence and exact runtime/audio restoration."""
import json
from pathlib import Path
r = json.loads((Path(__file__).parent/'depth-corpus/q3-kernel-probe.json').read_text())
assert len(r['micro']) == 12
hashes = {1:set(), 2:set()}
for row in r['micro']:
    d = row['result']
    assert d['weights'] == 'real_model_Q3_K' and d['layers'] == 12
    assert d['warmup'] == 20 and d['iterations'] == 100
    for c in d['cases']:
        assert c['normal_vs_packed']['max_abs'] <= 1e-5
        assert c['normal_vs_packed']['max_rel'] <= 1e-5
        assert c['normal_vs_swiglu']['max_abs'] <= 1e-5
        assert c['normal_vs_swiglu']['max_rel'] <= 1e-5
        for v in c['variants'].values():
            assert v['mean_us'] > 0 and v['checksum'] == v['repeat_checksum']
        if not row['label'].startswith('integer'):
            hashes[c['n']].add(c['variants']['packed']['checksum'])
assert all(len(h)==1 for h in hashes.values())
assert [x.split()[0] for x in r['baseline_binaries'].splitlines()] == [x.split()[0] for x in r['restored_binaries'].splitlines()]
assert len(r['restoration_speech']) == 2
for x in r['restoration_speech']:
    assert x['result']['returncode'] == 0
    assert x['result']['wav_sha256'].split()[0] == x['expected_sha256']
print('12 microbenchmark processes pass equivalence; three binaries and two reviewed WAVs restored exactly')
