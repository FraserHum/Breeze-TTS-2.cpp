#!/usr/bin/env python3
"""Check wave32 micro results and computed-history audio/boundary receipts."""
import json
from pathlib import Path
from statistics import mean

r = json.loads((Path(__file__).parent / 'depth-corpus/wave-trim-probe.json').read_text())
assert len(r['wave_micro']) == 4
for row in r['wave_micro']:
    d = row['result']
    assert d['layers'] == 12 and d['weights'] == 'real_model_Q3_K'
    assert d['warmup'] == 20 and d['iterations'] == 100
    for case in d['cases']:
        assert case['normal_vs_packed']['max_abs'] == 0
        for variant in case['variants'].values():
            assert variant['checksum'] == variant['repeat_checksum']
            assert variant['mean_us'] > 0
assert len(r['speech']) == 5
groups = {'baseline': [], 'trim11': []}
for run in r['speech']:
    tagged = run['label'] == 'trim11-tags'
    count = 2 if tagged else 3
    expected = r['tags_sha256' if tagged else 'ordinary_sha256']
    assert run['returncode'] == 0 and len(run['runs']) == count
    assert [x.split()[0] for x in run['wav_sha256'].splitlines()] == [expected] * count
    if not tagged:
        groups['baseline' if run['label'].startswith('baseline') else 'trim11'].extend(run['runs'][1:])
for label, runs in groups.items():
    assert len(runs) == 4
    print(f"{label}: vocoder {mean(x['stages_ms']['vocoder']/x['frames'] for x in runs):.3f} ms/frame")
for key in ['boundary_initial_log', 'boundary_restored_log']:
    assert r[key].count('max_diff=0 PASS') == 18 and 'FAIL' not in r[key]
print('Four micro processes repeat; 14 reviewed WAV hashes and 18 boundary cases match exactly')
