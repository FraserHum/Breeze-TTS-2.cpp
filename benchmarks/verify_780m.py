#!/usr/bin/env python3
"""Verify the archived 780M report against its raw stage logs (stdlib only)."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
d = json.loads((ROOT / '780m-2026-09-06.json').read_text())
report = (ROOT / 'audio-cpp-780m.md').read_text()
assert len(d['runs']) == 16
assert len({(r['config'], r['quant'], r['repeat']) for r in d['runs']}) == 16
for r in d['runs'] + d['dd2_runs']:
    log = d['logs'][f"{r['config']}-{r['quant']}-s{r['repeat']}"]
    frames, seconds = re.search(r'(\d+) frames, ([\d.]+) s audio', log).groups()
    assert int(frames) == r['frames'] and float(seconds) == r['audio_s']
    for stage, value in r['stages_ms'].items():
        assert float(re.search(r'^\s*' + stage + r'\s+([\d.]+) ms', log, re.M)[1]) == value
    total = sum(r['stages_ms'].values())
    decode = sum(r['stages_ms'][k] for k in ['backbone decode', 'depth decode', 'vocoder'])
    assert abs(r['stage_rtf'] - total / (1000 * r['audio_s'])) < 1e-12
    assert abs(r['decode_rtf'] - decode / (1000 * r['audio_s'])) < 1e-12
    assert f"{r['stage_rtf']:.4f}" in report
q = d['quality']
assert all(q['repeatability'].values())
for quant in ['q4_k', 'q8_0']:
    assert q['decisions']['matmul-' + quant]
    assert not q['decisions']['voc2-' + quant]
for smoke in d['smoke']:
    name = f"{smoke['matched_config']}-q4_k-s1"
    assert smoke['sha256'] == q['samples'][name]['sha256']
assert d['dd2_quality']['repeatable']
print('PASS: 18 measured runs, RTF arithmetic, quality decisions and default/opt-out smoke hashes')
