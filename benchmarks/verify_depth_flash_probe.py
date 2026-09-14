#!/usr/bin/env python3
"""Verify the depth flash investigation receipt, including expected precision failures."""
import json
from pathlib import Path
from statistics import mean

r = json.loads((Path(__file__).parent / 'depth-corpus/depth-flash-probe.json').read_text())
assert r['decision'] == 'opt-in candidate; awaiting user audio quality approval'
assert r['backend_final'] == 'acc4d4d7c52f592b56f2dde3481d17e3da5cb7dd'
assert r['production_changed'] is False
for label, row in r['speech'].items():
    assert row['returncode'] == 0 and len(row['runs']) == 3 and row['warmup'] == 1
    hashes = [line.split()[0] for line in row['wav_sha256'].splitlines()]
    assert len(hashes) == 3 and len(set(hashes)) == 1
    expected = r['audio_hashes']['tagged' if 'tags' in label else 'ordinary']['flash' if label.startswith('flash') else 'control']
    assert hashes == [expected] * 3
    assert '--voice' in row['command'] and 'calliope' in row['command']
    assert all(x['wall_ms'] > 0 and x['audio_s'] > 0 and x['frames'] < 500 for x in row['runs'])
    assert ('BREEZE_DD_FLASH_ATTN=1' in row['command']) == label.startswith('flash')
for name in ['attention-cpu', 'mask-cpu', 'f32-final', 'mask-f32-final']:
    assert r['primitive_logs'][name].count('PASS') == 10
    assert 'FAIL' not in r['primitive_logs'][name]
for name in ['native-final', 'mask-native-final']:
    log = r['primitive_logs'][name]
    assert log.count('PASS') + log.count('FAIL') == 10 and 'FAIL' in log
for label, names in [('control', ['control-fa-a', 'control-fa-b']), ('flash', ['flash-a', 'flash-b'])]:
    rows = [x for n in names for x in r['speech'][n]['runs'][1:]]
    print(f"{label}: RTF {mean(x['wall_rtf'] for x in rows):.6f}; wall {mean(x['wall_ms'] for x in rows):.2f} ms; depth {mean(x['stages_ms']['depth decode']/x['frames'] for x in rows):.3f} ms/frame")
print('Stable audio receipts; native precision failures preserved; FP32 diagnostic passes; quality approval pending')
