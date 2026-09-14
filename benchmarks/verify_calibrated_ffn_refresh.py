#!/usr/bin/env python3
"""Check frozen model selection and warmed runtime receipts."""
import json
from pathlib import Path
import statistics

record = json.loads((Path(__file__).parent / 'depth-corpus/calibrated-ffn-refresh.json').read_text())
model = record['model']
assert model['metadata_identical'] and model['other_tensors_byte_identical']
assert len(model['tensors']) == 1161
changed = [t for t in model['tensors'] if t['source'] == 'calibrated_ffn']
assert len(changed) == 36 and all(t['type'] == 'Q3_K' for t in changed)
assert sum(t['bytes'] for t in changed) == 129761280
assert record['quality_approval'] == 'pending user listening approval'
assert len(record['speech']) == 8
for row in record['speech']:
    assert row['returncode'] == 0 and len(row['runs']) == 3
    hashes = [line.split()[0] for line in row['wav_sha256'].splitlines()]
    assert len(hashes) == 3 and len(set(hashes)) == 1
    assert all(0 < r['frames'] < 750 and r['audio_s'] > 0 for r in row['runs'])
    assert all(abs(r['wall_rtf'] - r['wall_ms'] / 1000 / r['audio_s']) < 1e-9 for r in row['runs'])
for workload in ('report', 'home', 'voice'):
    for mode in ('q4', 'q3'):
        runs = [r for row in record['speech'] if row['label'].startswith(f'{workload}-{mode}-') for r in row['runs'][1:]]
        print(f'{workload} {mode}: RTF {statistics.mean(r["wall_rtf"] for r in runs):.6f}; depth {statistics.mean(r["stages_ms"]["depth decode"] / r["frames"] for r in runs):.3f} ms/frame')
print('36 selected tensors and 24 completed deterministic speech runs verified; quality approval remains pending')
