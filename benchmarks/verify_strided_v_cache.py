#!/usr/bin/env python3
"""Validate archived strided-cache receipts and print warmed speech comparisons."""
import json
from pathlib import Path
import statistics

record = json.loads((Path(__file__).parent / 'depth-corpus/strided-v-cache.json').read_text())
assert len(record['context']) == 12
for context in (64, 257, 1024):
    rows = [r for r in record['context'] if r['label'].startswith(f'context-{context}-')]
    assert len(rows) == 4
    fingerprints = set()
    for row in rows:
        assert row['returncode'] == 0
        assert 'strided_cache_snapshot_and_matvec_check=true' in row['output']
        assert 'exact_repeated_hidden_and_logits=true' in row['output']
        fingerprints.add(row['output'].split('output_fnv1a64=')[1].split()[0])
    assert len(fingerprints) == 1
assert len(record['speech']) == 8
for workload in ('plain', 'voice68'):
    rows = [r for r in record['speech'] if r['label'].startswith(workload + '-')]
    hashes = set()
    for row in rows:
        assert row['returncode'] == 0 and len(row['runs']) == 3
        wavs = row['wav_sha256'].splitlines()
        assert len(wavs) == 3
        hashes.update(line.split()[0] for line in wavs)
        assert all(r['frames'] > 0 and r['audio_s'] > 0 for r in row['runs'])
    assert len(hashes) == 1
    for mode in ('off', 'on'):
        warmed = [r['wall_rtf'] for row in rows if f'-{mode}-' in row['label'] for r in row['runs'][1:]]
        assert len(warmed) == 4
        print(f'{workload} {mode}: mean warmed RTF {statistics.mean(warmed):.6f}')
print('12 context runs and 24 speech WAVs verified; exact parity within each workload')
