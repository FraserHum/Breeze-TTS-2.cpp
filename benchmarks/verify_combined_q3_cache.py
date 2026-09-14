#!/usr/bin/env python3
"""Verify the frozen long-form cache comparison, including flush timing."""
import hashlib
import json
from pathlib import Path
import statistics
from quant_matrix_780m import parse_case_log

receipt = json.loads((Path(__file__).parent / 'depth-corpus/combined-q3-cache.json').read_text())
assert len(receipt['cases']) == 4
for workload in ('long', 'calliope-long'):
    fingerprints = set()
    for mode in ('off', 'on'):
        row = next(r for r in receipt['cases'] if r['label'] == f'{workload}-q3-{mode}-a')
        assert row['returncode'] == 0 and row['warmup'] == 1
        assert hashlib.sha256(row['log'].encode()).hexdigest() == row['log_sha256']
        assert parse_case_log(row['log'], 3) == row['runs']
        hashes = [line.split()[0] for line in row['wav_sha256'].splitlines()]
        assert len(hashes) == 3 and len(set(hashes)) == 1
        fingerprints.update(hashes)
        assert all(r['frames'] > 0 and r['audio_s'] > 0 for r in row['runs'])
        runs = row['runs'][1:]
        print(f'{workload} {mode}: warmed mean RTF {statistics.mean(r["wall_rtf"] for r in runs):.6f}; first audio max {max(r["first_audio_ms"] for r in runs):.0f} ms; delivery deficit max {max(r["max_playback_deficit_ms"] for r in runs):.1f} ms')
    assert len(fingerprints) == 1
print('12 long-form generations verified; cache modes match exactly within each workload')
