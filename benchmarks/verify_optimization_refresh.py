#!/usr/bin/env python3
"""Check refresh receipts against raw logs and paired output hashes (stdlib only)."""
import hashlib
import json
from pathlib import Path
import re

receipt = json.loads((Path(__file__).parent / 'depth-corpus/optimization-refresh.json').read_text())
groups = {'plain': set(), 'voice': set()}
count = 0
for block in receipt['blocks']:
    assert block['returncode'] == 0 and block['warmup'] == 1
    log = block['log']
    assert hashlib.sha256(log.encode()).hexdigest() == block['log_sha256']
    parts = log.split('generation wall')[1:]
    hashes = [line.split()[0] for line in block['wav_sha256'].splitlines()]
    assert len(parts) == len(block['runs']) == len(hashes) == 3
    for row, part, digest in zip(block['runs'], parts, hashes):
        assert re.fullmatch('[0-9a-f]{64}', digest)
        assert row['wall_ms'] == float(re.search(r'^\s+([\d.]+) ms', part)[1])
        frames, seconds = re.search(r'(\d+) frames, ([\d.]+) s audio', part).groups()
        assert (row['frames'], row['audio_s']) == (int(frames), float(seconds))
        assert abs(row['wall_rtf'] - row['wall_ms'] / (1000 * row['audio_s'])) < 1e-12
        for stage, value in row['stages_ms'].items():
            assert value == float(re.search(r'^\s*' + stage + r'\s+([\d.]+) ms', part, re.M)[1])
        groups['voice' if '--voice' in block['command'] else 'plain'].add(digest)
        count += 1
assert all(len(hashes) == 1 for hashes in groups.values())
assert count == 30
print(f'PASS: {count} runs, raw-log arithmetic, and plain/saved-voice byte parity')
