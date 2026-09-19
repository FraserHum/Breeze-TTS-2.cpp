"""Validate a small captured corpus and write its training manifest (no training).

Usage: python prepare_manifest.py CAPTURE_ROOT EXAMPLES_JSON RESULTS_DIR
Length probes and the separate parity directory never enter corpus splits.
"""
import hashlib
import json
from pathlib import Path
import sys
import wave

import numpy as np


def prepare(root, examples_file, results):
    examples = json.loads(examples_file.read_text())
    ids = [x['id'] for x in examples]
    texts = [x['text'].strip() for x in examples]
    assert len(ids) == len(set(ids)) and len(texts) == len(set(texts))
    records = []
    for example in examples:
        assert example['split'] in ('train', 'validation', 'test')
        name = example['id']
        assert name and all(c.isalnum() or c in '-_' for c in name)
        directory = root / name
        chunks = [r for r in json.loads((directory / 'manifest.json').read_text())
                  if r['id'].startswith('chunk-')]
        assert chunks
        codes_path = results / (name + '.codes')
        codes = np.loadtxt(codes_path, dtype=np.int32, ndmin=1)
        assert len(codes) % 16 == 0 and ((codes >= 0) & (codes < 2048)).all()
        baseline = []
        start = 0
        for record in chunks:
            n = record['n']
            assert record['start'] == start and 1 <= n <= 12
            assert record['frames'] == min(start, 72) + n
            assert record['window_begin'] == max(0, start - 88)
            for key, count in (('input', record['frames'] * 1024), ('target', n * 1920)):
                array = np.fromfile(directory / record[key], dtype='<f4')
                assert len(array) == count and np.isfinite(array).all()
            samples = np.fromfile(directory / (record['id'] + '.production.f32'), dtype='<f4')
            assert len(samples) == n * 1920 and np.isfinite(samples).all()
            baseline.append(samples)
            records.append({**record, 'utterance_id': name, 'split': example['split'],
                            'input': str((directory / record['input']).resolve()),
                            'target': str((directory / record['target']).resolve()),
                            'target_sha256': hashlib.sha256((directory / record['target']).read_bytes()).hexdigest(),
                            'codes_sha256': hashlib.sha256(codes_path.read_bytes()).hexdigest()})
            start += n
        assert start * 16 == len(codes)
        pcm = (np.clip(np.concatenate(baseline), -1, 1) * np.float32(32767)).astype('<i2')
        with wave.open(str(results / (name + '-generated.wav'))) as wav:
            assert wav.getnchannels() == 1 and wav.getsampwidth() == 2 and wav.getframerate() == 24000
            original = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
        assert np.array_equal(pcm, original), name
    (root / 'dataset.json').write_text(json.dumps(records, indent=2) + '\n')
    summary = {'utterances': len(examples), 'records': len(records),
               'seconds': round(sum(r['n'] for r in records) * .08, 2),
               'seconds_by_split': {s: round(sum(r['n'] for r in records if r['split'] == s) * .08, 2)
                                    for s in ('train', 'validation', 'test')},
               'baseline_pcm_replay': 'exact for every utterance',
               'length_probes_in_training': False}
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    prepare(*(Path(x) for x in sys.argv[1:]))
