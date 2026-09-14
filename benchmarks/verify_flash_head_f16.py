#!/usr/bin/env python3
"""Verify the F16-head conversion and paired CFG=1 flash speech receipts."""
import json
from pathlib import Path
from statistics import mean

r = json.loads((Path(__file__).parent / 'depth-corpus/flash-head-f16.json').read_text())
c = r['conversion']
assert c['source_sha256'] == '34483105709c75c38cdafcddcb4ec91e0b222097d30e76432507c448053a83de'
assert c['candidate_sha256'] == r['remote_candidate_sha256']
assert c['target'] == 'dd.codebooks_head.weight'
assert c['target_source_type'] == 'F32' and c['target_candidate_type'] == 'F16'
assert c['non_target_payloads_verified'] + 1 == c['tensor_count'] == 1161
assert c['metadata_prefix_unchanged'] and c['layout_contiguous_aligned']
assert c['target_f16_finite_elements'] == 1024 * 2051 * 15
assert c['source_bytes'] - c['candidate_bytes'] == 63006720
assert r['production_changed'] is False and r['cfg_scale'] == 1
assert len(r['speech']) == 8
for name, row in r['speech'].items():
    assert row['returncode'] == 0 and len(row['runs']) == 3 and row['warmup'] == 1
    assert 'BREEZE_DD_FLASH_ATTN=1' in row['command']
    expected = r['approved_audio']['tagged' if 'tags' in name else 'ordinary']
    assert [line.split()[0] for line in row['wav_sha256'].splitlines()] == [expected] * 3
    assert row['binary_md5'].split()[0] == '607db056a5fe5e62f76cc3c0a3537a00'
    model = '/tmp/breeze-current-ffn-q3-head-f16.gguf' if name.startswith('head') else '/tmp/breeze-current-ffn-q3-calibrated.gguf'
    assert model in row['command']
    for x in row['runs']:
        assert x['frames'] == (175 if 'tags' in name else 115)
        assert abs(x['wall_ms'] / (1000 * x['audio_s']) - x['wall_rtf']) < 1e-9
for kind in ['control', 'head']:
    rows = [x for suffix in ['a', 'b'] for x in r['speech'][f'{kind}-{suffix}']['runs'][1:]]
    print(f"{kind}: warm RTF {mean(x['wall_rtf'] for x in rows):.6f}; depth {mean(x['stages_ms']['depth decode']/x['frames'] for x in rows):.3f} ms/frame")
for kind in ['control', 'head']:
    rows = [x for suffix in ['tags', 'tags-b'] for x in r['speech'][f'{kind}-{suffix}']['runs'][1:]]
    print(f"{kind} tagged: warm RTF {mean(x['wall_rtf'] for x in rows):.6f}; depth {mean(x['stages_ms']['depth decode']/x['frames'] for x in rows):.3f} ms/frame")
print('Only head precision changed; all 24 WAVs match approved CFG=1 flash audio')
