"""Sequential, resumable pilot capture using the existing checked native tools.

Run from the native work directory: python capture_pilot.py PROMPTS_JSON NEW_ROOT.
A completed stage is reused only with its original immutable prompt list. Failed
stages retain their logs and rerun on the next invocation; no examples are dropped.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare_manifest import prepare


def sha256(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def capture(prompts, root):
    examples = json.loads(prompts.read_text())
    ids, texts, families = set(), set(), {}
    for e in examples:
        name = e['id']
        if (not name or not all(c.isalnum() or c in '-_' for c in name)
                or name in ids or e['text'].strip() in texts
                or e['split'] not in ('train', 'validation', 'test')):
            raise ValueError('Invalid or duplicated prompt')
        if families.setdefault(e['family'], e['split']) != e['split']:
            raise ValueError('Prompt family crosses splits')
        ids.add(name)
        texts.add(e['text'].strip())
    root.mkdir(parents=True, exist_ok=True)
    frozen = root / 'examples.json'
    if frozen.exists() and frozen.read_bytes() != prompts.read_bytes():
        raise ValueError('Cannot resume with changed prompts')
    frozen.write_bytes(prompts.read_bytes())
    results = root / 'results'
    results.mkdir(exist_ok=True)
    env = dict(os.environ, GGML_VK_VISIBLE_DEVICES='0', GGML_VK_ALLOW_GRAPHICS_QUEUE='1',
               BREEZE_DD_FLASH_ATTN='1', BREEZE_DD_FUSED='0', BREEZE_VOC_TRIM='1',
               BREEZE_VOC_CONVT_MATMUL='1', BREEZE_VOC_STATEFUL='0', BREEZE_V_CACHE_TRANSPOSED='0')
    for index, example in enumerate(examples):
        name = example['id']
        prefix = results / name
        textfile = results / (name + '.txt')
        textfile.write_text(example['text'])
        for stage, model in [('generate', 'production'), ('production', 'production'), ('teacher', 'teacher-f16')]:
            done = results / (name + '-' + stage + '.done')
            if done.exists():
                continue
            started = time.monotonic()
            if stage == 'generate':
                command = ['./teacher-compare', 'models/production.gguf', 'models/calliope.breeze',
                           str(textfile), str(prefix), 'generate']
            else:
                command = ['./tail-capture', f'models/{model}.gguf', str(prefix) + '.codes',
                           str(root / name), stage]
            with (results / (name + '-' + stage + '.log')).open('w') as log:
                subprocess.run(command, env=dict(env, BREEZE_PACK_WEIGHTS='1' if stage == 'generate' else '0'),
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
            receipt = dict(id=name, stage=stage, seconds=time.monotonic()-started,
                           index=index+1, total=len(examples), command=command)
            done.write_text(json.dumps(receipt) + '\n')
            print(json.dumps(receipt), flush=True)
    prepare(root, frozen, results)
    provenance = dict(prompts_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
                      voice='Calliope', cfg=1, seed=42, instruction='Speak clearly and naturally.',
                      flags={k:v for k,v in env.items() if k.startswith(('BREEZE_', 'GGML_VK_'))},
                      artifacts={p:sha256(p) for p in
                          ['models/production.gguf','models/teacher-f16.gguf','models/calliope.breeze',
                           'teacher-compare','tail-capture','capture_pilot.py','prepare_manifest.py']})
    (root / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    capture(Path(sys.argv[1]), Path(sys.argv[2]))
