#!/usr/bin/env python3
"""Build ordinary mixed depth-block controls from BF16, preserving all other tensors.

Run with /tmp/breeze-quality-venv/bin/python after building breeze-quantize.
Outputs live in ~/.cache/breeze-models/depth-quant-controls. No requantization
of the original Q4_K weights: the 84 selected matrices come from the master.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import gguf
import numpy as np

from depth_quant_screen import hash_file, hash_region

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path.home() / '.cache/breeze-models'
DEST = CACHE / 'depth-quant-controls'
RECEIPT = ROOT / 'benchmarks/depth-corpus/quant-controls.json'
SUFFIXES = {
    'attn_q': 'self_attn.q_proj', 'attn_k': 'self_attn.k_proj',
    'attn_v': 'self_attn.v_proj', 'attn_output': 'self_attn.o_proj',
    'ffn_gate': 'mlp.gate_proj', 'ffn_up': 'mlp.up_proj',
    'ffn_down': 'mlp.down_proj',
}
MAPPING = {f'dd.blk.{i}.{dst}.weight': f'depth_decoder.model.layers.{i}.{src}.weight'
           for i in range(12) for dst, src in SUFFIXES.items()}


def metadata(reader):
    return {k: (v.types, v.contents()) for k, v in reader.fields.items()
            if not k.startswith('GGUF.')}


def to_f16(weight):
    result = weight.astype(np.float16)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite/overflow in F16 conversion')
    delta = result.astype(np.float64) - weight
    return result, dict(changed_values=int(np.count_nonzero(delta)),
                        max_abs=float(np.max(np.abs(delta))),
                        relative_mse=float(np.sum(delta * delta) /
                                           np.sum(weight.astype(np.float64) ** 2)))


def verify_model(base, path, expected_type):
    candidate = gguf.GGUFReader(path)
    assert metadata(base) == metadata(candidate), 'GGUF metadata changed'
    assert [t.name for t in base.tensors] == [t.name for t in candidate.tensors]
    selected = []
    for before, after in zip(base.tensors, candidate.tensors):
        assert np.array_equal(before.shape, after.shape), before.name
        if before.name in MAPPING:
            assert after.tensor_type == expected_type, before.name
            selected.append(dict(name=after.name, type=after.tensor_type.name,
                                 shape=after.shape.tolist(), bytes=after.n_bytes,
                                 sha256=hash_region(path, after.data_offset, after.n_bytes)))
        else:
            assert before.tensor_type == after.tensor_type, before.name
            assert np.array_equal(before.data, after.data), before.name
    assert len(selected) == 84
    return dict(path=str(path), bytes=path.stat().st_size, sha256=hash_file(path),
                unchanged_tensor_count=len(base.tensors) - 84,
                unchanged_tensors_byte_identical=True, metadata_identical=True,
                selected_bytes=sum(t['bytes'] for t in selected), tensors=selected)


def build():
    base_path = CACHE / 'breeze-tts-2-q4_k.gguf'
    master = CACHE / 'depth-master/depth-range.bin'
    manifest_path = ROOT / 'benchmarks/depth-corpus/master-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    binary = ROOT / 'build-ffn-bench/breeze-quantize'
    assert hash_file(base_path) == '483418fbbb438f5f1c08dbe2b017e42db0f1d126765cd3f772d8622a30a4915c'
    assert hash_file(master) == manifest['selected']['range_sha256']
    base = gguf.GGUFReader(base_path)
    if RECEIPT.exists():
        receipt = json.loads(RECEIPT.read_text())
        for kind, item in receipt['models'].items():
            actual = verify_model(base, Path(item['path']), getattr(gguf.GGMLQuantizationType, kind.upper()))
            assert actual == item, f'{kind} receipt mismatch'
        print('Existing controls verified')
        return
    DEST.mkdir(exist_ok=True)
    paths = {kind: DEST / f'breeze-depth-blocks-{kind}.gguf' for kind in ('f16', 'q4_k', 'q3_k')}
    for path in paths.values():
        if path.exists() or path.with_suffix('.part').exists():
            raise ValueError(f'refusing to overwrite unreceipted artifact: {path}')
    if shutil.disk_usage(DEST).free < base_path.stat().st_size * 3 + (1 << 30):
        raise ValueError('need space for three controls plus 1 GiB headroom')
    entries = {e['name']: e for e in manifest['tensors']}
    part = paths['f16'].with_suffix('.part')
    writer = gguf.GGUFWriter(part, 'breeze-tts-2')
    for key, (types, value) in metadata(base).items():
        writer.add_key_value(key, value, types[0], types[-1] if len(types) > 1 else None)
    conversions = []
    for tensor in base.tensors:
        if tensor.name in MAPPING:
            entry = entries[MAPPING[tensor.name]]
            assert entry['dtype'] == 'BF16'
            assert list(reversed(entry['shape'])) == tensor.shape.tolist()
            assert hash_region(master, entry['local_offset'], entry['nbytes']) == entry['sha256']
            raw = np.memmap(master, dtype='<u2', mode='r', offset=entry['local_offset'], shape=tuple(entry['shape']))
            weight = (raw.astype(np.uint32) << 16).view(np.float32)
            f16, error = to_f16(weight)
            conversions.append(dict(name=tensor.name, master_name=entry['name'], **error))
            writer.add_tensor(tensor.name, f16)
        else:
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    part.rename(paths['f16'])
    models = {'f16': verify_model(base, paths['f16'], gguf.GGMLQuantizationType.F16)}
    commands = []
    for kind in ('q4_k', 'q3_k'):
        part = paths[kind].with_suffix('.part')
        command = [str(binary), str(paths['f16']), str(part), 'f16', '--depth', kind]
        commands.append(command)
        with (DEST / f'{kind}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        part.rename(paths[kind])
        models[kind] = verify_model(base, paths[kind], getattr(gguf.GGMLQuantizationType, kind.upper()))
    receipt = dict(schema='breeze-depth-quant-controls/v1', models=models,
                   source_model_sha256=hash_file(base_path), master_sha256=hash_file(master),
                   manifest_sha256=hash_file(manifest_path), quantizer_sha256=hash_file(binary),
                   script_sha256=hash_file(Path(__file__)), commands=commands, conversion_errors=conversions,
                   caveats=['Ordinary uncalibrated ggml quantization, not GPTQ or rotated.',
                            'Only 84 depth-block linears replaced. This is a mixed model, not full BF16.',
                            'F16 is an intermediate; tiny BF16 values may round or underflow as recorded.',
                            'No speech-quality or runtime acceptance is implied.'])
    RECEIPT.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print('Built and verified three mixed depth-block controls')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        assert len(MAPPING) == len(set(MAPPING.values())) == 84
        assert MAPPING['dd.blk.11.ffn_down.weight'].endswith('layers.11.mlp.down_proj.weight')
        _, error = to_f16(np.array([1.0, 1e-9], dtype=np.float32))
        assert error['changed_values'] == 1 and error['max_abs'] < 1e-8
        print('control builder self-test: PASS')
    else:
        build()
