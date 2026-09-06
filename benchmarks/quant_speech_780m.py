#!/usr/bin/env python3
"""Independent ASR disagreement and speaker-cosine probes; neither is acceptance.

Requires whisper-cli, multilingual Whisper weights, ffmpeg, numpy, OpenCC and
sherpa-onnx. Models and tools are explicit inputs; this script downloads nothing.
It evaluates complete natural fixtures only, never capped stress against full text.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import unicodedata


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while data := f.read(4 << 20):
            h.update(data)
    return h.hexdigest()


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = new
    return row[-1]


def tokens(text, language):
    text = unicodedata.normalize('NFKC', text).lower()
    if language == 'zh':
        from opencc import OpenCC
        return list(''.join(c for c in OpenCC('t2s').convert(text) if c.isalnum()))
    return re.findall(r'[a-z0-9]+', text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path, nargs='?')
    p.add_argument('--asr-bin', type=Path)
    p.add_argument('--asr-model', type=Path)
    p.add_argument('--speaker-model', type=Path)
    p.add_argument('--speaker-fixture', default='reference')
    p.add_argument('--output', type=Path)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--verify', type=Path)
    args = p.parse_args()
    if args.verify:
        result = json.loads(args.verify.read_text())
        assert result['schema'] == 'breeze-quant-speech/v1'
        for sample in result['samples'].values():
            expected, actual = sample['normalized_expected'], sample['normalized_transcript']
            assert sample['reference_units'] == len(expected) > 0
            assert sample['edit_distance'] == distance(expected, actual)
            assert sample['disagreement_rate'] == sample['edit_distance'] / len(expected)
            assert len(sample['wav_sha256']) == 64
        assert abs(result['speaker']['reference_self_cosine'] - 1) < 1e-6
        assert all(-1.000001 <= value <= 1.000001 for key, value in result['speaker'].items() if key.endswith('cosine'))
        print('speech receipt verification: PASS')
        return
    if args.self_test:
        assert distance('kitten', 'sitting') == 3
        assert distance([], ['x']) == 1 and distance(['a'], ['a']) == 0
        assert tokens('天氣，晴朗。', 'zh') == tokens('天气晴朗', 'zh')
        assert tokens('Blue notebooks, THREE.', 'en') == ['blue', 'notebooks', 'three']
        print('speech-probe self-test: PASS')
        return
    if not all([args.directory, args.asr_bin, args.asr_model, args.speaker_model, args.output]):
        p.error('directory, --asr-bin, --asr-model, --speaker-model and --output are required')
    import numpy as np
    import sherpa_onnx
    manifest = json.loads((args.directory / 'manifest.json').read_text())
    work = args.directory / 'speech-probes'
    work.mkdir(exist_ok=True)
    result = dict(schema='breeze-quant-speech/v1', samples={},
        models={k: dict(path=str(v), sha256=sha(v)) for k, v in
                [('asr_binary', args.asr_bin), ('asr_model', args.asr_model), ('speaker_model', args.speaker_model)]},
        versions={k: importlib.metadata.version(k) for k in ['numpy', 'sherpa-onnx', 'opencc-python-reimplemented']},
        normalization='NFKC/lowercase; English alphanumeric words; Mandarin OpenCC t2s then alphanumeric characters.',
        limitations=['ASR disagreement includes recognizer errors, spelling and number formatting; not human WER.',
                     'No expected transcript is given to the recognizer as a prompt.',
                     'Speaker cosine is descriptive; no task-specific acceptance threshold is established.',
                     'The reference clip is synthetic speech, not a human speaker validation set.',
                     'Only first-repeat WAVs are scored; repeated identical audio is not independent quality evidence.'])
    for fixture in manifest['fixtures']:
        if fixture['kind'] == 'capped-throughput-only':
            continue
        language = 'zh' if fixture['id'] == 'mandarin' else 'en'
        expected = tokens(fixture['text'], language)
        assert expected
        for kind in ['q4_k', 'q3_k']:
            name = fixture['id'] + '-' + kind
            source = args.directory / (name + '.wav')
            converted = work / (name + '.16k.wav')
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-ar', '16000',
                            '-ac', '1', '-c:a', 'pcm_s16le', str(converted)], check=True)
            prefix = work / name
            command = [str(args.asr_bin), '-m', str(args.asr_model), '-f', str(converted),
                       '-l', language, '-t', '2', '-nf', '-tp', '0', '-otxt', '-oj', '-of', str(prefix)]
            with prefix.with_suffix('.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
            transcript = prefix.with_suffix('.txt').read_text().strip()
            actual = tokens(transcript, language)
            errors = distance(expected, actual)
            result['samples'][name] = dict(wav_sha256=sha(source), language=language,
                expected_text=fixture['text'], transcript=transcript, normalized_expected=expected,
                normalized_transcript=actual, edit_distance=errors, reference_units=len(expected),
                metric='CER' if language == 'zh' else 'WER', disagreement_rate=errors / len(expected),
                command=command, raw_asr=json.loads(prefix.with_suffix('.json').read_text()))
            print(name, result['samples'][name]['metric'], errors / len(expected), transcript, flush=True)
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(args.speaker_model), num_threads=2, provider='cpu')
    assert config.validate()
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
    def embedding(path):
        raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-ar', '16000', '-ac', '1', '-f', 'f32le', '-'])
        audio = np.frombuffer(raw, dtype='<f4').copy()
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=16000, waveform=audio)
        stream.input_finished()
        assert extractor.is_ready(stream)
        vector = np.asarray(extractor.compute(stream), dtype=np.float64)
        assert np.isfinite(vector).all() and np.linalg.norm(vector) > 0
        return vector / np.linalg.norm(vector)
    reference = args.directory / 'reference-input.wav'
    ref = embedding(reference)
    vectors = {kind: embedding(args.directory / (args.speaker_fixture + '-' + kind + '.wav')) for kind in ['q4_k', 'q3_k']}
    result['speaker'] = dict(fixture=args.speaker_fixture, reference_sha256=sha(reference), embedding_dim=len(ref),
        reference_self_cosine=float(ref @ ref), q4_reference_cosine=float(vectors['q4_k'] @ ref),
        q3_reference_cosine=float(vectors['q3_k'] @ ref), q3_q4_cosine=float(vectors['q3_k'] @ vectors['q4_k']))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(result['speaker'])


if __name__ == '__main__':
    main()
