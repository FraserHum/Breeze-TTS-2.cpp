#!/usr/bin/env python3
"""Compare fixed-seed vocoder variants; requires numpy, pesq, torch, torchaudio, ffmpeg."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import wave

import numpy as np
from pesq import pesq
import torch

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
CONFIGS = ['legacy', 'matmul', 'voc2', 'all3']


def load(path):
    with wave.open(str(path), 'rb') as f:
        assert (f.getnchannels(), f.getsampwidth(), f.getframerate()) == (1, 2, 24000)
        return np.frombuffer(f.readframes(f.getnframes()), dtype='<i2').astype(np.float64) / 32768


def resample(path):
    return np.frombuffer(subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path),
                         '-f', 'f32le', '-ar', '16000', '-ac', '1', '-']), dtype='<f4').copy()


def main():
    torch.set_num_threads(2)
    model = torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong', trust_repo=True)
    checkpoint = Path(torch.hub.get_dir()) / 'checkpoints/utmos22_strong_step7459_v1.pt'
    result = dict(versions={n: importlib.metadata.version(n) for n in ['numpy','pesq','torch','torchaudio']},
                  utmos_source='https://github.com/tarepan/SpeechMOS/tree/v1.2.0',
                  utmos_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  samples={}, comparisons={}, repeatability={}, decisions={})
    arrays, narrow = {}, {}
    scores_by_hash = {}
    for quant in ['q4_k', 'q8_0']:
        for config in CONFIGS:
            for repeat in [1, 2]:
                name = f'{config}-{quant}-s{repeat}'
                path = ROOT / (name + '.wav')
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                a, b = load(path), resample(path)
                arrays[name], narrow[name] = a, b
                if digest not in scores_by_hash:
                    with torch.inference_mode():
                        scores_by_hash[digest] = float(model(torch.from_numpy(b)[None, :], 16000)[0])
                result['samples'][name] = dict(sha256=digest, samples=len(a), duration_s=len(a)/24000,
                                               rms=float(np.sqrt(np.mean(a*a))), peak=float(np.max(np.abs(a))),
                                               utmos=scores_by_hash[digest])
                print(name, 'UTMOS', scores_by_hash[digest], flush=True)
            names = [f'{config}-{quant}-s{i}' for i in [1, 2]]
            result['repeatability'][f'{config}-{quant}'] = len({result['samples'][n]['sha256'] for n in names}) == 1
        for config in ['matmul', 'voc2']:
            verdicts = []
            for repeat in [1, 2]:
                ref, candidate = f'legacy-{quant}-s{repeat}', f'{config}-{quant}-s{repeat}'
                a, b = arrays[ref], arrays[candidate]
                assert len(a) == len(b), 'VOC flags unexpectedly changed frame count'
                d = a-b
                snr = float(10*np.log10(np.sum(a*a)/np.sum(d*d))) if np.any(d) else 999.0
                corr = float(np.corrcoef(a, b)[0, 1])
                maximum = float(np.max(np.abs(d)))
                metrics = dict(snr_db=snr, correlation=corr, max_abs_diff=maximum,
                               rms_diff=float(np.sqrt(np.mean(d*d))),
                               pesq_wb=float(pesq(16000,narrow[ref],narrow[candidate],'wb')),
                               utmos_delta=result['samples'][candidate]['utmos']-result['samples'][ref]['utmos'])
                metrics['waveform_gate_pass'] = snr >= 30 and corr >= .999 and maximum < .03
                result['comparisons'][candidate] = metrics
                verdicts.append(metrics['waveform_gate_pass'])
            result['decisions'][f'{config}-{quant}'] = all(verdicts) and result['repeatability'][f'{config}-{quant}'] and result['repeatability'][f'legacy-{quant}']
        ref, fused = result['samples'][f'voc2-{quant}-s1'], result['samples'][f'all3-{quant}-s1']
        result['comparisons'][f'all3-{quant}'] = dict(duration_ratio=fused['duration_s']/ref['duration_s'],
                                                    utmos_delta=fused['utmos']-ref['utmos'],
                                                    decision='Keep opt-in: different sampling/RNG contract; single prompt insufficient.')
    (ROOT/'quality.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['decisions']),flush=True)


if __name__ == '__main__':
    main()
