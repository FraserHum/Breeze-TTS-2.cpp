"""Evaluate an exported student on one split, without updates or checkpoint selection."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from analyze_audio import analyze
from student import Student, read_f32
from train import write_wav


def evaluate(weights, dataset, split, out, device):
    records = json.loads(dataset.read_text())
    if isinstance(records, dict):
        records = records['records']
    groups = defaultdict(list)
    for r in records:
        if r['split'] == split:
            name = r['utterance_id']
            if not name or not all(c.isalnum() or c in '-_' for c in name):
                raise ValueError('Unsafe utterance ID')
            groups[name].append(r)
    if not groups:
        raise ValueError('Empty requested split')
    out.mkdir(parents=True, exist_ok=False)
    model = Student(weights).to(device).eval()
    report = {}
    with torch.no_grad():
        for name, rows in groups.items():
            rows.sort(key=lambda r:r['start'])
            predictions, targets, start = [], [], 0
            for r in rows:
                if r['start'] != start:
                    raise ValueError('Noncontiguous utterance')
                x = read_f32(dataset.parent / r['input'], (1,r['frames'],1024)).to(device)
                predictions.append(model(x,r['n']).cpu().numpy().ravel())
                targets.append(read_f32(dataset.parent / r['target'], (r['n']*1920,)).numpy())
                start += r['n']
            p,t = np.concatenate(predictions),np.concatenate(targets)
            report[name] = analyze(t.astype(np.float64),p.astype(np.float64),24000)
            from loss import MultiScaleMelLoss
            mel_loss_fn = MultiScaleMelLoss(sample_rate=24000).to(device)
            p_t, t_t = torch.from_numpy(p)[None].to(device), torch.from_numpy(t)[None].to(device)
            mel_l1 = mel_loss_fn(p_t, t_t).item()
            from train import band_ratios
            ratios = band_ratios(p, t)
            report[name].update(wave_mse=float(np.mean((p-t)**2)),
                                mel_l1=mel_l1,
                                clipping_fraction=float(np.mean(np.abs(p)>1)),
                                **ratios)
            write_wav(out/(name+'-student.wav'),p)
            write_wav(out/(name+'-teacher.wav'),t)
    (out/'metrics.json').write_text(json.dumps(dict(weights=str(weights),dataset=str(dataset),
        split=split,utterances=report),indent=2)+'\n')
    print(json.dumps({n:{k:v[k] for k in ('wave_mse','mel_l1','correlation','rms_ratio','ratio_500_2000','ratio_2000_4000')} for n,v in report.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--split',choices=('train','validation','test'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    evaluate(args.weights,args.dataset,args.split,args.out,args.device)
