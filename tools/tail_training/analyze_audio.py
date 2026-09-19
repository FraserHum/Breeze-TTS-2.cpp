"""Compare aligned mono PCM16 pairs; diagnostics are not perceptual quality scores.

Usage: python analyze_audio.py TEACHER.wav STUDENT.wav
Positive best_lag_samples means the student is delayed. Lag search is bounded to
10 ms: periodic speech can have several peaks, so lag rescue is not proof of a
timing defect. PCM metrics cannot recover clipping hidden during WAV export.
"""
import argparse
import json
import wave

import numpy as np


def read_wav(path):
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("Expected mono PCM16 WAV")
        rate = wav.getframerate()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    return rate, samples.astype(np.float64) / 32768


def correlation(a, b):
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator else None


def lag_search(teacher, student, limit):
    # FFT cross-correlation finds the candidate; report normalized correlation
    # on the corresponding overlap, retaining raw aligned metrics separately.
    size = 1 << (2 * len(teacher) - 1).bit_length()
    t, p = teacher - teacher.mean(), student - student.mean()
    cross = np.fft.irfft(np.fft.rfft(p, size) * np.conj(np.fft.rfft(t, size)), size)
    lags = np.arange(-limit, limit + 1)
    lag = int(lags[np.argmax(cross[lags % size])])
    a, b = (teacher[:-lag], student[lag:]) if lag > 0 else (
        (teacher[-lag:], student[:lag]) if lag < 0 else (teacher, student))
    return lag, correlation(a, b)


def analyze(teacher, student, rate):
    if teacher.shape != student.shape or teacher.ndim != 1 or len(teacher) < rate // 10:
        raise ValueError("Require matching waveforms of at least 100 ms")
    if rate != 24000 or not np.isfinite(teacher).all() or not np.isfinite(student).all():
        raise ValueError("Expected finite 24 kHz teacher/student waveforms")
    error = student - teacher
    lag, rescued = lag_search(teacher, student, rate // 100)
    rms = lambda x: float(np.sqrt(np.mean(x * x)))
    t_rms, p_rms = rms(teacher), rms(student)
    result = dict(samples=len(teacher), sample_rate=rate, wave_l1=float(np.abs(error).mean()),
                  error_rms=rms(error), teacher_rms=t_rms, student_rms=p_rms,
                  rms_ratio=p_rms / t_rms if t_rms else None,
                  correlation=correlation(teacher, student), best_lag_samples=lag,
                  best_lag_correlation=rescued)
    # Index b is the first sample of a new frame; np.diff at b-1 crosses it.
    boundaries = np.arange(1920, len(teacher), 1920)
    chunks = boundaries[boundaries % (12 * 1920) == 0]
    frames = boundaries[boundaries % (12 * 1920) != 0]
    derivative_error = np.diff(error)
    interior = np.ones(len(derivative_error), dtype=bool)
    for b in boundaries:
        interior[max(0, b - 25):min(len(interior), b + 24)] = False
    result["interior_derivative_error_mae"] = float(np.abs(derivative_error[interior]).mean())
    for label, positions in (("frame_80ms", frames), ("chunk_960ms", chunks)):
        values = np.abs(derivative_error[positions - 1])
        result[label] = dict(count=len(positions), derivative_error_mae=float(values.mean()) if len(values) else None,
                             derivative_error_max=float(values.max()) if len(values) else None)
    return result


def self_check():
    rng = np.random.default_rng(42)
    target = rng.normal(0, 0.1, 48000)
    exact = analyze(target, target, 24000)
    assert exact["wave_l1"] == 0 and exact["correlation"] > 0.999999
    for shift in (-37, 37):
        shifted = np.zeros_like(target)
        if shift > 0:
            shifted[shift:] = target[:-shift]
        else:
            shifted[:shift] = target[-shift:]
        lag, corr = lag_search(target, shifted, 240)
        assert lag == shift and corr > 0.999999
    broken = target.copy()
    broken[1920:] += 0.1
    result = analyze(target, broken, 24000)
    assert result["frame_80ms"]["derivative_error_max"] > 0.099
    assert result["interior_derivative_error_mae"] < 1e-15
    print("Identity, known signed shifts and injected frame seam: passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("teacher", nargs="?")
    parser.add_argument("student", nargs="?")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        if not args.teacher or not args.student:
            parser.error("Teacher and student WAVs are required")
        rate, teacher = read_wav(args.teacher)
        student_rate, student = read_wav(args.student)
        if rate != student_rate:
            raise ValueError("Sample rates differ")
        print(json.dumps(analyze(teacher, student, rate), indent=2, allow_nan=False))
