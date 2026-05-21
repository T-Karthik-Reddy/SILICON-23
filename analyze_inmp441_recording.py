









   

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import soundfile as sf


@dataclass
class Metrics:
    sample_rate: int
    channels: int
    duration_sec: float
    bit_depth: str
    rms_dbfs: float
    peak_dbfs: float
    crest_factor_db: float
    dc_offset: float
    clipped_ratio: float
    noise_floor_dbfs: float
    speech_level_dbfs: float
    est_snr_db: float
    zcr: float
    spectral_centroid_hz: float
    spectral_rolloff_95_hz: float
    low_mid_high: tuple


def dbfs(x: float, eps: float = 1e-12) -> float:
    return 20.0 * math.log10(max(x, eps))


def stft_mag(y: np.ndarray, n_fft: int = 2048, hop: int = 512):
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    win = np.hanning(n_fft)
    frames = []
    for i in range(0, len(y) - n_fft + 1, hop):
        x = y[i : i + n_fft] * win
        spec = np.abs(np.fft.rfft(x, n=n_fft))
        frames.append(spec)
    if not frames:
        return np.zeros((1, n_fft // 2 + 1), dtype=np.float64)
    return np.array(frames, dtype=np.float64)


def analyze(path: str) -> Metrics:
    y, sr = sf.read(path, always_2d=False)
    info = sf.info(path)

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    y = y.astype(np.float64)

    if np.max(np.abs(y)) > 1.5:
        y = y / 32768.0

    dur = len(y) / sr if sr > 0 else 0.0

    peak = float(np.max(np.abs(y)) + 1e-12)
    rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    crest = dbfs(peak / max(rms, 1e-12))
    dc = float(np.mean(y))

    clipped_ratio = float(np.mean(np.abs(y) >= 0.995))

    frame_len = int(0.03 * sr)
    hop = int(0.01 * sr)
    if frame_len < 1:
        frame_len = 1
    if hop < 1:
        hop = 1

    frame_rms = []
    zcr_vals = []
    for i in range(0, len(y) - frame_len + 1, hop):
        f = y[i : i + frame_len]
        frame_rms.append(np.sqrt(np.mean(f ** 2) + 1e-12))
        signs = np.signbit(f)
        zcr_vals.append(np.mean(signs[1:] != signs[:-1]))

    if not frame_rms:
        frame_rms = [rms]
        zcr_vals = [0.0]

    frame_rms = np.array(frame_rms)
    zcr = float(np.mean(zcr_vals))

    noise_floor = float(np.percentile(frame_rms, 15))
    speech_level = float(np.percentile(frame_rms, 85))
    est_snr = dbfs(speech_level) - dbfs(noise_floor)

    n_fft = 2048 if sr >= 16000 else 1024
    S = stft_mag(y, n_fft=n_fft, hop=max(1, n_fft // 4))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    mean_spec = np.mean(S, axis=0) + 1e-12
    centroid = float(np.sum(freqs * mean_spec) / np.sum(mean_spec))

    cumsum = np.cumsum(mean_spec)
    idx_95 = int(np.searchsorted(cumsum, 0.95 * cumsum[-1]))
    idx_95 = min(max(idx_95, 0), len(freqs) - 1)
    rolloff_95 = float(freqs[idx_95])

    low = np.mean(mean_spec[(freqs >= 50) & (freqs < 300)])
    mid = np.mean(mean_spec[(freqs >= 300) & (freqs < 3000)])
    high = np.mean(mean_spec[(freqs >= 3000) & (freqs <= 8000)])

    return Metrics(
        sample_rate=sr,
        channels=info.channels,
        duration_sec=dur,
        bit_depth=info.subtype,
        rms_dbfs=dbfs(rms),
        peak_dbfs=dbfs(peak),
        crest_factor_db=crest,
        dc_offset=dc,
        clipped_ratio=clipped_ratio,
        noise_floor_dbfs=dbfs(noise_floor),
        speech_level_dbfs=dbfs(speech_level),
        est_snr_db=est_snr,
        zcr=zcr,
        spectral_centroid_hz=centroid,
        spectral_rolloff_95_hz=rolloff_95,
        low_mid_high=(float(low), float(mid), float(high)),
    )


def print_report(path: str, m: Metrics):
    print("=" * 72)
    print("INMP441 Recording Quality Report")
    print("=" * 72)
    print(f"File                 : {path}")
    print(f"Duration             : {m.duration_sec:.2f} s")
    print(f"Format               : {m.bit_depth}, {m.channels} ch, {m.sample_rate} Hz")
    print()
    print("Level & Integrity")
    print(f"  RMS level          : {m.rms_dbfs:7.2f} dBFS")
    print(f"  Peak level         : {m.peak_dbfs:7.2f} dBFS")
    print(f"  Crest factor       : {m.crest_factor_db:7.2f} dB")
    print(f"  DC offset          : {m.dc_offset:+.6f}")
    print(f"  Clipped samples    : {100*m.clipped_ratio:.4f} %")
    print()
    print("Speech/Noise Estimate")
    print(f"  Noise floor (P15)  : {m.noise_floor_dbfs:7.2f} dBFS")
    print(f"  Speech lvl (P85)   : {m.speech_level_dbfs:7.2f} dBFS")
    print(f"  Estimated SNR      : {m.est_snr_db:7.2f} dB")
    print(f"  Zero-crossing rate : {m.zcr:7.4f}")
    print()
    print("Spectral Profile")
    print(f"  Centroid           : {m.spectral_centroid_hz:7.1f} Hz")
    print(f"  95% rolloff        : {m.spectral_rolloff_95_hz:7.1f} Hz")

    low, mid, high = m.low_mid_high
    low_db = dbfs(low)
    mid_db = dbfs(mid)
    high_db = dbfs(high)
    print(f"  Band energy 50-300 : {low_db:7.2f} dB (relative)")
    print(f"  Band energy 300-3k : {mid_db:7.2f} dB (relative)")
    print(f"  Band energy 3k-8k  : {high_db:7.2f} dB (relative)")

    print()
    print("Deployment Readiness (silicon-23)")
    rec = []
    if m.sample_rate != 16000:
        rec.append("Input is not 16kHz; resample to 16kHz before feature extraction (already done in silicon-23 preprocess).")
    if m.clipped_ratio > 0.001:
        rec.append("Clipping is non-trivial; reduce INMP441 gain/ADC scale.")
    if abs(m.dc_offset) > 0.02:
        rec.append("DC offset is high; keep DC-notch filter enabled.")
    if m.est_snr_db < 12:
        rec.append("Low estimated SNR; improve placement/shielding or raise speech level.")
    if m.peak_dbfs < -12:
        rec.append("Recording is conservative in level; consider slight gain increase.")

    if not rec:
        rec.append("Recording quality looks healthy for silicon-23 inference.")

    for r in rec:
        print(f"  - {r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    args = parser.parse_args()

    path = os.path.abspath(args.audio_path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    m = analyze(path)
    print_report(path, m)


if __name__ == "__main__":
    main()
