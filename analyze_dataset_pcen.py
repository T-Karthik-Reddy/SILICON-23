




   

import os
import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample_poly

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH = os.path.join(BASE_DIR, 'processed_dataset')
CLASSES = ['dog_bark', 'not_dog_bark']
OUTPUT_CSV = os.path.join(BASE_DIR, 'pcen_analysis_report_binary.csv')

SAMPLE_RATE = 16000
WINDOW_SIZE_MS = 30
WINDOW_STEP_MS = 10
N_MELS = 64
AUDIO_EXTS = ('.wav', '.ogg', '.mp3', '.m4a')


def get_mel_filters(sr, n_fft, n_mels):
    f_min = 0
    f_max = sr / 2
    mel_min = 2595 * np.log10(1 + f_min / 700)
    mel_max = 2595 * np.log10(1 + f_max / 700)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        for j in range(bin_points[i - 1], bin_points[i]):
            denom = max(1, (bin_points[i] - bin_points[i - 1]))
            filters[i - 1, j] = (j - bin_points[i - 1]) / denom
        for j in range(bin_points[i], bin_points[i + 1]):
            denom = max(1, (bin_points[i + 1] - bin_points[i]))
            filters[i - 1, j] = (bin_points[i + 1] - j) / denom
    return filters


def infer_source_group(filename, class_label):
    if class_label == 'dog_bark':
        return 'dog'
    name = filename.lower()
    if '_baby_cry_' in name:
        return 'baby_cry'
    if '_unknown_' in name:
        return 'unknown'
    return 'other_neg'


def analyze_file(file_path, class_label):
    try:
        y, sr = sf.read(file_path, always_2d=False)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        duration = len(y) / sr if sr > 0 else 0
        if sr != SAMPLE_RATE:
            y = resample_poly(y, SAMPLE_RATE, sr)
            sr = SAMPLE_RATE

        win_len = int(WINDOW_SIZE_MS * sr / 1000)
        hop_len = int(WINDOW_STEP_MS * sr / 1000)

        frames = []
        for i in range(0, len(y) - win_len + 1, hop_len):
            frames.append(y[i:i + win_len])
        frames = np.array(frames)
        if len(frames) == 0:
            return None

        frame_energies = np.sqrt(np.mean(frames ** 2, axis=1))
        frame_energies = np.maximum(frame_energies, 1e-10)

        mean_energy = np.mean(frame_energies)
        std_energy = np.std(frame_energies)
        max_energy = np.max(frame_energies)
        min_energy = np.min(frame_energies)
        median_energy = np.median(frame_energies)
        energy_cv = std_energy / mean_energy if mean_energy > 0 else 0

        energy_thresh = np.percentile(frame_energies, 20)
        active_mask = frame_energies > energy_thresh
        active_ratio = np.mean(active_mask)
        silence_ratio = 1.0 - active_ratio

        peak = max_energy
        thresh_low = 0.05 * peak
        thresh_high = 0.95 * peak

        rise_times = []
        decay_times = []
        for i in range(1, len(frame_energies)):
            if frame_energies[i - 1] < thresh_low and frame_energies[i] >= thresh_low:
                for j in range(i, len(frame_energies)):
                    if frame_energies[j] >= thresh_high:
                        rise_times.append((j - i) * WINDOW_STEP_MS / 1000)
                        break
            if frame_energies[i - 1] > thresh_high and frame_energies[i] <= thresh_high:
                for j in range(i, len(frame_energies)):
                    if frame_energies[j] <= thresh_low:
                        decay_times.append((j - i) * WINDOW_STEP_MS / 1000)
                        break

        mean_rise = np.mean(rise_times) if rise_times else 0
        mean_decay = np.mean(decay_times) if decay_times else 0

        n_fft = 512
        mel_basis = get_mel_filters(sr, n_fft, N_MELS)

        mel_spectrogram = []
        for frame in frames:
            spec = np.abs(np.fft.rfft(frame * np.hanning(len(frame)), n=n_fft))
            mel_spec = np.dot(mel_basis, spec)
            mel_spectrogram.append(mel_spec)
        mel_spectrogram = np.array(mel_spectrogram)

        mel_means = np.mean(mel_spectrogram, axis=0)
        mel_stds = np.std(mel_spectrogram, axis=0)

        diffs = np.diff(mel_spectrogram, axis=0)
        rise_rates = np.mean(np.maximum(0, diffs), axis=0)
        fall_rates = np.mean(np.maximum(0, -diffs), axis=0)
        smoothness = np.var(diffs, axis=0)

        low_energy = np.sum(mel_spectrogram[:, :N_MELS // 2], axis=1)
        high_energy = np.sum(mel_spectrogram[:, N_MELS // 2:], axis=1)
        ratio_h_l = np.mean(high_energy / (low_energy + 1e-10))

        mel_min = 2595 * np.log10(1 + 0 / 700)
        mel_max = 2595 * np.log10(1 + (sr / 2) / 700)
        mel_pts = np.linspace(mel_min, mel_max, N_MELS + 2)
        mel_freqs = 700 * (10 ** (mel_pts / 2595) - 1)[1:-1]
        brightness = np.mean(np.dot(mel_spectrogram, mel_freqs) / (np.sum(mel_spectrogram, axis=1) + 1e-10))

        stats = {
            'binary_label': class_label,
            'source_group': infer_source_group(os.path.basename(file_path), class_label),
            'filename': os.path.basename(file_path),
            'duration_sec': duration,
            'sample_rate': sr,
            'mean_frame_energy': mean_energy,
            'std_frame_energy': std_energy,
            'median_frame_energy': median_energy,
            'max_frame_energy': max_energy,
            'min_frame_energy': min_energy,
            'frame_energy_cv': energy_cv,
            'avg_active_ratio': active_ratio,
            'avg_silence_ratio': silence_ratio,
            'mean_rise_time': mean_rise,
            'mean_decay_time': mean_decay,
            'mean_energy_ratio_h_l': ratio_h_l,
            'spectral_brightness': brightness,
            'avg_rise_rate_all_bands': np.mean(rise_rates),
            'avg_fall_rate_all_bands': np.mean(fall_rates),
            'avg_smoothness_all_bands': np.mean(smoothness),
        }

        for b in [0, 16, 32, 48, 63]:
            stats[f'mel_band_mean_{b}'] = mel_means[b]
            stats[f'mel_band_std_{b}'] = mel_stds[b]
            stats[f'mel_band_rise_{b}'] = rise_rates[b]
            stats[f'mel_band_fall_{b}'] = fall_rates[b]

        return stats

    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None


def analyze():
    print("\n" + "=" * 72)
    print("SILICON-23: BINARY PCEN DATASET ANALYZER")
    print("=" * 72)

    results = []
    for label in CLASSES:
        class_dir = os.path.join(PROCESSED_PATH, label)
        if not os.path.exists(class_dir):
            print(f"Missing class directory: {class_dir}")
            continue

        files = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith(AUDIO_EXTS)
        ]

        print(f"Analyzing {label}: {len(files)} files")
        for f in tqdm(files):
            stats = analyze_file(f, label)
            if stats:
                results.append(stats)

    if not results:
        print("No files analyzed.")
        return

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved report: {OUTPUT_CSV}")

    print("\nSummary by binary label:")
    print(
        df.groupby('binary_label')[
            ['mean_frame_energy', 'frame_energy_cv', 'spectral_brightness', 'mean_rise_time']
        ].mean()
    )

    print("\nNegative source mix (not_dog_bark):")
    neg = df[df['binary_label'] == 'not_dog_bark']
    if len(neg) > 0:
        mix = neg['source_group'].value_counts()
        total = mix.sum()
        for k, v in mix.items():
            print(f"- {k}: {v} ({100.0 * v / total:.1f}%)")
    else:
        print("- No negative files found in processed_dataset/not_dog_bark")


if __name__ == '__main__':
    analyze()
