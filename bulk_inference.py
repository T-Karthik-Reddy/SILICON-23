

   

import os
import sys
import numpy as np
import soundfile as sf
from tqdm import tqdm
from preprocess import AFGPreprocessor
from model import create_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.keras')
STATS_PATH = os.path.join(BASE_DIR, 'normalization_stats.npy')
AUDIO_EXTS = ('.wav', '.mp3', '.ogg', '.m4a')


def run_bulk_inference(target_dir):
    if not os.path.exists(target_dir):
        print(f"Directory not found: {target_dir}")
        return

    model = create_model(input_shape=(64, 198, 1))
    _ = model(np.zeros((1, 64, 198, 1), dtype=np.float32))
    model.load_weights(MODEL_PATH)

    if os.path.exists(STATS_PATH):
        s = np.load(STATS_PATH, allow_pickle=True).item()
        mean_val, std_val = s['mean'].squeeze(), s['std'].squeeze()
    else:
        mean_val, std_val = 0.0, 1.0

    prep = AFGPreprocessor()
    files = []
    for root, _, fs in os.walk(target_dir):
        for f in fs:
            if f.lower().endswith(AUDIO_EXTS):
                files.append(os.path.join(root, f))

    threshold = 0.80
    summary = {"total": 0, "dog": 0, "not_dog": 0, "errors": 0}

    for path in tqdm(files, desc='Processing'):
        summary['total'] += 1
        try:
            y, sr = sf.read(path, always_2d=False)
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            y = prep._manual_resample(y, sr, 16000)

            window_samples = 32000
            step_samples = 8000
            probs = []
            for start in range(0, len(y) - window_samples + 1, step_samples):
                window = y[start:start + window_samples]
                if np.sqrt(np.mean(window ** 2)) < 0.002:
                    continue

                feat = prep.extract_features(raw_audio=window, is_inference=False)
                if feat is None:
                    continue

                feat = np.squeeze(np.array(feat)).reshape(64, 198)
                feat = feat[:, :, np.newaxis]
                feat = (feat - mean_val) / (std_val + 1e-9)
                probs.append(float(model.predict(feat[np.newaxis, ...], verbose=0)[0][0]))

            pred_dog = any(p >= threshold for p in probs)
            rel = os.path.relpath(path, target_dir)
            if pred_dog:
                summary['dog'] += 1
                print(f"[DOG] {rel}")
            else:
                summary['not_dog'] += 1

        except Exception:
            summary['errors'] += 1

    print("\nSummary")
    print(summary)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 bulk_inference.py <directory_path>")
    else:
        run_bulk_inference(sys.argv[1])
