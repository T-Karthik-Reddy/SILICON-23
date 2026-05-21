

   

import os
import sys
import numpy as np
import soundfile as sf
from preprocess import AFGPreprocessor
from model import create_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.keras')
STATS_PATH = os.path.join(BASE_DIR, 'normalization_stats.npy')


def run_file_inference(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
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
    y, sr = sf.read(file_path, always_2d=False)
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    y = prep._manual_resample(y, sr, 16000)

    window_samples = 32000
    step_samples = 8000
    threshold = 0.80

    print("offset_s | dog_prob | decision")
    for start in range(0, len(y) - window_samples + 1, step_samples):
        window = y[start:start + window_samples]
        rms = np.sqrt(np.mean(window ** 2))
        if rms < 0.002:
            continue

        feat = prep.extract_features(raw_audio=window, is_inference=False)
        if feat is None:
            continue

        feat = np.squeeze(np.array(feat)).reshape(64, 198)
        feat = feat[:, :, np.newaxis]
        feat = (feat - mean_val) / (std_val + 1e-9)

        prob = float(model.predict(feat[np.newaxis, ...], verbose=0)[0][0])
        decision = "DOG" if prob >= threshold else "NOT_DOG"
        print(f"{start/16000:7.2f} | {prob:8.3f} | {decision}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 inference_file.py <path_to_audio_file>")
    else:
        run_file_inference(sys.argv[1])
