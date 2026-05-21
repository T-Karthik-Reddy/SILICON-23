



   

import os
import numpy as np
from tqdm import tqdm
from model import create_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH = os.path.join(BASE_DIR, 'processed_dataset')
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.keras')
STATS_PATH = os.path.join(BASE_DIR, 'normalization_stats.npy')

POS_CLASS = 'dog_bark'
NEG_CLASS = 'not_dog_bark'


def _load_stats():
    if not os.path.exists(STATS_PATH):
        return 0.0, 1.0
    s = np.load(STATS_PATH, allow_pickle=True).item()
    return s['mean'].squeeze(), s['std'].squeeze()


def _predict_probs(model, class_name, mean_val, std_val, max_samples=2000):
    class_dir = os.path.join(PROCESSED_PATH, class_name)
    if not os.path.exists(class_dir):
        return []

    files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.npy')]
    if not files:
        return []

    rng = np.random.default_rng(42)
    if len(files) > max_samples:
        files = list(rng.choice(files, size=max_samples, replace=False))

    probs = []
    for f in tqdm(files, desc=f"Scoring {class_name}"):
        feat = np.load(f)
        feat = np.squeeze(feat)
        try:
            feat = feat.reshape(64, 198)
        except ValueError:
            continue

        feat = feat[:, :, np.newaxis]
        feat = (feat - mean_val) / (std_val + 1e-9)
        p = float(model.predict(feat[np.newaxis, ...], verbose=0)[0][0])
        probs.append(p)

    return probs


def run_validation():
    print("\n" + "=" * 72)
    print("SILICON-23: BINARY CONFIDENCE VALIDATOR")
    print("=" * 72)

    model = create_model(input_shape=(64, 198, 1))
    _ = model(np.zeros((1, 64, 198, 1), dtype=np.float32))
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return
    model.load_weights(MODEL_PATH)

    mean_val, std_val = _load_stats()
    pos_probs = _predict_probs(model, POS_CLASS, mean_val, std_val)
    neg_probs = _predict_probs(model, NEG_CLASS, mean_val, std_val)

    if not pos_probs or not neg_probs:
        print("Missing scored samples. Run prepare_data.py and train.py first.")
        return

    print(f"Samples scored -> {POS_CLASS}: {len(pos_probs)}, {NEG_CLASS}: {len(neg_probs)}")

    print("\nThreshold sweep:")
    print("thr | precision | recall | fpr")
    print("-" * 36)

    best = None
    for thr in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        tp = sum(1 for p in pos_probs if p >= thr)
        fn = len(pos_probs) - tp
        fp = sum(1 for p in neg_probs if p >= thr)
        tn = len(neg_probs) - fp

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        print(f"{thr:.2f} | {precision:9.2%} | {recall:6.2%} | {fpr:5.2%}")

        if precision >= 0.98 and recall >= 0.95:
            if best is None or fpr < best['fpr']:
                best = {'thr': thr, 'precision': precision, 'recall': recall, 'fpr': fpr}

    print("\nTarget gate: Precision > 98%, Recall > 95%")
    if best:
        print(f"Candidate threshold: {best['thr']:.2f} (precision={best['precision']:.2%}, recall={best['recall']:.2%}, fpr={best['fpr']:.2%})")
    else:
        print("No threshold met target on this dataset split. Keep hard-negative tuning and policy tuning.")


if __name__ == '__main__':
    run_validation()
