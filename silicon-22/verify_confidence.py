import os
import numpy as np
import tensorflow as tf
from model import create_model
from tqdm import tqdm
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH = os.path.join(BASE_DIR, 'processed_dataset')
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.keras')
STATS_PATH = os.path.join(BASE_DIR, 'normalization_stats.npy')
CLASSES = ['baby_cry', 'dog', 'unknown']

def run_validation():
    print('\n' + '=' * 70)
    print('      SILICON-22: CRYNET CONFIDENCE & POLICY VALIDATOR')
    print('=' * 70)
    model = create_model(input_shape=(64, 198, 1), num_classes=3)
    _ = model(np.zeros((1, 64, 198, 1)))
    if os.path.exists(MODEL_PATH):
        model.load_weights(MODEL_PATH)
        print(' SILICON-22 Model loaded.')
    else:
        print(' Warning: Model weights not found.')
        return
    if os.path.exists(STATS_PATH):
        stats = np.load(STATS_PATH, allow_pickle=True).item()
        mean_val = stats['mean'].squeeze() if hasattr(stats['mean'], 'squeeze') else stats['mean']
        std_val = stats['std'].squeeze() if hasattr(stats['std'], 'squeeze') else stats['std']
        print(' PCEN Normalization Stats Loaded.')
    else:
        mean_val, std_val = (0, 1)
        print(' Warning: Normalization stats not found. Using defaults.')
    results = {label: [] for label in CLASSES}
    for idx, label in enumerate(CLASSES):
        class_dir = os.path.join(PROCESSED_PATH, label)
        if not os.path.exists(class_dir):
            continue
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.npy')]
        if not files:
            continue
        subset_count = min(500, len(files))
        subset_files = np.random.choice(files, subset_count, replace=False)
        print(f' Analyzing {label} ({subset_count} samples)...')
        for f in tqdm(subset_files):
            feat = np.load(f)
            feat = np.squeeze(feat)
            try:
                feat = feat.reshape(64, 198)
            except ValueError:
                continue
            feat_3d = feat[:, :, np.newaxis]
            feat_norm = (feat_3d - mean_val) / (std_val + 1e-09)
            probs = model.predict(feat_norm[np.newaxis, ...], verbose=0)[0]
            results[label].append(probs)
    print('\n' + '-' * 70)
    print('               RAW MODEL PROBABILITY ANALYSIS')
    print('-' * 70)
    thresholds = [0.65, 0.8, 0.9]
    for t in thresholds:
        print(f'\n[ THRESHOLD: {t} ]')
        for target_idx, target_label in enumerate(['baby_cry', 'dog']):
            tp = sum((1 for p in results[target_label] if p[target_idx] >= t))
            recall = tp / len(results[target_label])
            fp_unknown = sum((1 for p in results['unknown'] if p[target_idx] >= t))
            other_label = 'dog' if target_label == 'baby_cry' else 'baby_cry'
            fp_other = sum((1 for p in results[other_label] if p[target_idx] >= t))
            precision = tp / (tp + fp_unknown + fp_other) if tp + fp_unknown + fp_other > 0 else 0
            print(f'   {target_label.upper():<10} | Precision: {precision:.2%} | Recall: {recall:.2%}')
    print('\n' + '-' * 70)
    print('               MASTER POLICY V1 SIMULATION')
    print(' (Margin: 0.15 | Baby Suppression: 0.35 | Thresholds: B 0.65, D 0.90)')
    print('-' * 70)
    tp_b = sum((1 for p in results['baby_cry'] if p[0] >= 0.65))
    fp_b = sum((1 for p in results['dog'] if p[0] >= 0.65)) + sum((1 for p in results['unknown'] if p[0] >= 0.65))

    def dog_policy(p):
        return p[1] >= 0.9 and p[1] - p[0] > 0.15 and (p[0] < 0.35)
    tp_d = sum((1 for p in results['dog'] if dog_policy(p)))
    fp_d = sum((1 for p in results['baby_cry'] if dog_policy(p))) + sum((1 for p in results['unknown'] if dog_policy(p)))
    print(f'   BABY CRY  | Precision: {tp_b / (tp_b + fp_b):.2%} | Recall: {tp_b / len(results['baby_cry']):.2%}')
    print(f'   DOG BARK  | Precision: {tp_d / (tp_d + fp_d):.2%} | Recall: {tp_d / len(results['dog']):.2%}')
    print('\n' + '=' * 70)
    print(' SILICON-22 Insight:')
    print("   The Master Policy V1 combined with Silicon-22's MVP-optimized")
    print('   architecture is designed to keep Dog Precision > 98% on-device.')
    print('=' * 70 + '\n')
if __name__ == '__main__':
    run_validation()