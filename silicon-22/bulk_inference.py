import os
import sys
import numpy as np
import tensorflow as tf
import soundfile as sf
from tqdm import tqdm
from preprocess import AFGPreprocessor
from model import create_model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.h5')
STATS_PATH = os.path.join(BASE_DIR, 'normalization_stats.npy')
CLASSES = ['baby_cry', 'dog', 'unknown']

def run_bulk_inference(target_dir):
    if not os.path.exists(target_dir):
        print(f' Directory not found: {target_dir}')
        return
    print('\n' + '=' * 75)
    print(f'      SILICON-21: PRODUCTION PCEN BULK INFERENCE (CryNet)')
    print(f'      Mode: Stateful PCEN + Master Policy V1')
    print('=' * 75)
    model = create_model(input_shape=(64, 198, 1), num_classes=3)
    _ = model(np.zeros((1, 64, 198, 1)))
    if os.path.exists(MODEL_PATH):
        model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
        print(' Model loaded.')
    else:
        print(' Warning: Model weights not found.')
    if os.path.exists(STATS_PATH):
        stats = np.load(STATS_PATH, allow_pickle=True).item()
        mean_val = stats['mean'].item() if hasattr(stats['mean'], 'item') else stats['mean']
        std_val = stats['std'].item() if hasattr(stats['std'], 'item') else stats['std']
        print(' PCEN Normalization Stats Loaded.')
    else:
        mean_val, std_val = (0, 1)
        print(' Warning: Normalization stats not found. Using defaults.')
    prep = AFGPreprocessor()
    thresholds = {'baby_cry': 0.65, 'dog': 0.9}
    required_streaks = {'baby_cry': 2, 'dog': 4}
    decay_rates = {'baby_cry': 1.0, 'dog': 0.3}
    baby_suppression_floor = 0.35
    window_samples = 32000
    step_samples = 4000
    audio_extensions = ('.wav', '.mp3', '.ogg', '.m4a')
    all_files = []
    for root, _, files in os.walk(target_dir):
        for f in files:
            if f.lower().endswith(audio_extensions):
                all_files.append(os.path.join(root, f))
    print(f' Found {len(all_files)} audio files. Analyzing...\n')
    summary = {'total': 0, 'detected_baby': 0, 'detected_dog': 0, 'errors': 0}
    for f_path in tqdm(all_files, desc='Processing'):
        try:
            summary['total'] += 1
            y, sr = sf.read(f_path, always_2d=False)
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            y = prep._manual_resample(y, sr, 16000)
            prep.reset_pcen_state()
            streaks = {'baby_cry': 0.0, 'dog': 0.0}
            triggered = {'baby_cry': False, 'dog': False}
            for start in range(0, len(y) - window_samples + 1, step_samples):
                window = y[start:start + window_samples]
                rms = np.sqrt(np.mean(window ** 2))
                if rms < 0.002:
                    prep.reset_pcen_state()
                    for k in streaks:
                        streaks[k] = max(0, streaks[k] - decay_rates[k])
                    continue
                frame_05s = window[-8000:]
                env = np.sqrt(np.mean(frame_05s.reshape(-1, 400) ** 2, axis=1))
                slope = np.mean(np.diff(env))
                if slope < 0.0005:
                    streaks['dog'] = max(0, streaks['dog'] - decay_rates['dog'])
                feat = prep.extract_features(raw_audio=window, is_inference=True)
                if feat is None:
                    for k in streaks:
                        streaks[k] = max(0, streaks[k] - decay_rates[k])
                    continue
                feat_3d = feat[..., np.newaxis]
                feat_norm = (feat_3d - mean_val) / (std_val + 1e-09)
                probs = model.predict(feat_norm[np.newaxis, ...], verbose=0)[0]
                baby_prob = probs[0]
                dog_prob = probs[1]
                if max(baby_prob, dog_prob) < 0.7:
                    for k in streaks:
                        streaks[k] = max(0, streaks[k] - decay_rates[k])
                elif baby_prob >= thresholds['baby_cry']:
                    streaks['baby_cry'] += 1
                    streaks['dog'] = max(0, streaks['dog'] - decay_rates['dog'])
                elif dog_prob >= thresholds['dog'] and dog_prob - baby_prob > 0.15 and (baby_prob < baby_suppression_floor):
                    streaks['dog'] += 1
                    streaks['baby_cry'] = max(0, streaks['baby_cry'] - decay_rates['baby_cry'])
                else:
                    for k in streaks:
                        streaks[k] = max(0, streaks[k] - decay_rates[k])
                if streaks['baby_cry'] >= required_streaks['baby_cry']:
                    triggered['baby_cry'] = True
                elif streaks['dog'] >= required_streaks['dog']:
                    triggered['dog'] = True
            rel_path = os.path.relpath(f_path, target_dir)
            if triggered['baby_cry']:
                summary['detected_baby'] += 1
                print(f'\x1b[92m[ CRY] {rel_path:<45}\x1b[0m')
            elif triggered['dog']:
                summary['detected_dog'] += 1
                print(f'\x1b[94m[ DOG] {rel_path:<45}\x1b[0m')
        except Exception:
            summary['errors'] += 1
            continue
    print('\n' + '=' * 75)
    print('               FINAL PCEN BULK REPORT (SILICON-21)')
    print('-' * 75)
    print(f'Total Files Processed   : {summary['total']}')
    print(f'Baby Cry Detections     : {summary['detected_baby']}')
    print(f'Dog Bark Detections     : {summary['detected_dog']}')
    print(f'Preprocessing Errors    : {summary['errors']}')
    print('=' * 75 + '\n')
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 bulk_inference.py <directory_path>')
    else:
        run_bulk_inference(sys.argv[1])