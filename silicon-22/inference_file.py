import os
import sys
import numpy as np
import tensorflow as tf
import soundfile as sf
from preprocess import AFGPreprocessor
from model import create_model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.h5')
CLASSES = ['baby_cry', 'dog', 'unknown']

def run_file_inference(file_path):
    if not os.path.exists(file_path):
        print(f' File not found: {file_path}')
        return
    print('\n' + '=' * 60)
    print(f'      SILICON-18: FILE INFERENCE ({os.path.basename(file_path)})')
    print('=' * 60)
    model = create_model(input_shape=(64, 198, 1), num_classes=3)
    _ = model(np.zeros((1, 64, 198, 1)))
    if os.path.exists(MODEL_PATH):
        model.load_weights(MODEL_PATH)
        print(' Model loaded.')
    else:
        print(' Warning: Model weights not found.')
    print(' Loading audio...')
    y, sr = sf.read(file_path, always_2d=False)
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    prep = AFGPreprocessor()
    y = prep._manual_resample(y, sr, 16000)
    duration = len(y) / 16000
    print(f' Audio loaded: {duration:.2f}s duration')
    window_samples = 32000
    step_samples = 8000
    threshold = 0.7
    print('\n' + '-' * 60)
    print(f'{'OFFSET (s)':<12} | {'PREDICTION':<15} | {'CONFIDENCE':<10}')
    print('-' * 60)
    for start in range(0, len(y) - window_samples, step_samples):
        window = y[start:start + window_samples]
        feat = prep.extract_features(raw_audio=window)
        if feat is None:
            continue
        feat_input = feat[np.newaxis, ..., np.newaxis]
        probs = model.predict(feat_input, verbose=0)[0]
        pred_idx = np.argmax(probs)
        conf = probs[pred_idx]
        label = CLASSES[pred_idx]
        offset_sec = start / 16000
        if conf > threshold and label != 'unknown':
            meter = '█' * int(conf * 10)
            color = '\x1b[92m' if label == 'baby_cry' else '\x1b[94m'
            icon = '' if label == 'baby_cry' else ''
            desc = 'CRY DETECTED' if label == 'baby_cry' else 'DOG BARKING '
            print(f'{color}{offset_sec:05.1f}s - {offset_sec + 2.0:0.1f}s | {icon} {desc} | {conf:.2%} {meter:<10}\x1b[0m')
        else:
            pass
    print('-' * 60)
    print(' Analysis Complete.')
    print('=' * 60 + '\n')
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 inference_file.py <path_to_audio_file>')
    else:
        run_file_inference(sys.argv[1])