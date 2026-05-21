import os
import numpy as np
from tqdm import tqdm
from preprocess import AFGPreprocessor
import soundfile as sf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'Dataset')
PROCESSED_PATH = os.path.join(BASE_DIR, 'processed_dataset')
CLASSES = ['baby_cry', 'dog', 'unknown']

def prepare():
    print('\n' + '=' * 60)
    print('      SILICON-22: PCEN DATA PREPARATION (V1)')
    print('=' * 60)
    prep = AFGPreprocessor()
    if not os.path.exists(PROCESSED_PATH):
        os.makedirs(PROCESSED_PATH)
    stats = {cls: {'kept': 0, 'rejected': 0} for cls in CLASSES}
    for label in CLASSES:
        source_dir = os.path.join(DATASET_PATH, label)
        if not os.path.exists(source_dir):
            print(f' Warning: {source_dir} not found. Skipping...')
            continue
        output_dir = os.path.join(PROCESSED_PATH, label)
        os.makedirs(output_dir, exist_ok=True)
        files = [f for f in os.listdir(source_dir) if f.endswith(('.wav', '.mp3', '.ogg', '.m4a'))]
        print(f'\n Checking {label} signal quality ({len(files)} source files)...')
        for f_name in tqdm(files):
            f_path = os.path.join(source_dir, f_name)
            f_stem = os.path.splitext(f_name)[0]
            try:
                info = sf.info(f_path)
                num_windows = 1
                results = prep.extract_features(audio_path=f_path, label=label, return_audio=True)
                if results is None:
                    stats[label]['rejected'] += 1
                    continue
                valid_windows_in_file = 0
                for i, (feat, audio) in enumerate(results):
                    valid_windows_in_file += 1
                    suffix = f'_{i}'
                    np.save(os.path.join(output_dir, f'{f_stem}{suffix}_clean.npy'), feat)
                    sf.write(os.path.join(output_dir, f'{f_stem}{suffix}_clean.wav'), audio, 16000)
                    for aug_idx in range(2):
                        aug_results = prep.extract_features(audio_path=f_path, label=label, augment=True, return_audio=True)
                        if aug_results:
                            feat_aug, audio_aug = aug_results[0]
                            np.save(os.path.join(output_dir, f'{f_stem}{suffix}_aug_{aug_idx}.npy'), feat_aug)
                            sf.write(os.path.join(output_dir, f'{f_stem}{suffix}_aug_{aug_idx}.wav'), audio_aug, 16000)
                if valid_windows_in_file > 0:
                    stats[label]['kept'] += 1
                else:
                    stats[label]['rejected'] += 1
            except Exception:
                stats[label]['rejected'] += 1
                continue
    print('\n' + '-' * 60)
    print('               PREPROCESSING RESULTS')
    print('-' * 60)
    for cls in CLASSES:
        if cls in stats:
            print(f'[{cls.upper()}] Kept: {stats[cls]['kept']} | Rejected: {stats[cls]['rejected']} (Low Signal)')
    print(f'\n Clean data in: {PROCESSED_PATH}')
    print('=' * 60 + '\n')
if __name__ == '__main__':
    prepare()