





   

import os
import shutil
import numpy as np
from tqdm import tqdm
import soundfile as sf
from preprocess import AFGPreprocessor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'Dataset')
PROCESSED_PATH = os.path.join(BASE_DIR, 'processed_dataset')


OUT_POS = 'dog_bark'
OUT_NEG = 'not_dog_bark'


SRC_DOG = 'dog'
SRC_BABY = 'baby_cry'
SRC_UNKNOWN = 'unknown'
AUDIO_EXTS = ('.wav', '.mp3', '.ogg', '.m4a')


TARGET_DOG = 4400
TARGET_BABY = 2600
TARGET_UNKNOWN = 2600
RNG_SEED = 42


def _list_audio_files(src_dir):
    if not os.path.exists(src_dir):
        return []
    return [f for f in os.listdir(src_dir) if f.lower().endswith(AUDIO_EXTS)]


def _prepare_out_dirs():
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    for cls in (OUT_POS, OUT_NEG):
        cls_dir = os.path.join(PROCESSED_PATH, cls)
        if os.path.exists(cls_dir):
            shutil.rmtree(cls_dir)
        os.makedirs(cls_dir, exist_ok=True)


def _save_feature_and_audio(output_dir, stem, suffix, feat, audio):
    np.save(os.path.join(output_dir, f"{stem}{suffix}.npy"), feat)
    sf.write(os.path.join(output_dir, f"{stem}{suffix}.wav"), audio, 16000)


def _extract_and_store(prep, src_label, out_label, file_list, max_kept_samples=None):
    src_dir = os.path.join(DATASET_PATH, src_label)
    out_dir = os.path.join(PROCESSED_PATH, out_label)

    kept_files = 0
    rejected_files = 0
    saved_samples = 0

    for f_name in tqdm(file_list, desc=f"{src_label} -> {out_label}"):
        if max_kept_samples is not None and saved_samples >= max_kept_samples:
            break

        f_path = os.path.join(src_dir, f_name)
        f_stem = os.path.splitext(f_name)[0]

        try:
            results = prep.extract_features(audio_path=f_path, label=src_label, return_audio=True)
            if results is None:
                rejected_files += 1
                continue

            file_saved = 0
            for i, (feat, audio) in enumerate(results):
                if max_kept_samples is not None and saved_samples >= max_kept_samples:
                    break

                suffix = f"_{src_label}_{i}_clean"
                _save_feature_and_audio(out_dir, f_stem, suffix, feat, audio)
                saved_samples += 1
                file_saved += 1


                for aug_idx in range(2):
                    if max_kept_samples is not None and saved_samples >= max_kept_samples:
                        break
                    aug_results = prep.extract_features(
                        audio_path=f_path,
                        label=src_label,
                        augment=True,
                        return_audio=True,
                    )
                    if aug_results:
                        feat_aug, audio_aug = aug_results[0]
                        aug_suffix = f"_{src_label}_{i}_aug_{aug_idx}"
                        _save_feature_and_audio(out_dir, f_stem, aug_suffix, feat_aug, audio_aug)
                        saved_samples += 1

            if file_saved > 0:
                kept_files += 1
            else:
                rejected_files += 1

        except Exception:
            rejected_files += 1

    return {
        'kept_files': kept_files,
        'rejected_files': rejected_files,
        'saved_samples': saved_samples,
    }


def prepare():
    print("\n" + "=" * 72)
    print("   SILICON-23: BINARY PREPARATION (DOG_BARK vs NOT_DOG_BARK)")
    print("=" * 72)

    _prepare_out_dirs()
    prep = AFGPreprocessor()

    dog_files = _list_audio_files(os.path.join(DATASET_PATH, SRC_DOG))
    baby_files = _list_audio_files(os.path.join(DATASET_PATH, SRC_BABY))
    unknown_files = _list_audio_files(os.path.join(DATASET_PATH, SRC_UNKNOWN))

    rng = np.random.default_rng(RNG_SEED)
    rng.shuffle(dog_files)
    rng.shuffle(baby_files)
    rng.shuffle(unknown_files)

    print(f"Source files - dog: {len(dog_files)}, baby: {len(baby_files)}, unknown: {len(unknown_files)}")
    print(
        f"Target samples - dog: {TARGET_DOG}, baby->neg: {TARGET_BABY}, unknown->neg: {TARGET_UNKNOWN}"
    )

    dog_stats = _extract_and_store(
        prep=prep,
        src_label=SRC_DOG,
        out_label=OUT_POS,
        file_list=dog_files,
        max_kept_samples=TARGET_DOG,
    )

    baby_stats = _extract_and_store(
        prep=prep,
        src_label=SRC_BABY,
        out_label=OUT_NEG,
        file_list=baby_files,
        max_kept_samples=TARGET_BABY,
    )

    unknown_stats = _extract_and_store(
        prep=prep,
        src_label=SRC_UNKNOWN,
        out_label=OUT_NEG,
        file_list=unknown_files,
        max_kept_samples=TARGET_UNKNOWN,
    )

    print("\n" + "-" * 72)
    print("Summary")
    print("-" * 72)
    print(f"DOG -> {OUT_POS}: kept_files={dog_stats['kept_files']}, rejected={dog_stats['rejected_files']}, samples={dog_stats['saved_samples']}")
    print(f"BABY -> {OUT_NEG}: kept_files={baby_stats['kept_files']}, rejected={baby_stats['rejected_files']}, samples={baby_stats['saved_samples']}")
    print(f"UNKNOWN -> {OUT_NEG}: kept_files={unknown_stats['kept_files']}, rejected={unknown_stats['rejected_files']}, samples={unknown_stats['saved_samples']}")

    neg_total = baby_stats['saved_samples'] + unknown_stats['saved_samples']
    print(f"\nFinal binary samples: {OUT_POS}={dog_stats['saved_samples']}, {OUT_NEG}={neg_total}")
    print(f"Output directory: {PROCESSED_PATH}")
    print("=" * 72 + "\n")


if __name__ == '__main__':
    prepare()
