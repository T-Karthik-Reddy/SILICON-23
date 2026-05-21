



   

import os
import numpy as np
import tensorflow as tf
from model import create_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.keras')
STATS_PATH = os.path.join(BASE_DIR, 'normalization_stats.npy')
PROCESSED_PATH = os.path.join(BASE_DIR, 'processed_dataset')

FLOAT_TFLITE_PATH = os.path.join(BASE_DIR, 'model_float32.tflite')
INT8_TFLITE_PATH = os.path.join(BASE_DIR, 'model_int8.tflite')
SAVED_MODEL_DIR = os.path.join(BASE_DIR, 'export_saved_model')


def _load_model():
    model = create_model(input_shape=(64, 198, 1))
    _ = model(np.zeros((1, 64, 198, 1), dtype=np.float32))
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing trained model: {MODEL_PATH}")
    model.load_weights(MODEL_PATH)
    return model


def _load_norm_stats():
    if not os.path.exists(STATS_PATH):
        raise FileNotFoundError(f"Missing normalization stats: {STATS_PATH}")
    stats = np.load(STATS_PATH, allow_pickle=True).item()
    return stats['mean'].squeeze(), stats['std'].squeeze()


def _iter_feature_files(max_per_class=300):
    for cls in ('dog_bark', 'not_dog_bark'):
        cls_dir = os.path.join(PROCESSED_PATH, cls)
        if not os.path.exists(cls_dir):
            continue
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith('.npy')]
        files = sorted(files)[:max_per_class]
        for f in files:
            yield f


def representative_dataset():
    mean_val, std_val = _load_norm_stats()
    for f in _iter_feature_files(max_per_class=300):
        feat = np.load(f)
        feat = np.squeeze(feat)
        try:
            feat = feat.reshape(64, 198)
        except ValueError:
            continue
        feat = feat[:, :, np.newaxis]
        feat = (feat - mean_val) / (std_val + 1e-9)
        feat = feat.astype(np.float32)
        yield [feat[np.newaxis, ...]]


def export_float32(model):
    model.export(SAVED_MODEL_DIR)
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    tflite_model = converter.convert()
    with open(FLOAT_TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved float32 TFLite: {FLOAT_TFLITE_PATH}")


def export_int8(model):
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(INT8_TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved int8 TFLite: {INT8_TFLITE_PATH}")


def main():
    model = _load_model()
    export_float32(model)
    export_int8(model)


if __name__ == '__main__':
    main()
