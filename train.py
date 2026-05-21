

   

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import create_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH = os.path.join(BASE_DIR, 'processed_dataset')
CLASSES = ['not_dog_bark', 'dog_bark']
BATCH_SIZE = 32
EPOCHS = 250
INPUT_SHAPE = (64, 198, 1)


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def _load_features_for_class(class_name):
    class_dir = os.path.join(PROCESSED_PATH, class_name)
    if not os.path.exists(class_dir):
        return [], []

    files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.npy')]
    X, y = [], []
    label = 1 if class_name == 'dog_bark' else 0

    print(f"Loading {class_name}: {len(files)} samples")
    for f in files:
        feat = np.load(f)
        feat = np.squeeze(feat)

        if feat.shape != INPUT_SHAPE[:2]:
            if feat.shape[0] == INPUT_SHAPE[0]:
                if feat.shape[1] > INPUT_SHAPE[1]:
                    feat = feat[:, :INPUT_SHAPE[1]]
                else:
                    feat = np.pad(feat, ((0, 0), (0, INPUT_SHAPE[1] - feat.shape[1])), mode='constant')
            else:
                continue

        X.append(feat[..., np.newaxis])
        y.append(label)

    return X, y


def train():
    log_file = os.path.join(BASE_DIR, 'training_log.txt')
    sys.stdout = Logger(log_file)

    print("\n" + "=" * 72)
    print("SILICON-23: BINARY DOG BARK TRAINING")
    print("=" * 72)

    X, y = [], []
    for cls in CLASSES:
        xi, yi = _load_features_for_class(cls)
        X.extend(xi)
        y.extend(yi)

    if not X:
        print("No samples found. Run prepare_data.py first.")
        return

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"Dataset size: {len(X)} | positives(dog): {int(np.sum(y))} | negatives: {int(len(y)-np.sum(y))}")

    mean = X.mean(axis=(0, 1, 2), keepdims=True)
    std = X.std(axis=(0, 1, 2), keepdims=True) + 1e-9
    X = (X - mean) / std
    np.save(os.path.join(BASE_DIR, 'normalization_stats.npy'), {'mean': mean, 'std': std})

    idx = np.arange(len(X))
    idx_train, idx_val = train_test_split(idx, test_size=0.15, stratify=y, random_state=42)
    x_train, x_val = X[idx_train], X[idx_val]
    y_train, y_val = y[idx_train], y[idx_val]

    model = create_model(input_shape=INPUT_SHAPE)
    print(f"Total params: {model.count_params()}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(curve='PR', name='auc_pr'),
        ],
    )

    class_weight = {0: 1.0, 1: 1.6}
    print(f"Class weights: {class_weight}")

    model_path = os.path.join(BASE_DIR, 'best_model.keras')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='val_auc_pr', mode='max', verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=35, restore_best_weights=True, mode='min'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, mode='min', verbose=1),
    ]

    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(hist.history['precision'], label='train_precision')
    plt.plot(hist.history['val_precision'], label='val_precision')
    plt.title('Precision')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(hist.history['recall'], label='train_recall')
    plt.plot(hist.history['val_recall'], label='val_recall')
    plt.title('Recall')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(hist.history['auc_pr'], label='train_auc_pr')
    plt.plot(hist.history['val_auc_pr'], label='val_auc_pr')
    plt.title('PR-AUC')
    plt.grid(True, alpha=0.3)
    plt.legend()

    report_path = os.path.join(BASE_DIR, 'training_report.png')
    plt.tight_layout()
    plt.savefig(report_path)

    print(f"\nTraining complete. Best model: {model_path}")
    print(f"Training report: {report_path}")


if __name__ == '__main__':
    train()
