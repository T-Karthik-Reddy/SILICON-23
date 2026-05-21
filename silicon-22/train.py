import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import create_model
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH = os.path.join(BASE_DIR, 'processed_dataset')
CLASSES = ['baby_cry', 'dog', 'unknown']
BATCH_SIZE = 32
EPOCHS = 350
INPUT_SHAPE = (64, 198, 1)

class Logger(object):

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def train():
    log_file = os.path.join(BASE_DIR, 'training_log.txt')
    sys.stdout = Logger(log_file)
    print('\n' + '=' * 70)
    print('      SILICON-22: CRYNET MVP-OPTIMIZED TRAINING (PCEN)')
    print('=' * 70)
    X = []
    y = []
    for idx, label in enumerate(CLASSES):
        class_dir = os.path.join(PROCESSED_PATH, label)
        if not os.path.exists(class_dir):
            print(f' Warning: {class_dir} not found. Skipping...')
            continue
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.npy')]
        print(f' Loading {label}: {len(files)} samples')
        for f in files:
            feat = np.load(f)
            if feat.shape != INPUT_SHAPE[:2]:
                if feat.shape[0] == INPUT_SHAPE[0]:
                    if feat.shape[1] > INPUT_SHAPE[1]:
                        feat = feat[:, :INPUT_SHAPE[1]]
                    else:
                        feat = np.pad(feat, ((0, 0), (0, INPUT_SHAPE[1] - feat.shape[1])), mode='constant')
                else:
                    continue
            X.append(feat[..., np.newaxis])
            y.append(idx)
    if not X:
        print(' Error: No samples found! Run prepare_data.py first.')
        return
    X = np.array(X, dtype=np.float32)
    y_raw = np.array(y)
    y_cat = tf.keras.utils.to_categorical(y_raw, num_classes=len(CLASSES))
    print('\n Normalizing PCEN features...')
    mean = X.mean(axis=(0, 1, 2), keepdims=True)
    std = X.std(axis=(0, 1, 2), keepdims=True) + 1e-09
    X = (X - mean) / std
    np.save(os.path.join(BASE_DIR, 'normalization_stats.npy'), {'mean': mean, 'std': std})
    idx_train, idx_val = train_test_split(np.arange(len(X)), test_size=0.15, stratify=y_raw, random_state=42)
    x_train, x_val = (X[idx_train], X[idx_val])
    y_train, y_val = (y_cat[idx_train], y_cat[idx_val])
    class_weight = {0: 1.5, 1: 1.0, 2: 1.0}
    print(f' Applied Class Weights: {class_weight}')
    model = create_model(input_shape=INPUT_SHAPE, num_classes=len(CLASSES))
    print(f' Total Parameters: {model.count_params()}')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(class_id=0, name='recall_baby'), tf.keras.metrics.Precision(class_id=0, name='precision_baby')])
    model_save_path = os.path.join(BASE_DIR, 'best_model.keras')
    callbacks = [tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1), tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, mode='min', restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, mode='min', verbose=1)]
    print('\n Starting CryNet Silicon-22 MVP-Optimized Training...')
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=class_weight, callbacks=callbacks, shuffle=True, verbose=2)
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 3, 2)
    plt.plot(history.history['recall_baby'], label='Train Recall')
    plt.plot(history.history['val_recall_baby'], label='Val Recall')
    plt.title('Baby Cry Sensitivity (Recall)')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 3, 3)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    report_path = os.path.join(BASE_DIR, 'training_report.png')
    plt.tight_layout()
    plt.savefig(report_path)
    print(f'\n Training Complete. Silicon-22 model saved to: {model_save_path}')
    print(f' Detailed training metrics saved to: {report_path}')
if __name__ == '__main__':
    train()