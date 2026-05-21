

   

import os
import time
import numpy as np
import tensorflow as tf
import pyaudio
from preprocess import AFGPreprocessor


class ProductionInference:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.base_dir, 'model_float32.tflite')
        self.stats_path = os.path.join(self.base_dir, 'normalization_stats.npy')

        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        if os.path.exists(self.stats_path):
            stats = np.load(self.stats_path, allow_pickle=True).item()
            self.norm_mean = stats['mean'].squeeze()
            self.norm_std = stats['std'].squeeze()
        else:
            self.norm_mean, self.norm_std = 0.0, 1.0

        self.prep = AFGPreprocessor()
        self.sample_rate = 16000
        self.chunk_size = 4000
        self.buffer = np.zeros(32000, dtype=np.float32)

        self.energy_threshold = 0.002
        self.prob_threshold = 0.85
        self.streak_needed = 3
        self.cooldown_sec = 2.5
        self.streak = 0
        self.last_trigger_ts = 0.0

        self.p = pyaudio.PyAudio()

    def _predict_prob(self, feat_input):
        tensor = feat_input.astype(np.float32)
        self.interpreter.set_tensor(self.input_details['index'], tensor)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details['index'])
        return float(np.squeeze(out))

    def listen(self):
        stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        print("\nMonitoring room (SILICON-23 float32 TFLite binary dog bark policy)")
        print("-" * 72)

        try:
            while True:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                new_samples = np.frombuffer(data, dtype=np.float32)
                self.buffer = np.roll(self.buffer, -len(new_samples))
                self.buffer[-len(new_samples):] = new_samples

                rms = np.sqrt(np.mean(self.buffer ** 2))
                if rms < self.energy_threshold:
                    self.prep.reset_pcen_state()
                    self.streak = max(0, self.streak - 1)
                    continue

                feat = self.prep.extract_features(raw_audio=self.buffer, is_inference=True)
                if feat is None:
                    self.streak = max(0, self.streak - 1)
                    continue

                feat = np.squeeze(np.array(feat)).reshape(64, 198)
                feat = feat[:, :, np.newaxis]
                feat = (feat - self.norm_mean) / (self.norm_std + 1e-9)
                prob = self._predict_prob(feat[np.newaxis, ...])

                if prob >= self.prob_threshold and rms >= self.energy_threshold:
                    self.streak += 1
                else:
                    self.streak = max(0, self.streak - 1)

                now = time.time()
                can_trigger = (now - self.last_trigger_ts) >= self.cooldown_sec
                if self.streak >= self.streak_needed and can_trigger:
                    self.last_trigger_ts = now
                    ts = time.strftime("%H:%M:%S")
                    print(f"\033[94m{ts} | DOG BARK DETECTED | prob={prob:.2f} streak={self.streak}\033[0m")
                elif self.streak > 0:
                    ts = time.strftime("%H:%M:%S")
                    print(f"{ts} | ...dog? prob={prob:.2f} streak={self.streak}")

        except KeyboardInterrupt:
            print("\nMonitor stopped.")
        finally:
            stream.stop_stream()
            stream.close()
            self.p.terminate()


if __name__ == '__main__':
    ProductionInference().listen()
