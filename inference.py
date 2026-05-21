

   

import os
import time
import numpy as np
import pyaudio
from preprocess import AFGPreprocessor
from model import create_model


class ProductionInference:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.base_dir, 'best_model.keras')
        self.stats_path = os.path.join(self.base_dir, 'normalization_stats.npy')

        self.model = create_model(input_shape=(64, 198, 1))
        _ = self.model(np.zeros((1, 64, 198, 1), dtype=np.float32))
        if os.path.exists(self.model_path):
            self.model.load_weights(self.model_path)
            print("Model loaded.")
        else:
            print("Warning: model weights not found.")

        if os.path.exists(self.stats_path):
            stats = np.load(self.stats_path, allow_pickle=True).item()
            self.norm_mean = stats['mean'].squeeze()
            self.norm_std = stats['std'].squeeze()
            print("Normalization stats loaded.")
        else:
            self.norm_mean, self.norm_std = 0.0, 1.0
            print("Warning: normalization stats not found.")

        self.prep = AFGPreprocessor()
        self.sample_rate = 16000
        self.chunk_size = 4000
        self.buffer = np.zeros(32000, dtype=np.float32)

        self.energy_threshold = 0.0001
        self.prob_threshold = 0.75
        self.streak_needed = 2
        self.cooldown_sec = 2.5
        self.streak = 0.0
        self.last_trigger_ts = 0.0
        self.debug_print_every_frame = False

        self.p = pyaudio.PyAudio()

    def _auto_calibrate_energy_threshold(self, stream, seconds=3.0):
                                                                                
        num_chunks = max(1, int(seconds * self.sample_rate / self.chunk_size))
        rms_vals = []
        print(f"Calibrating ambient noise for {seconds:.1f}s... stay quiet.")
        try:
            if not stream.is_active():
                stream.start_stream()

            for _ in range(2):
                _ = stream.read(self.chunk_size, exception_on_overflow=False)

            for _ in range(num_chunks):
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                x = np.frombuffer(data, dtype=np.float32)
                rms_vals.append(float(np.sqrt(np.mean(x ** 2) + 1e-12)))
        except Exception as e:
            print(f"Calibration read failed ({e}). Using default energy_threshold={self.energy_threshold:.6f}")
            return

        noise_floor = float(np.median(rms_vals)) if rms_vals else 0.0
        adaptive = max(3.0 * noise_floor, 0.00008)
        self.energy_threshold = adaptive
        print(
            f"Ambient RMS median={noise_floor:.6f} | "
            f"energy_threshold set to {self.energy_threshold:.6f}"
        )

    def listen(self):
        stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        print("\nMonitoring room (SILICON-23 binary dog bark policy)")
        print("-" * 72)
        self._auto_calibrate_energy_threshold(stream, seconds=3.0)

        try:
            while True:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                new_samples = np.frombuffer(data, dtype=np.float32)
                self.buffer = np.roll(self.buffer, -len(new_samples))
                self.buffer[-len(new_samples):] = new_samples

                rms = np.sqrt(np.mean(self.buffer ** 2))
                if rms < self.energy_threshold:
                    self.prep.reset_pcen_state()
                    self.streak = max(0.0, self.streak - 0.5)
                    if self.debug_print_every_frame:
                        ts = time.strftime("%H:%M:%S")
                        print(
                            f"{ts} | rms={rms:.6f} | below energy gate "
                            f"({self.energy_threshold:.6f}) | streak={self.streak}/{self.streak_needed}"
                        )
                    continue

                feat = self.prep.extract_features(raw_audio=self.buffer, is_inference=True)
                if feat is None:
                    self.streak = max(0, self.streak - 1)
                    continue

                feat = np.squeeze(np.array(feat)).reshape(64, 198)
                feat = feat[:, :, np.newaxis]
                feat = (feat - self.norm_mean) / (self.norm_std + 1e-9)

                prob = float(self.model.predict(feat[np.newaxis, ...], verbose=0)[0][0])

                if prob >= self.prob_threshold and rms >= self.energy_threshold:
                    self.streak += 1
                else:
                    self.streak = max(0.0, self.streak - 1.0)

                if self.debug_print_every_frame:
                    ts = time.strftime("%H:%M:%S")
                    print(
                        f"{ts} | rms={rms:.4f} | prob={prob:.3f} | "
                        f"streak={self.streak}/{self.streak_needed} | thr={self.prob_threshold:.2f}"
                    )

                now = time.time()
                can_trigger = (now - self.last_trigger_ts) >= self.cooldown_sec
                if self.streak >= self.streak_needed and can_trigger:
                    self.last_trigger_ts = now
                    ts = time.strftime("%H:%M:%S")
                    print(f"\033[94m{ts} | DOG BARK DETECTED | prob={prob:.2f} streak={self.streak}\033[0m")
                elif self.streak > 0:
                    ts = time.strftime("%H:%M:%S")
                    print(f"\033[96m{ts} | ...dog? prob={prob:.2f} streak={self.streak}\033[0m")

        except KeyboardInterrupt:
            print("\nMonitor stopped.")
        finally:
            stream.stop_stream()
            stream.close()
            self.p.terminate()


if __name__ == '__main__':
    ProductionInference().listen()
