import os
import time
import numpy as np
import tensorflow as tf
import pyaudio
from preprocess import AFGPreprocessor

class ProductionInference:

    def __init__(self):
        self.classes = ['baby_cry', 'dog', 'unknown']
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.base_dir, 'int8_model.tflite')
        self.stats_path = os.path.join(self.base_dir, 'normalization_stats.npy')
        if os.path.exists(self.model_path):
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f' SILICON-22 INT8 TFLite Model Loaded: {self.model_path}')
        else:
            print(f' Error: TFLite model not found at {self.model_path}')
            exit(1)
        if os.path.exists(self.stats_path):
            stats = np.load(self.stats_path, allow_pickle=True).item()
            self.norm_mean = stats['mean'].squeeze() if hasattr(stats['mean'], 'squeeze') else stats['mean']
            self.norm_std = stats['std'].squeeze() if hasattr(stats['std'], 'squeeze') else stats['std']
            print(' PCEN Normalization Stats Loaded.')
        else:
            self.norm_mean = 0
            self.norm_std = 1
            print(' Warning: Normalization stats not found. Using defaults.')
        self.prep = AFGPreprocessor()
        self.chunk_size = 4000
        self.sample_rate = 16000
        self.p = pyaudio.PyAudio()
        self.thresholds = {'baby_cry': 0.65, 'dog': 0.9}
        self.streaks = {'baby_cry': 0.0, 'dog': 0.0}
        self.required_streaks = {'baby_cry': 2, 'dog': 4}
        self.decay_rates = {'baby_cry': 1.0, 'dog': 0.3}
        self.baby_suppression_floor = 0.35

    def listen(self):
        stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=self.sample_rate, input=True, frames_per_buffer=self.chunk_size)
        buffer = np.zeros(32000, dtype=np.float32)
        print('\n Monitoring room (SILICON-22: CryNet INT8 + Master Policy V1)')
        print('-' * 75)
        try:
            while True:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                new_samples = np.frombuffer(data, dtype=np.float32)
                buffer = np.roll(buffer, -len(new_samples))
                buffer[-len(new_samples):] = new_samples
                rms = np.sqrt(np.mean(buffer ** 2))
                if rms < 0.002:
                    self.prep.reset_pcen_state()
                    for k in self.streaks:
                        self.streaks[k] = max(0, self.streaks[k] - self.decay_rates[k])
                    continue
                frame_05s = buffer[-8000:]
                env = np.sqrt(np.mean(frame_05s.reshape(-1, 400) ** 2, axis=1))
                slope = np.mean(np.diff(env))
                if slope < 0.0005:
                    self.streaks['dog'] = max(0, self.streaks['dog'] - self.decay_rates['dog'])
                    pass
                feat = self.prep.extract_features(raw_audio=buffer, is_inference=True)
                if feat is None:
                    for k in self.streaks:
                        self.streaks[k] = max(0, self.streaks[k] - self.decay_rates[k])
                    continue
                feat = np.array(feat)
                feat = np.squeeze(feat)
                try:
                    feat = feat.reshape(64, 198)
                except ValueError:
                    print(f' Unexpected feature shape after squeeze: {feat.shape}')
                    continue
                feat = feat[:, :, np.newaxis]
                feat = (feat - self.norm_mean) / (self.norm_std + 1e-09)
                feat_input = feat[np.newaxis, ...].astype(np.float32)
                input_details = self.input_details[0]
                if input_details['dtype'] == np.int8:
                    scale, zero_point = input_details['quantization']
                    feat_input = (feat_input / scale + zero_point).astype(np.int8)
                self.interpreter.set_tensor(input_details['index'], feat_input)
                self.interpreter.invoke()
                output_details = self.output_details[0]
                probs = self.interpreter.get_tensor(output_details['index'])[0]
                if output_details['dtype'] == np.int8:
                    scale, zero_point = output_details['quantization']
                    probs = (probs.astype(np.float32) - zero_point) * scale
                baby_prob = probs[0]
                dog_prob = probs[1]
                if max(baby_prob, dog_prob) < 0.7:
                    for k in self.streaks:
                        self.streaks[k] = max(0, self.streaks[k] - self.decay_rates[k])
                elif baby_prob >= self.thresholds['baby_cry']:
                    self.streaks['baby_cry'] += 1
                    self.streaks['dog'] = max(0, self.streaks['dog'] - self.decay_rates['dog'])
                elif dog_prob >= self.thresholds['dog'] and dog_prob - baby_prob > 0.15 and (baby_prob < self.baby_suppression_floor):
                    self.streaks['dog'] += 1
                    self.streaks['baby_cry'] = max(0, self.streaks['baby_cry'] - self.decay_rates['baby_cry'])
                else:
                    for k in self.streaks:
                        self.streaks[k] = max(0, self.streaks[k] - self.decay_rates[k])
                timestamp = time.strftime('%H:%M:%S')
                b_strk = round(self.streaks['baby_cry'], 1)
                d_strk = round(self.streaks['dog'], 1)
                if self.streaks['baby_cry'] >= self.required_streaks['baby_cry']:
                    meter = '#' * 12
                    print(f'\x1b[92m{timestamp:<15} | BABY CRYING  | Streak: {b_strk:<5} {meter}\x1b[0m')
                elif self.streaks['dog'] >= self.required_streaks['dog']:
                    meter = '#' * 12
                    print(f'\x1b[94m{timestamp:<15} | DOG BARKING  | Streak: {d_strk:<5} {meter}\x1b[0m')
                elif self.streaks['baby_cry'] > 0:
                    meter = '*' * int(self.streaks['baby_cry'] * 4)
                    print(f'{timestamp:<15} | ... baby?     | Streak: {b_strk:<5} {meter}')
                elif self.streaks['dog'] > 0:
                    meter = '*' * int(self.streaks['dog'] * 2)
                    print(f'{timestamp:<15} | ... dog?      | Streak: {d_strk:<5} {meter}')
        except KeyboardInterrupt:
            print('\nMonitor stopped.')
        finally:
            stream.stop_stream()
            stream.close()
            self.p.terminate()
if __name__ == '__main__':
    monitor = ProductionInference()
    monitor.listen()