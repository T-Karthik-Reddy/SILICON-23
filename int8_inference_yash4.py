import os
import time
import numpy as np
import tflite_runtime.interpreter as tflite
import pyaudio
import cv2
import serial
from preprocess import AFGPreprocessor

class AudioSentry:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.audio_model_path = os.path.join(self.base_dir, 'model_int8.tflite')
        self.vision_model_path = os.path.join(self.base_dir, 'vision_model.tflite')
        self.stats_path = os.path.join(self.base_dir, 'normalization_stats.npy')


        try:
            self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
            print("✅ XIAO connected via USB.")
        except Exception as e:
            print(f"⚠️ XIAO Serial Error: {e}. Check cable and permissions.")
            self.ser = None


        self.audio_interpreter = tflite.Interpreter(model_path=self.audio_model_path)
        self.audio_interpreter.allocate_tensors()
        self.a_in = self.audio_interpreter.get_input_details()[0]
        self.a_out = self.audio_interpreter.get_output_details()[0]

        self.vision_interpreter = tflite.Interpreter(model_path=self.vision_model_path)
        self.vision_interpreter.allocate_tensors()
        self.v_in = self.vision_interpreter.get_input_details()[0]
        self.v_out = self.vision_interpreter.get_output_details()[0]


        if os.path.exists(self.stats_path):
            stats = np.load(self.stats_path, allow_pickle=True).item()
            self.norm_mean = stats['mean'].squeeze()
            self.norm_std = stats['std'].squeeze()
            print("✅ Audio normalization stats loaded.")
        else:
            self.norm_mean, self.norm_std = 0.0, 1.0
            print("⚠️ Stats file missing - using defaults.")

        self.prep = AFGPreprocessor()
        

        self.sample_rate = 16000        
        self.chunk_size = 4000         
        self.buffer_samples = 32000    
        self.buffer = np.zeros(self.buffer_samples, dtype=np.float32)


        self.energy_threshold = 0.002
        self.audio_threshold = 0.85     
        self.vision_threshold = 0.80    
        self.streak_needed = 3          
        self.cooldown_duration = 2.5    
        self.streak = 0
        
        self.p = pyaudio.PyAudio()

    def _reset_pcen_state(self):
        
                                                                                    
        if hasattr(self.prep, 'reset_pcen_state'):
            self.prep.reset_pcen_state()
        elif hasattr(self.prep, 'reset'):
            self.prep.reset()
        else:
            self.prep = AFGPreprocessor()

    def calibrate_ambient_noise(self, stream, duration_seconds=2.0):
        
                                                                                                 
        print(f"🎙️ Calibrating noise floor for {duration_seconds}s. Keep environment quiet...")
        chunks_to_read = int((duration_seconds * self.sample_rate) / self.chunk_size)
        rms_values = []

        for _ in range(max(1, chunks_to_read)):
            try:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.float32)
                rms = np.sqrt(np.mean(samples ** 2))
                rms_values.append(rms)
            except Exception:
                continue

        if rms_values:

            stable_ambient_baseline = np.percentile(rms_values, 90)
            calibrated_thresh = float(stable_ambient_baseline * 1.5)
            self.energy_threshold = max(0.002, calibrated_thresh)
        
        print(f"✅ Dynamic noise gate initialized to RMS: {self.energy_threshold:.5f}")

    def run_vision_check(self):
                                                                       
        print("📸 Triggering Camera Check...")
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release() 

        if not ret:
            print("❌ Camera Error: Could not capture frame.")
            return 0.0

        input_h, input_w = self.v_in['shape'][1], self.v_in['shape'][2]
        img = cv2.resize(frame, (input_w, input_h))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        self.vision_interpreter.set_tensor(self.v_in['index'], img)
        self.vision_interpreter.invoke()
        v_output = self.vision_interpreter.get_tensor(self.v_out['index'])
        
        return float(np.max(v_output))

    def _predict_audio(self, feat_input):
                                                                               
        tensor = feat_input.astype(np.float32)
        if self.a_in['dtype'] == np.int8:
            scale, zero = self.a_in['quantization']
            tensor = (tensor / scale + zero).astype(np.int8)

        self.audio_interpreter.set_tensor(self.a_in['index'], tensor)
        self.audio_interpreter.invoke()
        out = self.audio_interpreter.get_tensor(self.a_out['index'])
        
        if self.a_out['dtype'] == np.int8:
            scale, zero = self.a_out['quantization']
            out = (out.astype(np.float32) - zero) * scale
        return float(np.squeeze(out))

    def start_monitoring(self):
        stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=1 
        )

        self.calibrate_ambient_noise(stream)

        print("\n👂 Sentry Mode: Monitoring for Barks...")
        try:
            while True:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                new_samples = np.frombuffer(data, dtype=np.float32)
                
                self.buffer = np.roll(self.buffer, -len(new_samples))
                self.buffer[-len(new_samples):] = new_samples


                rms = np.sqrt(np.mean(self.buffer ** 2))
                if rms < self.energy_threshold:
                    self.streak = max(0, self.streak - 1)
                    self._reset_pcen_state() 
                    continue


                feat = self.prep.extract_features(raw_audio=self.buffer, is_inference=True)
                if feat is None: 
                    self.streak = max(0, self.streak - 1)
                    continue

                feat = np.squeeze(np.array(feat)).reshape(64, 198)
                feat = feat[:, :, np.newaxis]
                feat = (feat - self.norm_mean) / (self.norm_std + 1e-9)
                
                audio_prob = self._predict_audio(feat[np.newaxis, ...])

                if audio_prob >= self.audio_threshold:
                    self.streak += 1
                else:
                    self.streak = max(0, self.streak - 1)

                if self.streak >= self.streak_needed:
                    print(f"\033[92m🔊 BARK DETECTED ({audio_prob:.2f})\033[0m")
                    
                    stream.stop_stream()
                    
                    vision_conf = self.run_vision_check()
                    print(f"👁️ Vision Confidence: {vision_conf:.2f}")

                    if vision_conf >= self.vision_threshold:
                        print("🎯 DOG CONFIRMED: Sending Signal to XIAO!")
                        if self.ser:
                            self.ser.write(b'B')
                    else:
                        print("☁️ Sound detected, but vision check did not confirm dog.")

                    print(f"⏳ Cooldown active for {self.cooldown_duration}s...")
                    time.sleep(self.cooldown_duration)
                    

                    self.buffer.fill(0)
                    self.streak = 0
                    self._reset_pcen_state() 
                    
                    stream.start_stream()

        except KeyboardInterrupt:
            print("\nShutting down.")
        finally:
            stream.stop_stream()
            stream.close()
            self.p.terminate()

if __name__ == '__main__':
    AudioSentry().start_monitoring()