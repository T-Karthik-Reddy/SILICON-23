





   

import numpy as np
import scipy.signal
import soundfile as sf

class AFGPreprocessor:
    def __init__(self):

        self.sample_rate = 16000
        self.sample_len_ms = 2000
        self.window_size_ms = 30
        self.window_step_ms = 10
        self.n_mels = 64
        self.expected_frames = 198
        self.n_fft = 512
        

        self.pcen_alpha = 0.65
        self.pcen_delta = 0.2
        self.pcen_r = 0.30
        self.pcen_eps = 1e-6
        self.pcen_time_constant = 0.5
        

        self.window_samples = int(self.sample_rate * (self.sample_len_ms / 1000))
        self.hop_length = int(self.sample_rate * (self.window_step_ms / 1000))
        self.win_length = int(self.sample_rate * (self.window_size_ms / 1000))
        

        self.pcen_b = 1.0 - np.exp(-self.hop_length / (self.pcen_time_constant * self.sample_rate))
        

        self.mel_basis = self._get_mel_basis(self.sample_rate, self.n_fft, self.n_mels).astype(np.float32)
        self.hann_window = np.hanning(self.win_length).astype(np.float32)
        

        self.pcen_state = None

    def _get_mel_basis(self, sr, n_fft, n_mels):
                                                   
        def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
        def mel_to_hz(mel): return 700.0 * (10.0**(mel / 2595.0) - 1.0)
        
        fmin, fmax = 50.0, 8000.0
        mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
        hz_pts = mel_to_hz(mel_pts)
        
        bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
        basis = np.zeros((n_mels, int(n_fft // 2 + 1)), dtype=np.float32)
        for i in range(n_mels):
            if bin_pts[i+1] > bin_pts[i]:
                for j in range(bin_pts[i], bin_pts[i+1]):
                    basis[i, j] = (j - bin_pts[i]) / (bin_pts[i+1] - bin_pts[i])
            if bin_pts[i+2] > bin_pts[i+1]:
                for j in range(bin_pts[i+1], bin_pts[i+2]):
                    basis[i, j] = (bin_pts[i+2] - j) / (bin_pts[i+2] - bin_pts[i+1])
        return basis.astype(np.float32)

    def _manual_resample(self, y, orig_sr, target_sr):
                                                                  
        if orig_sr == target_sr: return y
        gcd = np.gcd(int(orig_sr), int(target_sr))
        up = int(target_sr // gcd)
        down = int(orig_sr // gcd)
        return scipy.signal.resample_poly(y, up, down)

    def _apply_dc_notch_filter(self, audio):
        b, a = [1, -1], [1, -0.95]
        return scipy.signal.lfilter(b, a, audio)

    def _get_activity_mask(self, y):
                                                                   
        frame_len = int(self.sample_rate * 0.1)
        hop = frame_len // 2
        

        frame_energies = []
        for i in range(0, len(y) - frame_len + 1, hop):
            chunk = y[i : i + frame_len]
            frame_energies.append(np.sqrt(np.mean(chunk**2)))
        frame_energies = np.array(frame_energies)
        
        if len(frame_energies) == 0: return np.zeros(len(y), dtype=bool)
        

        energy_thresh = max(np.median(frame_energies) * 0.5, 0.002)
        
        mask = np.zeros(len(y), dtype=bool)
        idx = 0
        for i in range(0, len(y) - frame_len + 1, hop):
            if frame_energies[idx] > energy_thresh:
                mask[i : i + frame_len] = True
            idx += 1
        return mask

    def _get_windows(self, y, label=None):
                                                                       
        mask = self._get_activity_mask(y)
        

        if len(y) / self.sample_rate < 1.0: return []
        

        if len(y) < self.window_samples:
            y = np.pad(y, (0, self.window_samples - len(y)), mode='constant')
            mask = np.pad(mask, (0, self.window_samples - len(mask)), mode='constant')

        windows = []
        if label == 'baby_cry':

            best_rms = -1
            best_window = None
            step = self.window_samples // 2
            for start in range(0, len(y) - self.window_samples + 1, step):
                w = y[start:start + self.window_samples]
                m = mask[start:start + self.window_samples]
                if np.mean(m) >= 0.3:
                    rms = np.sqrt(np.mean(w**2))
                    if rms > best_rms:
                        best_rms = rms
                        best_window = w
            if best_window is not None: windows.append(best_window)

        elif label in ['dog', 'unknown']:

            candidates = []
            step = self.window_samples // 2
            for start in range(0, len(y) - self.window_samples + 1, step):
                w = y[start:start + self.window_samples]
                m = mask[start:start + self.window_samples]
                if np.mean(m) >= 0.3:
                    rms = np.sqrt(np.mean(w**2))
                    candidates.append((rms, w))
            
            candidates.sort(key=lambda x: x[0], reverse=True)
            for i in range(min(len(candidates), 2)):
                windows.append(candidates[i][1])

        return windows

    def _augment_audio(self, y):
        




           

        gain_db = np.random.uniform(-6, +3)
        y *= 10**(gain_db / 20)


        if np.random.uniform() < 0.4:
            rms = np.sqrt(np.mean(y**2))
            if rms > 1e-4:
                target_snr_db = np.random.uniform(10, 25)
                noise_rms = rms / (10**(target_snr_db / 20))
                y += noise_rms * np.random.normal(size=len(y))
        

        if np.random.uniform() < 0.5:
            max_shift = int(0.05 * self.sample_rate)
            shift = np.random.randint(-max_shift, max_shift)
            y = np.roll(y, shift)
            
        return np.clip(y, -1.0, 1.0)

    def pcen_transform(self, mel_energy, is_inference=False):
        



           
        T, F = mel_energy.shape
        pcen = np.zeros_like(mel_energy)
        

        if is_inference:
            if self.pcen_state is None:
                M = np.zeros((self.n_mels,), dtype=np.float32)
            else:
                M = self.pcen_state.copy()
        else:
            M = np.zeros((self.n_mels,), dtype=np.float32)
            
        for t in range(T):

            M = (1 - self.pcen_b) * M + self.pcen_b * mel_energy[t]
            

            pcen[t] = ((mel_energy[t] / (self.pcen_eps + M)**self.pcen_alpha) + self.pcen_delta)**self.pcen_r - self.pcen_delta**self.pcen_r
            

        if is_inference:
            self.pcen_state = M
            
        return pcen

    def reset_pcen_state(self):
                                                      
        self.pcen_state = None

    def extract_features(self, audio_path=None, raw_audio=None, label=None, augment=False, return_audio=False, is_inference=False):
                                                           
        if audio_path:
            y, sr = sf.read(audio_path, always_2d=False)
            if len(y.shape) > 1: y = np.mean(y, axis=1)
            y = self._manual_resample(y, sr, self.sample_rate)
            audios = self._get_windows(y, label)
            if not audios: return None
        else:
            y = raw_audio

            if len(y) < self.window_samples:
                y = np.pad(y, (0, self.window_samples - len(y)), mode='constant')
            audios = [y[:self.window_samples]]

        results = []
        for a in audios:
            proc = self._process_single_window(a, augment, is_inference)
            if proc:
                if return_audio: results.append(proc)
                else: results.append(proc[0])
        
        if not results: return None
        return results if audio_path else results[0]

    def _process_single_window(self, y, augment=False, is_inference=False):

        y = self._apply_dc_notch_filter(y)
        

        if augment:
            y = self._augment_audio(y)
            


        win_len = self.win_length
        hop_len = self.hop_length
        
        frames = []
        for i in range(0, len(y) - win_len + 1, hop_len):
            chunk = y[i : i + win_len] * self.hann_window

            spec = np.abs(np.fft.rfft(chunk, n=self.n_fft))**2
            mel_spec = np.dot(self.mel_basis, spec)
            frames.append(mel_spec)
        
        mel_energy = np.array(frames).astype(np.float32)
        mel_energy = np.maximum(mel_energy, 1e-10).astype(np.float32)
        

        pcen_spec = self.pcen_transform(mel_energy, is_inference)
        

        if is_inference:

            if np.mean(pcen_spec) < 0.01:
                return None



        pcen_spec = (pcen_spec - np.mean(pcen_spec)) / (np.std(pcen_spec) + 1e-9)
        

        pcen_spec = pcen_spec.T
        
        if pcen_spec.shape[1] > self.expected_frames:
            pcen_spec = pcen_spec[:, :self.expected_frames]
        elif pcen_spec.shape[1] < self.expected_frames:
            pcen_spec = np.pad(pcen_spec, ((0,0), (0, self.expected_frames - pcen_spec.shape[1])), mode='constant')
            
        return pcen_spec, y

if __name__ == '__main__':
    print("🚀 SILICON-21: PCEN-Powered Preprocessor Activated (16kHz, 64-Mel)")
